"""
Asynchronous Video Training Job Manager.
Orchestrates custom video ingestion, landmark feature extraction,
manifest & vocabulary synchronization, background model training,
and post-training test verification directly from the web interface.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.app.core.config import settings
from backend.app.schemas.api_models import TrainJobStatusResponse
from backend.app.services.inference_service import InferenceService
from backend.app.translation.translator import EnglishToTeluguTranslator
from ml.data.dataset import ISLDataset, isl_collate_fn
from ml.data.manifest_generator import ManifestGenerator
from ml.data.pose_extractor import PoseExtractor
from ml.data.schemas import ISLTaskType, SignSample, SplitType, VideoMetadata
from ml.data.video_processor import VideoProcessor
from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer
from ml.training.trainer import ISLTrainer


class TrainingJobManager:
    """Manages asynchronous training jobs for uploaded video files."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "TrainingJobManager":
        if cls._instance is None:
            cls._instance = TrainingJobManager()
        return cls._instance

    def __init__(self):
        self.jobs: Dict[str, TrainJobStatusResponse] = {}
        self.video_processor = VideoProcessor(target_fps=25)
        self.pose_extractor = PoseExtractor()
        self.translator = EnglishToTeluguTranslator()
        self.executor = ThreadPoolExecutor(max_workers=1)  # 1 active training job at a time

    def create_job(self, filename: str, english_text: str, telugu_text: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = TrainJobStatusResponse(
            job_id=job_id,
            status="pending",
            progress=0.0,
            stage="Queued for processing",
            english_ground_truth=english_text,
            telugu_ground_truth=telugu_text,
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainJobStatusResponse]:
        return self.jobs.get(job_id)

    def _clean_text_id(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")[:25]

    def _process_train_job_sync(
        self,
        job_id: str,
        video_path: str,
        english_text: str,
        telugu_text: Optional[str],
        signer_id: str,
        epochs: int,
        batch_size: int,
    ):
        try:
            job = self.jobs[job_id]
            job.status = "processing"
            job.total_epochs = epochs

            # 1. Inspect Video
            job.stage = "Inspecting video integrity and codecs..."
            job.progress = 0.05
            meta = self.video_processor.inspect_video(video_path)
            if meta.is_corrupt:
                job.status = "failed"
                job.error = "Video file is corrupt or has an unsupported codec."
                return

            # Auto-translate Telugu if omitted
            if not telugu_text or not telugu_text.strip():
                telugu_text = self.translator.translate(english_text).telugu_text
            job.telugu_ground_truth = telugu_text

            # 2. Extract 3D Landmarks (Streaming, thread-isolated, low memory)
            job.stage = "Extracting 3D holistic landmarks (hands, face, torso)..."
            job.progress = 0.20
            extractor = PoseExtractor(model_complexity=0)
            try:
                landmarks = extractor.extract_from_video(
                    video_path,
                    target_fps=25,
                    max_frames=300,
                    max_dimension=480,
                    normalize=True,
                )
            finally:
                extractor.close()

            if landmarks is None or len(landmarks) == 0:
                job.status = "failed"
                job.error = "Could not extract landmark features from video."
                return
            num_frames = len(landmarks)

            # Save primary landmarks
            processed_dir = "data/processed"
            os.makedirs(processed_dir, exist_ok=True)
            base_id = f"{self._clean_text_id(english_text)}_{int(time.time())}"
            primary_pose_path = os.path.join(processed_dir, f"custom_{base_id}.npy")
            np.save(primary_pose_path, landmarks)

            # 4. Generate Augmentations & Update Manifests
            job.stage = "Generating spatial & temporal augmentations..."
            job.progress = 0.35
            glosses = [w.upper() for w in re.sub(r"[.,!?;:]", "", english_text).split() if w]

            new_samples = [
                SignSample(
                    sample_id=f"custom_{base_id}_orig",
                    dataset_name="User_Custom_UI_Videos",
                    video_path=video_path,
                    pose_path=primary_pose_path,
                    signer_id=signer_id,
                    task_type=ISLTaskType.CONTINUOUS_TRANSLATION,
                    english_text=english_text.strip(),
                    telugu_text=telugu_text.strip(),
                    gloss_tokens=glosses,
                    split=SplitType.TRAIN,
                    video_meta=meta,
                )
            ]

            # 4 spatial augmentations
            for aug_i in range(1, 5):
                jitter = np.random.normal(0, 0.012, size=landmarks.shape).astype(np.float32)
                aug_feat = landmarks + jitter
                aug_pose_path = os.path.join(processed_dir, f"custom_{base_id}_aug{aug_i}.npy")
                np.save(aug_pose_path, aug_feat)
                new_samples.append(
                    SignSample(
                        sample_id=f"custom_{base_id}_aug{aug_i}",
                        dataset_name="User_Custom_UI_Videos",
                        video_path=video_path,
                        pose_path=aug_pose_path,
                        signer_id=signer_id,
                        task_type=ISLTaskType.CONTINUOUS_TRANSLATION,
                        english_text=english_text.strip(),
                        telugu_text=telugu_text.strip(),
                        gloss_tokens=glosses,
                        split=SplitType.TRAIN,
                        video_meta=meta,
                    )
                )

            # Update manifests
            manifest_dir = "data/manifests"
            train_manifest_path = os.path.join(manifest_dir, "train_manifest.json")
            val_manifest_path = os.path.join(manifest_dir, "val_manifest.json")
            vocab_path = os.path.join(manifest_dir, "vocab.json")

            existing_samples: List[SignSample] = []
            for split in ["train", "val", "test"]:
                manifest_file = os.path.join(manifest_dir, f"{split}_manifest.json")
                if os.path.exists(manifest_file):
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        existing_samples.extend([SignSample(**s) for s in json.load(f)])

            def _norm_txt(t: str) -> str:
                return re.sub(r"[^\w\s]", "", (t or "").lower()).strip()

            target_norm = _norm_txt(english_text)

            # Keep at most 5 samples per other distinct sentence class, and replace old versions of current sentence
            filtered_existing: List[SignSample] = []
            seen_counts: Dict[str, int] = {}
            for s in existing_samples:
                s_norm = _norm_txt(s.english_text)
                if s_norm == target_norm:
                    continue
                count = seen_counts.get(s_norm, 0)
                if count < 5:
                    filtered_existing.append(s)
                    seen_counts[s_norm] = count + 1

            all_samples = filtered_existing + new_samples

            generator = ManifestGenerator(random_seed=42)
            summary = generator.generate_manifest_bundle(
                samples=all_samples,
                output_dir=manifest_dir,
                reports_dir=os.path.join("data", "reports"),
                dataset_name="ISL_Continuously_Augmented",
            )

            # Sync root vocab.json
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            with open("vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, indent=2)

            # 5. Model Training Loop
            job.stage = f"Initializing Seq2Seq Transformer (Vocab: {len(vocab_data)} tokens)..."
            job.progress = 0.40

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_dataset = ISLDataset(
                manifest_path=train_manifest_path,
                vocab_path=vocab_path,
                max_seq_len=500,
                max_target_len=60,
            )
            val_dataset = ISLDataset(
                manifest_path=val_manifest_path,
                vocab_path=vocab_path,
                max_seq_len=500,
                max_target_len=60,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=isl_collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=isl_collate_fn,
            )

            model = ISLSeq2SeqTransformer(
                vocab_size=len(vocab_data),
                input_dim=225,
                d_model=256,
                nhead=4,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                max_seq_len=500,
                max_target_len=60,
                pad_idx=vocab_data.get("<pad>", 0),
                bos_idx=vocab_data.get("<bos>", 1),
                eos_idx=vocab_data.get("<eos>", 2),
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0006,
                weight_decay=0.0001,
            )

            trainer_cfg = {
                "model": {
                    "input_dim": 225,
                    "d_model": 256,
                    "nhead": 4,
                    "num_encoder_layers": 4,
                    "num_decoder_layers": 4,
                    "dim_feedforward": 1024,
                    "dropout": 0.1,
                    "max_seq_len": 500,
                    "max_target_len": 60,
                },
                "training": {
                    "save_dir": "ml/checkpoints",
                    "max_epochs": epochs,
                    "accumulate_grad_batches": 1,
                    "gradient_clip_val": 1.0,
                },
                "loss": {
                    "ctc_weight": 0.3,
                    "label_smoothing": 0.1,
                },
            }

            trainer = ISLTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                vocab=vocab_data,
                config=trainer_cfg,
                device=device,
                save_dir="ml/checkpoints",
            )

            # Train with progress updates
            best_bleu = 0.0
            best_loss = float("inf")
            saved_any = False
            for epoch in range(1, epochs + 1):
                train_res = trainer.train_epoch(epoch)
                val_res = trainer.evaluate()

                bleu_4 = float(val_res.get("bleu_4", 0.0))
                loss = float(train_res.get("loss", 999.0))

                is_best = False
                if loss < best_loss:
                    best_loss = loss
                    is_best = True
                elif bleu_4 > best_bleu:
                    best_bleu = bleu_4
                    is_best = True

                if is_best:
                    saved_any = True
                    trainer.save_checkpoint("isl_transformer_best.pt", epoch, val_res)

                trainer.save_checkpoint("isl_transformer_last.pt", epoch, val_res)

                job.current_epoch = epoch
                job.loss = round(loss, 4)
                job.bleu = round(bleu_4, 2)
                # Progress spans 0.40 to 0.90
                job.total_epochs = epochs
                job.progress = round(0.40 + 0.50 * (epoch / epochs), 2)
                job.stage = f"Training Epoch {epoch}/{epochs} (Loss: {job.loss:.2f}, BLEU: {job.bleu:.1f})"

            if not saved_any:
                trainer.save_checkpoint("isl_transformer_best.pt", epochs, val_res)

            # Sync checkpoint to root
            checkpoint_path = "ml/checkpoints/isl_transformer_best.pt"
            shutil.copy2(checkpoint_path, "isl_transformer_best.pt")

            # 6. Test Model on This Video
            job.stage = "Testing newly trained model on your video..."
            job.progress = 0.95

            # Re-initialize InferenceService with new weights
            InferenceService._instance = None
            service = InferenceService.get_instance()
            test_pred = service.translate_pose_sequence(landmarks, provisional=False)

            # Generate TTS audio for test results
            audio_en = service.tts.generate_audio_base64(test_pred["english_text"], lang="en")
            audio_te = service.tts.generate_audio_base64(test_pred["telugu_text"], lang="te")

            job.test_result = {
                "english_text": test_pred["english_text"],
                "telugu_text": test_pred["telugu_text"],
                "confidence": test_pred["confidence"],
                "is_uncertain": test_pred["is_uncertain"],
                "audio_en": audio_en.get("audio_base64"),
                "audio_te": audio_te.get("audio_base64"),
            }

            # 7. Complete Job
            job.status = "completed"
            job.progress = 1.0
            job.stage = "Model training, verification, and deployment completed successfully!"

        except Exception as e:
            job = self.jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = f"Training failed: {str(e)}"
        finally:
            # Privacy clean-up of temp raw video if configured
            if settings.AUTO_DELETE_PROCESSED_VIDEOS and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

    async def submit_train_job(
        self,
        job_id: str,
        video_path: str,
        english_text: str,
        telugu_text: Optional[str] = None,
        signer_id: str = "custom_signer",
        epochs: int = 20,
        batch_size: int = 8,
    ):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self.executor,
            self._process_train_job_sync,
            job_id,
            video_path,
            english_text,
            telugu_text,
            signer_id,
            epochs,
            batch_size,
        )
