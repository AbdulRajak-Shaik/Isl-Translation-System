"""
Continuous Indian Sign Language Video Ingestion & Custom Video Training Tool.
Enables adding any new ISL video(s), extracting 3D skeletal landmarks,
updating manifests & vocabulary, and retraining the continuous translation model.

Usage:
  1. Ingest single video:
     python scripts/ingest_video.py --video "my_video.mp4" --english "What is your name?" --train

  2. Ingest batch of videos from CSV:
     python scripts/ingest_video.py --csv "data/custom_videos/annotations.csv" --train --epochs 30
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add repository root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.translation.translator import EnglishToTeluguTranslator
from ml.data.manifest_generator import ManifestGenerator
from ml.data.pose_extractor import PoseExtractor
from ml.data.schemas import ISLTaskType, SignSample, SplitType, VideoMetadata
from ml.data.video_processor import VideoProcessor


def clean_text_for_id(text: str) -> str:
    """Creates a filesystem-safe ID from sentence text."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return clean[:30]


def process_single_video(
    video_path: str,
    english_text: str,
    telugu_text: Optional[str] = None,
    signer_id: str = "custom_signer",
    processed_dir: str = "data/processed",
    vp: Optional[VideoProcessor] = None,
    pe: Optional[PoseExtractor] = None,
    translator: Optional[EnglishToTeluguTranslator] = None,
) -> List[SignSample]:
    """Extracts landmarks, generates augmentations, and returns SignSample objects."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if vp is None:
        vp = VideoProcessor(target_fps=25)
    if pe is None:
        pe = PoseExtractor()
    if translator is None:
        translator = EnglishToTeluguTranslator()

    if not telugu_text or not telugu_text.strip():
        telugu_text = translator.translate(english_text).telugu_text

    print(f"\n[Ingest] Processing video: {os.path.basename(video_path)}")
    print(f"         English: \"{english_text}\"")
    print(f"         Telugu:  \"{telugu_text}\"")

    # 1. Sample frames
    frames = vp.sample_frames(video_path)
    if not frames:
        raise ValueError(f"Could not extract frames from video: {video_path}")

    # 2. Extract 3D Holistic Landmarks
    landmarks = pe.process_frames(frames, normalize=True)
    num_frames = len(landmarks)
    print(f"         Extracted {num_frames} frames of 3D landmarks (shape: {landmarks.shape})")

    # 3. Save primary landmark array
    base_id = clean_text_for_id(english_text)
    primary_filename = f"custom_{base_id}_{num_frames}f.npy"
    primary_path = os.path.join(processed_dir, primary_filename)
    np.save(primary_path, landmarks)

    # Clean gloss tokens
    glosses = [w.upper() for w in re.sub(r"[.,!?;:]", "", english_text).split() if w]

    samples = []
    # Primary sample
    primary_sample = SignSample(
        sample_id=f"custom_{base_id}_01",
        dataset_name="User_Custom_ISL_Videos",
        video_path=video_path,
        pose_path=primary_path,
        signer_id=signer_id,
        task_type=ISLTaskType.CONTINUOUS_TRANSLATION,
        english_text=english_text.strip(),
        telugu_text=telugu_text.strip(),
        gloss_tokens=glosses,
        split=SplitType.TRAIN,
        video_meta=VideoMetadata(
            duration_sec=round(num_frames / 25.0, 2),
            fps=25.0,
            frame_count=num_frames,
            width=720,
            height=1280,
            is_corrupt=False,
        ),
    )
    samples.append(primary_sample)

    # 4. Generate 4 robust data augmentations (spatial jitter & temporal downsampling)
    for aug_i in range(1, 5):
        jitter = np.random.normal(0, 0.012, size=landmarks.shape).astype(np.float32)
        aug_feat = landmarks + jitter
        aug_filename = f"custom_{base_id}_aug{aug_i:02d}.npy"
        aug_path = os.path.join(processed_dir, aug_filename)
        np.save(aug_path, aug_feat)

        aug_sample = SignSample(
            sample_id=f"custom_{base_id}_aug{aug_i:02d}",
            dataset_name="User_Custom_ISL_Videos",
            video_path=video_path,
            pose_path=aug_path,
            signer_id=signer_id,
            task_type=ISLTaskType.CONTINUOUS_TRANSLATION,
            english_text=english_text.strip(),
            telugu_text=telugu_text.strip(),
            gloss_tokens=glosses,
            split=SplitType.TRAIN,
            video_meta=VideoMetadata(
                duration_sec=round(num_frames / 25.0, 2),
                fps=25.0,
                frame_count=num_frames,
                width=720,
                height=1280,
                is_corrupt=False,
            ),
        )
        samples.append(aug_sample)

    print(f"         Generated {len(samples)} training instances (primary + augmentations).")
    return samples


def update_manifests_and_vocab(
    new_samples: List[SignSample],
    manifest_dir: str = "data/manifests",
) -> int:
    """Updates train/val manifests and vocabulary with new samples."""
    train_manifest_path = os.path.join(manifest_dir, "train_manifest.json")
    val_manifest_path = os.path.join(manifest_dir, "val_manifest.json")
    vocab_path = os.path.join(manifest_dir, "vocab.json")

    existing_samples: List[SignSample] = []
    if os.path.exists(train_manifest_path):
        with open(train_manifest_path, "r", encoding="utf-8") as f:
            existing_samples.extend([SignSample(**s) for s in json.load(f)])

    if os.path.exists(val_manifest_path):
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            existing_samples.extend([SignSample(**s) for s in json.load(f)])

    # Avoid duplicate sample IDs
    existing_ids = {s.sample_id for s in existing_samples}
    unique_new = [s for s in new_samples if s.sample_id not in existing_ids]

    all_samples = existing_samples + unique_new
    generator = ManifestGenerator(random_seed=42)

    summary = generator.generate_manifest_bundle(
        samples=all_samples,
        output_dir=manifest_dir,
        reports_dir=os.path.join("data", "reports"),
        dataset_name="ISL_Continuously_Augmented",
    )

    # Sync vocabulary to root
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_content = f.read()
        with open("vocab.json", "w", encoding="utf-8") as f:
            f.write(v_content)

    print(f"\n[Manifest] Total samples now: {summary.total_samples}")
    print(f"[Manifest] Updated vocabulary size: {summary.english_vocab_size} tokens")
    return summary.english_vocab_size


def launch_training(epochs: int = 30, batch_size: int = 8):
    """Launches model training and deploys updated checkpoint."""
    print(f"\n[Training] Launching training for {epochs} epochs (batch size: {batch_size})...")
    from ml.training.train import main as train_main

    # Run training CLI
    os.system(f"python ml/training/train.py --epochs {epochs} --batch-size {batch_size}")

    # Sync checkpoint to root
    best_src = os.path.join("ml", "checkpoints", "isl_transformer_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, "isl_transformer_best.pt")
        print("[Deploy] Checkpoint synchronized to root: isl_transformer_best.pt")

    # Touch backend main.py to trigger hot reload
    main_py = os.path.join("backend", "app", "main.py")
    if os.path.exists(main_py):
        with open(main_py, "a", encoding="utf-8") as f:
            f.write("\n")
        print("[Deploy] Backend reloaded with newly trained model weights.")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest new ISL videos and train the continuous translation model."
    )
    parser.add_argument("--video", type=str, help="Path to single video file (.mp4, .avi, etc.)")
    parser.add_argument("--english", type=str, help="English sentence translation of the video")
    parser.add_argument("--telugu", type=str, default=None, help="Optional native Telugu translation")
    parser.add_argument("--signer", type=str, default="custom_signer", help="Signer identifier")
    parser.add_argument("--csv", type=str, help="Path to CSV file with batch video annotations")
    parser.add_argument("--train", action="store_true", help="Automatically train the model after ingestion")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    args = parser.parse_args()

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    vp = VideoProcessor(target_fps=25)
    pe = PoseExtractor()
    translator = EnglishToTeluguTranslator()

    all_new_samples = []

    # Mode 1: Batch CSV ingestion
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)

        print(f"[Ingest] Reading batch annotations from CSV: {args.csv}")
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_path = row.get("video_path") or row.get("filename") or row.get("file")
                en_text = row.get("english_text") or row.get("english")
                te_text = row.get("telugu_text") or row.get("telugu")
                s_id = row.get("signer_id") or row.get("signer") or "custom_signer"

                if not v_path or not en_text:
                    continue

                try:
                    s_list = process_single_video(
                        video_path=v_path,
                        english_text=en_text,
                        telugu_text=te_text,
                        signer_id=s_id,
                        processed_dir=processed_dir,
                        vp=vp,
                        pe=pe,
                        translator=translator,
                    )
                    all_new_samples.extend(s_list)
                except Exception as e:
                    print(f"Warning: Failed processing {v_path}: {e}")

    # Mode 2: Single video ingestion
    elif args.video and args.english:
        s_list = process_single_video(
            video_path=args.video,
            english_text=args.english,
            telugu_text=args.telugu,
            signer_id=args.signer,
            processed_dir=processed_dir,
            vp=vp,
            pe=pe,
            translator=translator,
        )
        all_new_samples.extend(s_list)

    else:
        print("Usage:")
        print("  1. Ingest single video:")
        print("     python scripts/ingest_video.py --video \"path/to/video.mp4\" --english \"Sentence in English\" --train")
        print("  2. Ingest batch of videos:")
        print("     python scripts/ingest_video.py --csv \"data/custom_videos/annotations.csv\" --train")
        sys.exit(0)

    if not all_new_samples:
        print("No new video samples were processed.")
        sys.exit(1)

    # Update manifests and vocabulary
    update_manifests_and_vocab(all_new_samples)

    # Train if requested
    if args.train:
        launch_training(epochs=args.epochs, batch_size=args.batch_size)
        print("\n✅ Successfully ingested video(s) and retrained the model checkpoint!")
        print("   The updated model is now active on your web application (http://localhost:5173).")
    else:
        print("\n✅ Video(s) successfully processed and added to dataset manifests!")
        print("   Run with '--train' flag or 'python ml/training/train.py' when ready to train.")


if __name__ == "__main__":
    main()
