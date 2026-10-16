"""
Inference Engine and Continuous Streaming Predictor for ISL Translation.
Handles checkpoint restoration, landmark feature processing, calibrated uncertainty estimation,
and temporal streaming window buffer management.
"""

from collections import deque
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from ml.data.pose_extractor import PoseExtractor
from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer


class ISLPredictor:
    """
    High-level continuous inference service for Indian Sign Language.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "ml/checkpoints/isl_transformer_best.pt",
        vocab_path: str = "data/manifests/vocab.json",
        device: Optional[str] = None,
        confidence_threshold: float = 0.55,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold
        self.pose_extractor = PoseExtractor()

        # Load Vocabulary
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab: Dict[str, int] = json.load(f)
        else:
            self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<blank>": 4}

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_idx = self.vocab.get("<pad>", 0)
        self.bos_idx = self.vocab.get("<bos>", 1)
        self.eos_idx = self.vocab.get("<eos>", 2)

        # Inspect checkpoint config if available
        d_model = 256
        nhead = 4
        num_encoder_layers = 4
        num_decoder_layers = 4
        dim_feedforward = 1024
        dropout = 0.1
        max_seq_len = 250
        max_target_len = 60

        checkpoint_data = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
                if "config" in checkpoint_data and "model" in checkpoint_data["config"]:
                    m_cfg = checkpoint_data["config"]["model"]
                    d_model = m_cfg.get("d_model", d_model)
                    nhead = m_cfg.get("nhead", nhead)
                    num_encoder_layers = m_cfg.get("num_encoder_layers", num_encoder_layers)
                    num_decoder_layers = m_cfg.get("num_decoder_layers", num_decoder_layers)
                    dim_feedforward = m_cfg.get("dim_feedforward", dim_feedforward)
                    dropout = m_cfg.get("dropout", dropout)
                    max_seq_len = m_cfg.get("max_seq_len", max_seq_len)
                if "vocab" in checkpoint_data:
                    self.vocab = checkpoint_data["vocab"]
                    self.inv_vocab = {v: k for k, v in self.vocab.items()}
                if "model_state_dict" in checkpoint_data:
                    sd = checkpoint_data["model_state_dict"]
                    if "encoder.pos_encoder.pe" in sd:
                        max_seq_len = sd["encoder.pos_encoder.pe"].shape[1]
                    if "decoder.pos_encoder.pe" in sd:
                        max_target_len = sd["decoder.pos_encoder.pe"].shape[1]
                    if "encoder.input_projection.weight" in sd:
                        d_model = sd["encoder.input_projection.weight"].shape[0]
            except Exception as e:
                print(f"[Inference] Notice: could not pre-parse config from checkpoint: {e}")

        # Build Model Architecture
        self.model = ISLSeq2SeqTransformer(
            vocab_size=len(self.vocab),
            input_dim=225,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            max_target_len=max_target_len,
            pad_idx=self.pad_idx,
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx,
        ).to(self.device)

        self.model_loaded = False
        if checkpoint_data and "model_state_dict" in checkpoint_data:
            try:
                self.model.load_state_dict(checkpoint_data["model_state_dict"])
                self.model_loaded = True
                print(f"[Inference] Successfully loaded trained model weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[Inference] Warning: Failed loading state_dict: {e}")

        self.model.eval()

    def predict_pose_sequence(self, pose_seq: np.ndarray, provisional: bool = False) -> Dict[str, Any]:
        """
        Translates a (T, 225) normalized pose sequence into an English sentence with uncertainty calibration.
        """
        if pose_seq.size == 0 or len(pose_seq) < 5:
            return {
                "english_text": "",
                "confidence": 0.0,
                "is_uncertain": True,
                "provisional": provisional,
                "status": "empty_input",
            }

        # Convert to tensor: (1, T, 225)
        src = torch.tensor(pose_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        src_mask = torch.ones((1, src.size(1)), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            gen = self.model.generate(
                src=src,
                src_mask=src_mask,
                max_length=45,
                confidence_threshold=self.confidence_threshold,
            )

        token_ids = gen["token_ids"]
        raw_words = [self.inv_vocab.get(t, "") for t in token_ids if t in self.inv_vocab]
        
        # Clean text
        english_text = " ".join([w for w in raw_words if w and w not in ("<pad>", "<bos>", "<eos>", "<blank>")])
        # Capitalize and format
        if english_text:
            english_text = english_text.capitalize()
            if not english_text.endswith((".", "?", "!")):
                english_text += "."

        return {
            "english_text": english_text,
            "confidence": gen["confidence"],
            "is_uncertain": gen["is_uncertain"],
            "mean_entropy": gen["mean_entropy"],
            "provisional": provisional,
            "status": "success",
        }

    def predict_frames(self, frames: List[np.ndarray], provisional: bool = False) -> Dict[str, Any]:
        """
        Extracts holistic landmarks from RGB frames and returns translation.
        """
        if not frames:
            return {
                "english_text": "",
                "confidence": 0.0,
                "is_uncertain": True,
                "provisional": provisional,
                "status": "no_frames",
            }

        pose_features = self.pose_extractor.process_frames(frames, normalize=True)
        return self.predict_pose_sequence(pose_features, provisional=provisional)


class StreamingWindowBuffer:
    """
    Temporal sliding window buffer for continuous live video stream translation.
    Maintains temporal context across frames, provides provisional translations,
    and detects discourse pauses/sentence boundaries.
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 15,
        rest_threshold_frames: int = 20,
    ):
        self.window_size = window_size
        self.stride = stride
        self.rest_threshold_frames = rest_threshold_frames
        self.buffer: deque = deque(maxlen=window_size)
        self.frames_since_last_inference = 0

    def add_frame(self, frame_features: np.ndarray) -> bool:
        """Adds a frame feature vector (225-dim). Returns True if window is ready for inference."""
        self.buffer.append(frame_features)
        self.frames_since_last_inference += 1

        if len(self.buffer) >= 30 and self.frames_since_last_inference >= self.stride:
            self.frames_since_last_inference = 0
            return True
        return False

    def get_current_window(self) -> np.ndarray:
        """Returns the current window as a numpy array of shape (T, 225)."""
        return np.array(list(self.buffer), dtype=np.float32)

    def clear(self):
        self.buffer.clear()
        self.frames_since_last_inference = 0
