"""
Inference Service Singleton.
Combines ISLPredictor, EnglishToTeluguTranslator, and TTSEngine.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from backend.app.core.config import settings
from backend.app.translation.translator import EnglishToTeluguTranslator
from backend.app.tts.engine import TTSEngine
from ml.inference.predictor import ISLPredictor


class InferenceService:
    """Unified service exposing model inference, translation, and TTS."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = InferenceService()
        return cls._instance

    def __init__(self):
        print(f"[InferenceService] Initializing with checkpoint: {settings.CHECKPOINT_PATH} on {settings.DEVICE}")
        self.predictor = ISLPredictor(
            checkpoint_path=settings.CHECKPOINT_PATH,
            vocab_path=settings.VOCAB_PATH,
            device=settings.DEVICE,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )
        self.translator = EnglishToTeluguTranslator()
        self.tts = TTSEngine()

    def translate_sign_frames(self, frames: List[np.ndarray], provisional: bool = False) -> Dict[str, Any]:
        """Runs continuous translation on a batch of RGB frames."""
        pred = self.predictor.predict_frames(frames, provisional=provisional)
        en_text = pred["english_text"]
        is_uncertain = pred["is_uncertain"]

        # Cascade into Telugu
        te_res = self.translator.translate(en_text, is_uncertain=is_uncertain)

        return {
            "english_text": en_text,
            "telugu_text": te_res.telugu_text,
            "confidence": pred["confidence"],
            "is_uncertain": is_uncertain,
            "provisional": provisional,
            "mean_entropy": pred.get("mean_entropy", 0.0),
            "script_valid": te_res.script_valid,
        }

    def translate_pose_sequence(self, pose_seq: np.ndarray, provisional: bool = False) -> Dict[str, Any]:
        """Runs continuous translation directly on a pre-extracted/normalized pose sequence."""
        pred = self.predictor.predict_pose_sequence(pose_seq, provisional=provisional)
        en_text = pred["english_text"]
        is_uncertain = pred["is_uncertain"]

        # Cascade into Telugu
        te_res = self.translator.translate(en_text, is_uncertain=is_uncertain)

        return {
            "english_text": en_text,
            "telugu_text": te_res.telugu_text,
            "confidence": pred["confidence"],
            "is_uncertain": is_uncertain,
            "provisional": provisional,
            "mean_entropy": pred.get("mean_entropy", 0.0),
            "script_valid": te_res.script_valid,
        }
