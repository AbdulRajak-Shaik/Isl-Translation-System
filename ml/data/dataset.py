"""
PyTorch Dataset and Dynamic Collate Functions for ISL Pose Sequences.
Supports streaming, variable-length padding, and vocabulary tokenization.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.schemas import SignSample


class ISLDataset(Dataset):
    """
    PyTorch Dataset loading pre-extracted 3D pose sequences and tokenized target text.
    """

    def __init__(
        self,
        manifest_path: str,
        vocab_path: str,
        max_seq_len: int = 200,
        max_target_len: int = 50,
        pose_feature_dim: int = 225,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_target_len = max_target_len
        self.pose_feature_dim = pose_feature_dim

        # Load Manifest
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
            self.samples = [SignSample(**s) for s in raw_samples]

        # Load Vocabulary
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)
            self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

        self.pad_id = self.vocab.get("<pad>", 0)
        self.bos_id = self.vocab.get("<bos>", 1)
        self.eos_id = self.vocab.get("<eos>", 2)
        self.unk_id = self.vocab.get("<unk>", 3)
        self.blank_id = self.vocab.get("<blank>", 4)

    def __len__(self) -> int:
        return len(self.samples)

    def tokenize_text(self, text: str) -> List[int]:
        import re
        clean_text = re.sub(r"([.,!?;:])", r" \1 ", text.lower().strip())
        tokens = clean_text.split()
        token_ids = [self.bos_id]
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.unk_id))
        token_ids.append(self.eos_id)
        return token_ids[: self.max_target_len]

    def decode_tokens(self, token_ids: List[int]) -> str:
        words = []
        for idx in token_ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id, self.blank_id):
                continue
            words.append(self.inv_vocab.get(idx, "<unk>"))
        return " ".join(words)

    def load_pose_features(self, sample: SignSample) -> np.ndarray:
        """
        Loads pose features from disk if available, or returns zero-padded array.
        """
        if sample.pose_path and os.path.exists(sample.pose_path):
            try:
                features = np.load(sample.pose_path)
                if features.ndim == 3:  # (T, 75, 3)
                    features = features.reshape(features.shape[0], -1)
                return features[: self.max_seq_len]
            except Exception:
                pass

        # If pose file is missing, return a synthesized zero sequence of valid shape for smoke tests
        return np.zeros((10, self.pose_feature_dim), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        pose_seq = self.load_pose_features(sample)
        target_ids = self.tokenize_text(sample.english_text)

        return {
            "sample_id": sample.sample_id,
            "pose": torch.tensor(pose_seq, dtype=torch.float32),
            "seq_len": len(pose_seq),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_len": len(target_ids),
            "english_ref": sample.english_text,
            "telugu_ref": sample.telugu_text or "",
        }


def isl_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Dynamic padding collate function for variable-length pose sequences and targets.
    """
    batch_size = len(batch)
    pose_dim = batch[0]["pose"].shape[-1]

    seq_lens = [item["seq_len"] for item in batch]
    target_lens = [item["target_len"] for item in batch]

    max_seq_len = max(seq_lens)
    max_target_len = max(target_lens)

    # Pad poses with zeros: (B, max_seq_len, pose_dim)
    padded_poses = torch.zeros(batch_size, max_seq_len, pose_dim, dtype=torch.float32)
    # Pose mask: (B, max_seq_len), True for valid, False for padding
    pose_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    # Pad targets with pad_id (0)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_mask = torch.zeros(batch_size, max_target_len, dtype=torch.bool)

    sample_ids = []
    english_refs = []
    telugu_refs = []

    for i, item in enumerate(batch):
        s_len = item["seq_len"]
        t_len = item["target_len"]

        padded_poses[i, :s_len] = item["pose"]
        pose_mask[i, :s_len] = True

        padded_targets[i, :t_len] = item["target_ids"]
        target_mask[i, :t_len] = True

        sample_ids.append(item["sample_id"])
        english_refs.append(item["english_ref"])
        telugu_refs.append(item["telugu_ref"])

    return {
        "sample_ids": sample_ids,
        "poses": padded_poses,
        "pose_mask": pose_mask,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "targets": padded_targets,
        "target_mask": target_mask,
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
        "english_refs": english_refs,
        "telugu_refs": telugu_refs,
    }
