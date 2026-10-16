"""
Unit Tests for Data Preparation, Normalization, and Manifest Generation.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from ml.data.schemas import ISLTaskType, SignSample, SplitType, VideoMetadata
from ml.data.pose_extractor import PoseExtractor
from ml.data.manifest_generator import ManifestGenerator


class TestDataPipeline:

    def test_pose_normalization_invariance(self):
        """Verifies that mid-shoulder centering and scale normalization work as intended."""
        # Create synthetic sequence of shape (10, 75, 3)
        seq = np.random.normal(10.0, 2.0, (10, 75, 3)).astype(np.float32)
        # Set left shoulder (idx 53) and right shoulder (idx 54) to known positions
        seq[:, 53] = [10.0, 10.0, 0.0]
        seq[:, 54] = [12.0, 10.0, 0.0]

        normalized = PoseExtractor.normalize_keypoints(seq)
        assert normalized.shape == (10, 75, 3)

        # After normalization, midpoint between shoulders should be at origin (0, 0, 0)
        mid = (normalized[:, 53] + normalized[:, 54]) / 2.0
        np.testing.assert_allclose(mid, 0.0, atol=1e-5)

        # Shoulder distance should normalize to 1.0
        dist = np.linalg.norm(normalized[:, 53] - normalized[:, 54], axis=-1)
        np.testing.assert_allclose(dist, 1.0, atol=1e-5)

    def test_leakage_detection(self):
        """Ensures that signer leakage across partitions is caught."""
        s1 = SignSample(
            sample_id="s1",
            dataset_name="Test",
            signer_id="signer_A",
            english_text="Hello",
            split=SplitType.TRAIN,
        )
        s2 = SignSample(
            sample_id="s2",
            dataset_name="Test",
            signer_id="signer_B",
            english_text="Doctor",
            split=SplitType.VAL,
        )
        s3_leaked = SignSample(
            sample_id="s3",
            dataset_name="Test",
            signer_id="signer_A",  # Leaked into Val!
            english_text="Hospital",
            split=SplitType.VAL,
        )

        passed_clean, _ = ManifestGenerator.audit_leakage([s1], [s2], [])
        assert passed_clean is True

        passed_leaked, violations = ManifestGenerator.audit_leakage([s1], [s3_leaked], [])
        assert passed_leaked is False
        assert len(violations) > 0

    def test_vocabulary_construction(self):
        """Tests vocabulary mapping with standard special tokens."""
        generator = ManifestGenerator(random_seed=42)
        samples = [
            SignSample(sample_id="1", dataset_name="T", english_text="Where is the hospital?", split=SplitType.TRAIN),
            SignSample(sample_id="2", dataset_name="T", english_text="Good morning, doctor.", split=SplitType.TRAIN),
        ]
        vocab = generator.build_vocabulary(samples)

        assert "<pad>" in vocab
        assert "<bos>" in vocab
        assert "<eos>" in vocab
        assert "<unk>" in vocab
        assert "<blank>" in vocab
        assert "hospital" in vocab
        assert "doctor" in vocab
