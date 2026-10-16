"""
Manifest Generator and Dataset Audit Tool.
Enforces signer-disjoint partitioning, audits vocabulary/annotations,
detects data leakage, and outputs versioned manifests and audit reports.
"""

from collections import Counter
from datetime import datetime
import json
import os
import random
import re
from typing import Dict, List, Optional, Set, Tuple

from ml.data.schemas import ManifestSummary, SignSample, SplitType


class ManifestGenerator:
    """Manages dataset partitioning, leakage prevention, and manifest exports."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    @staticmethod
    def audit_leakage(
        train_samples: List[SignSample],
        val_samples: List[SignSample],
        test_samples: List[SignSample],
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that no signer ID or exact video source appears across splits.
        Returns (passed, list_of_violations).
        """
        violations = []

        # 1. Signer Disjoint Check
        train_signers = {
            s.signer_id
            for s in train_samples
            if s.signer_id and s.signer_id != "unknown"
        }
        val_signers = {
            s.signer_id
            for s in val_samples
            if s.signer_id and s.signer_id != "unknown"
        }
        test_signers = {
            s.signer_id
            for s in test_samples
            if s.signer_id and s.signer_id != "unknown"
        }

        val_overlap = train_signers.intersection(val_signers)
        if val_overlap:
            violations.append(f"Signer leakage between Train and Val: {val_overlap}")

        test_overlap = train_signers.intersection(test_signers)
        if test_overlap:
            violations.append(f"Signer leakage between Train and Test: {test_overlap}")

        # 2. Source Video Path Check
        train_videos = {s.video_path for s in train_samples if s.video_path}
        val_videos = {s.video_path for s in val_samples if s.video_path}
        test_videos = {s.video_path for s in test_samples if s.video_path}

        vid_val_overlap = train_videos.intersection(val_videos)
        if vid_val_overlap:
            violations.append(f"Video path duplicate between Train and Val: {len(vid_val_overlap)} files")

        vid_test_overlap = train_videos.intersection(test_videos)
        if vid_test_overlap:
            violations.append(f"Video path duplicate between Train and Test: {len(vid_test_overlap)} files")

        passed = len(violations) == 0
        return passed, violations

    def partition_by_signer(
        self,
        samples: List[SignSample],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[SignSample], List[SignSample], List[SignSample]]:
        """
        Partition samples ensuring signer disjointness whenever signer_id is available.
        """
        signer_groups: Dict[str, List[SignSample]] = {}
        unknown_samples: List[SignSample] = []

        for sample in samples:
            if sample.signer_id and sample.signer_id != "unknown":
                signer_groups.setdefault(sample.signer_id, []).append(sample)
            else:
                unknown_samples.append(sample)

        train_samples: List[SignSample] = []
        val_samples: List[SignSample] = []
        test_samples: List[SignSample] = []

        # If signers are known and there are enough distinct signers (>= 6), distribute distinct signers across partitions
        if len(signer_groups) >= 6:
            signers = list(signer_groups.keys())
            random.shuffle(signers)

            n_test_signers = max(1, round(len(signers) * test_ratio))
            n_val_signers = max(1, round(len(signers) * val_ratio))

            test_signers = set(signers[:n_test_signers])
            val_signers = set(signers[n_test_signers : n_test_signers + n_val_signers])
            train_signers = set(signers[n_test_signers + n_val_signers :])

            for s_id, s_samples in signer_groups.items():
                if s_id in test_signers:
                    for s in s_samples:
                        s.split = SplitType.TEST
                        test_samples.append(s)
                elif s_id in val_signers:
                    for s in s_samples:
                        s.split = SplitType.VAL
                        val_samples.append(s)
                else:
                    for s in s_samples:
                        s.split = SplitType.TRAIN
                        train_samples.append(s)
        else:
            # When signers are few (< 6), partition stratified by sentence so all classes are learned
            sentence_groups: Dict[str, List[SignSample]] = {}
            for s in samples:
                key = re.sub(r"[^\w\s]", "", (s.english_text or "").lower()).strip()
                sentence_groups.setdefault(key, []).append(s)

            for key, group in sentence_groups.items():
                if len(group) == 1:
                    group[0].split = SplitType.TRAIN
                    train_samples.append(group[0])
                elif len(group) == 2:
                    group[0].split = SplitType.TRAIN
                    train_samples.append(group[0])
                    group[1].split = SplitType.VAL
                    val_samples.append(group[1])
                else:
                    val_s = group[-1]
                    val_s.split = SplitType.VAL
                    val_samples.append(val_s)
                    if len(group) >= 6:
                        test_s = group[-2]
                        test_s.split = SplitType.TEST
                        test_samples.append(test_s)
                        rem = group[:-2]
                    else:
                        rem = group[:-1]
                    for s in rem:
                        s.split = SplitType.TRAIN
                        train_samples.append(s)
            return train_samples, val_samples, test_samples

        # Distribute unknown-signer samples by video/sample ID
        if unknown_samples:
            random.shuffle(unknown_samples)
            n_test = round(len(unknown_samples) * test_ratio)
            n_val = round(len(unknown_samples) * val_ratio)

            for s in unknown_samples[:n_test]:
                s.split = SplitType.TEST
                test_samples.append(s)
            for s in unknown_samples[n_test : n_test + n_val]:
                s.split = SplitType.VAL
                val_samples.append(s)
            for s in unknown_samples[n_test + n_val :]:
                s.split = SplitType.TRAIN
                train_samples.append(s)

        return train_samples, val_samples, test_samples

    @staticmethod
    def clean_and_tokenize(text: str) -> List[str]:
        """Normalize and tokenize text separating punctuation marks."""
        import re
        text = text.lower().strip()
        # Separate punctuation with whitespace
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        return text.split()

    def build_vocabulary(
        self,
        samples: List[SignSample],
        min_freq: int = 1,
    ) -> Dict[str, int]:
        """
        Builds English token vocabulary mapping with standard special tokens:
        <pad>: 0, <bos>: 1, <eos>: 2, <unk>: 3, <blank>: 4 (for CTC)
        """
        word_counts = Counter()
        for sample in samples:
            tokens = self.clean_and_tokenize(sample.english_text)
            word_counts.update(tokens)

        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<blank>": 4,
        }
        idx = 5
        for word, count in word_counts.most_common():
            if count >= min_freq and word not in vocab:
                vocab[word] = idx
                idx += 1

        return vocab

    def generate_manifest_bundle(
        self,
        samples: List[SignSample],
        output_dir: str = "data/manifests",
        reports_dir: str = "data/reports",
        dataset_name: str = "ISL_Consolidated",
    ) -> ManifestSummary:
        """Partition samples, build manifests, audit leakage, and save artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        train_s, val_s, test_s = self.partition_by_signer(samples)
        passed, violations = self.audit_leakage(train_s, val_s, test_s)

        # Build vocabulary strictly from the TRAINING partition to avoid leakage
        vocab = self.build_vocabulary(train_s)
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)

        # Export split manifests
        for split_name, split_data in [
            ("train", train_s),
            ("val", val_s),
            ("test", test_s),
        ]:
            split_file = os.path.join(output_dir, f"{split_name}_manifest.json")
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump([s.model_dump() for s in split_data], f, indent=2, ensure_ascii=False)

        # Compute summary metrics
        total_duration = sum(
            s.video_meta.duration_sec for s in samples if s.video_meta
        ) / 3600.0

        all_signers = {s.signer_id for s in samples if s.signer_id}

        summary = ManifestSummary(
            dataset_name=dataset_name,
            total_samples=len(samples),
            split_distribution={
                "train": len(train_s),
                "val": len(val_s),
                "test": len(test_s),
            },
            total_duration_hours=round(total_duration, 3),
            signer_count=len(all_signers),
            english_vocab_size=len(vocab),
            gloss_vocab_size=0,
            leakage_check_passed=passed,
            created_at=datetime.utcnow().isoformat(),
        )

        # Write audit report
        report_path = os.path.join(reports_dir, "dataset_audit_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            report_data = {
                "summary": summary.model_dump(),
                "leakage_violations": violations,
            }
            json.dump(report_data, f, indent=2)

        return summary
