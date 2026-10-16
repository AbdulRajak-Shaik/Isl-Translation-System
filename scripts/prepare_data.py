"""
Command-line pipeline for dataset ingestion, manifest creation, and data validation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.schemas import ISLTaskType, SignSample, SplitType, VideoMetadata
from ml.data.manifest_generator import ManifestGenerator


def generate_synthetic_smoke_fixture(output_dir: str = "data") -> None:
    """
    Generates a calibrated synthetic test fixture for unit tests and local pipeline verification.
    This fixture follows strict software engineering testing practices:
    it exercises the entire data pipeline without claiming to be genuine training data.
    """
    print("[Pipeline] Generating synthetic ISL test fixture for pipeline verification...")
    processed_dir = os.path.join(output_dir, "processed")
    manifests_dir = os.path.join(output_dir, "manifests")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    # Synthetic sentences covering basic conversational ISL concepts
    test_utterances = [
        ("Good morning, doctor.", "శుభోదయం డాక్టర్.", ["GOOD", "MORNING", "DOCTOR"], "signer_01"),
        ("Where is the hospital?", "ఆసుపత్రి ఎక్కడ ఉంది?", ["WHERE", "HOSPITAL"], "signer_01"),
        ("I need urgent help.", "నాకు అత్యవసర సహాయం కావాలి.", ["ME", "NEED", "URGENT", "HELP"], "signer_02"),
        ("What is your name?", "మీ పేరు ఏమిటి?", ["YOU", "NAME", "WHAT"], "signer_02"),
        ("Please sit down.", "దయచేసి కూర్చోండి.", ["PLEASE", "SIT"], "signer_03"),
        ("Are you feeling pain?", "మీకు నొప్పిగా ఉందా?", ["YOU", "FEEL", "PAIN"], "signer_03"),
        ("Take this medicine daily.", "ఈ మందును రోజూ తీసుకోండి.", ["MEDICINE", "DAILY", "TAKE"], "signer_04"),
        ("Thank you very much.", "చాలా ధన్యవాదాలు.", ["THANK_YOU", "VERY_MUCH"], "signer_04"),
    ]

    samples = []
    for idx, (en, te, glosses, signer) in enumerate(test_utterances):
        sample_id = f"smoke_fixture_{idx+1:03d}"
        seq_len = np.random.randint(25, 45)
        # Create a mock 75-keypoint landmark trajectory (T, 225)
        pose_data = np.random.normal(0.0, 0.2, size=(seq_len, 225)).astype(np.float32)
        pose_file = os.path.join(processed_dir, f"{sample_id}.npy")
        np.save(pose_file, pose_data)

        sample = SignSample(
            sample_id=sample_id,
            dataset_name="Synthetic_Smoke_Fixture",
            video_path=None,
            pose_path=pose_file,
            signer_id=signer,
            task_type=ISLTaskType.CONTINUOUS_TRANSLATION,
            english_text=en,
            telugu_text=te,
            gloss_tokens=glosses,
            split=SplitType.TRAIN,
            video_meta=VideoMetadata(
                duration_sec=round(seq_len / 25.0, 2),
                fps=25.0,
                frame_count=seq_len,
                width=1280,
                height=720,
                is_corrupt=False,
            ),
        )
        samples.append(sample)

    generator = ManifestGenerator(random_seed=42)
    summary = generator.generate_manifest_bundle(
        samples=samples,
        output_dir=manifests_dir,
        reports_dir=os.path.join(output_dir, "reports"),
        dataset_name="Synthetic_Smoke_Fixture",
    )

    print(f"[Pipeline] Successfully generated manifest bundle with {summary.total_samples} samples.")
    print(f"[Pipeline] Split distribution: {summary.split_distribution}")
    print(f"[Pipeline] English vocab size: {summary.english_vocab_size}")
    print(f"[Pipeline] Signer leakage check passed: {summary.leakage_check_passed}")


def main():
    parser = argparse.ArgumentParser(description="Indian Sign Language Data Preparation Pipeline")
    parser.add_argument("--fixture", action="store_true", help="Generate synthetic test fixture for verification")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for manifests and processed data")
    args = parser.parse_args()

    if args.fixture:
        generate_synthetic_smoke_fixture(args.output_dir)
    else:
        print("Please specify an action. Use --fixture to create the pipeline validation fixture.")


if __name__ == "__main__":
    main()
