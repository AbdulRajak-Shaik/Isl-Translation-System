"""
Training script for Indian Sign Language Continuous Translation.
CLI entry point to launch baseline training or smoke-tests.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

# Add repository root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.data.dataset import ISLDataset, isl_collate_fn
from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer
from ml.training.trainer import ISLTrainer


def load_config(config_path: str = "ml/configs/baseline_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train ISL Continuous Translation Model")
    parser.add_argument("--config", type=str, default="ml/configs/baseline_config.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 2-epoch smoke test")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    if args.smoke_test:
        config["training"]["max_epochs"] = 2
        config["training"]["batch_size"] = 2
        print("[Smoke-Test Mode] Activated: Running 2 epochs on available data.")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Using device: {device} " f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Paths
    manifest_dir = config["data"]["manifest_dir"]
    vocab_path = config["data"]["vocab_path"]
    train_manifest = os.path.join(manifest_dir, "train_manifest.json")
    val_manifest = os.path.join(manifest_dir, "val_manifest.json")

    # Load Vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"[Data] Loaded vocabulary with {vocab_size} tokens from {vocab_path}")

    # Datasets and Loaders
    train_dataset = ISLDataset(
        manifest_path=train_manifest,
        vocab_path=vocab_path,
        max_seq_len=config["model"]["max_seq_len"],
        max_target_len=config["model"]["max_target_len"],
    )
    val_dataset = ISLDataset(
        manifest_path=val_manifest,
        vocab_path=vocab_path,
        max_seq_len=config["model"]["max_seq_len"],
        max_target_len=config["model"]["max_target_len"],
    )

    batch_size = config["training"]["batch_size"]
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

    print(f"[Data] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Instantiate Model
    m_cfg = config["model"]
    model = ISLSeq2SeqTransformer(
        vocab_size=vocab_size,
        input_dim=m_cfg["input_dim"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_encoder_layers=m_cfg["num_encoder_layers"],
        num_decoder_layers=m_cfg["num_decoder_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        max_seq_len=m_cfg["max_seq_len"],
        max_target_len=m_cfg["max_target_len"],
        pad_idx=vocab.get("<pad>", 0),
        bos_idx=vocab.get("<bos>", 1),
        eos_idx=vocab.get("<eos>", 2),
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Initialized {m_cfg['name']} with {num_params:,} trainable parameters.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    trainer = ISLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        vocab=vocab,
        config=config,
        device=device,
        save_dir=config["training"]["save_dir"],
    )

    trainer.fit(epochs=config["training"]["max_epochs"])
    print("[Training] Training run complete. Checkpoints saved to:", config["training"]["save_dir"])


if __name__ == "__main__":
    main()
