"""
Distributed Data Parallel (DDP) Multi-GPU Training Script for Large GPU Clusters.
Launch with torchrun:
    torchrun --nproc_per_node=4 ml/training/train_ddp.py --config ml/configs/baseline_config.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.data.dataset import ISLDataset, isl_collate_fn
from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer
from ml.training.metrics import compute_bleu, compute_chrf, compute_wer


def setup_ddp():
    """Initializes distributed process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        # Fallback to single process
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_ddp():
    parser = argparse.ArgumentParser(description="Multi-GPU DDP Training for ISL Translation")
    parser.add_argument("--config", type=str, default="ml/configs/baseline_config.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    is_main_process = (rank == 0)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"[DDP Cluster] World Size: {world_size} GPUs. Launching multi-GPU distributed training...")

    # Load Vocabulary
    vocab_path = config["data"]["vocab_path"]
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_idx = vocab.get("<pad>", 0)
    blank_idx = vocab.get("<blank>", 4)

    # Manifests
    manifest_dir = config["data"]["manifest_dir"]
    train_manifest = os.path.join(manifest_dir, "train_manifest.json")
    val_manifest = os.path.join(manifest_dir, "val_manifest.json")

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

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=isl_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=isl_collate_fn,
        num_workers=2,
    )

    # Initialize Model
    m_cfg = config["model"]
    raw_model = ISLSeq2SeqTransformer(
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
        pad_idx=pad_idx,
    ).to(device)

    if world_size > 1 and torch.cuda.is_available():
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = raw_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    criterion_ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    best_bleu = -1.0
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            poses = batch["poses"].to(device, non_blocking=True)
            pose_mask = batch["pose_mask"].to(device, non_blocking=True)
            seq_lens = batch["seq_lens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            target_lens = batch["target_lens"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    src=poses,
                    tgt=targets[:, :-1],
                    src_mask=pose_mask,
                    tgt_mask=target_mask[:, :-1],
                )
                B, U_m1, V = outputs["decoder_logits"].shape
                ce_loss = criterion_ce(outputs["decoder_logits"].reshape(-1, V), targets[:, 1:].reshape(-1))

                log_probs = outputs["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
                ctc_loss = criterion_ctc(log_probs, targets[:, 1:-1], torch.clamp(seq_lens, min=1), torch.clamp(target_lens - 2, min=1))
                loss = (0.7 * ce_loss) + (0.3 * ctc_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        if is_main_process:
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Train Loss: {avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, "isl_transformer_cluster_latest.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "vocab": vocab,
                "config": config,
            }, checkpoint_path)

    if is_main_process:
        print("[DDP Cluster] Distributed training run completed successfully.")

    cleanup_ddp()


if __name__ == "__main__":
    train_ddp()
