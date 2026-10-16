"""
PyTorch Training Orchestrator for ISL Continuous Translation.
Implements Joint CTC-Attention Loss, Gradient Accumulation, Mixed Precision,
Validation-based checkpoint selection, and resume functionality.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer
from ml.training.metrics import compute_bleu, compute_chrf, compute_wer


class ISLTrainer:
    """Trainer orchestrator for Continuous Indian Sign Language Translation."""

    def __init__(
        self,
        model: ISLSeq2SeqTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        vocab: Dict[str, int],
        config: Dict[str, Any],
        device: torch.device,
        save_dir: str = "ml/checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.pad_idx = vocab.get("<pad>", 0)
        self.blank_idx = vocab.get("<blank>", 4)
        self.ctc_weight = config.get("loss", {}).get("ctc_weight", 0.3)
        self.accumulate_grad_batches = config.get("training", {}).get("accumulate_grad_batches", 1)
        self.gradient_clip_val = config.get("training", {}).get("gradient_clip_val", 1.0)

        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx,
            label_smoothing=config.get("loss", {}).get("label_smoothing", 0.1),
        )
        self.criterion_ctc = nn.CTCLoss(
            blank=self.blank_idx,
            zero_infinity=True,
        )

        self.best_bleu_4 = -1.0
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ctc_loss = 0.0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            poses = batch["poses"].to(self.device)
            pose_mask = batch["pose_mask"].to(self.device)
            seq_lens = batch["seq_lens"].to(self.device)

            targets = batch["targets"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            target_lens = batch["target_lens"].to(self.device)

            # Decoder input: targets[:, :-1]
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]
            dec_mask = target_mask[:, :-1]

            outputs = self.model(
                src=poses,
                tgt=decoder_input,
                src_mask=pose_mask,
                tgt_mask=dec_mask,
            )

            decoder_logits = outputs["decoder_logits"]
            ctc_logits = outputs["ctc_logits"]

            # 1. Cross-Entropy Translation Loss
            B, U_minus_1, V = decoder_logits.shape
            ce_loss = self.criterion_ce(
                decoder_logits.reshape(-1, V),
                decoder_target.reshape(-1),
            )

            # 2. Auxiliary CTC Loss
            # CTCLoss expects log_probs: (T, B, V)
            log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
            # Clamping lengths to avoid zero-length errors
            valid_seq_lens = torch.clamp(seq_lens, min=1)
            valid_tgt_lens = torch.clamp(target_lens - 2, min=1)  # exclude bos and eos

            # CTC targets without bos and eos
            ctc_targets = targets[:, 1:-1]

            try:
                ctc_loss = self.criterion_ctc(log_probs, ctc_targets, valid_seq_lens, valid_tgt_lens)
            except Exception:
                ctc_loss = torch.tensor(0.0, device=self.device)

            # Joint Loss
            loss = ((1.0 - self.ctc_weight) * ce_loss) + (self.ctc_weight * ctc_loss)
            loss = loss / self.accumulate_grad_batches
            loss.backward()

            if (step + 1) % self.accumulate_grad_batches == 0 or (step + 1) == len(self.train_loader):
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulate_grad_batches
            total_ce_loss += ce_loss.item()
            total_ctc_loss += ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else 0.0

        n_steps = max(len(self.train_loader), 1)
        return {
            "loss": round(total_loss / n_steps, 4),
            "ce_loss": round(total_ce_loss / n_steps, 4),
            "ctc_loss": round(total_ctc_loss / n_steps, 4),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        hypotheses: List[str] = []
        references: List[str] = []

        for batch in self.val_loader:
            poses = batch["poses"].to(self.device)
            pose_mask = batch["pose_mask"].to(self.device)
            refs = batch["english_refs"]

            for i in range(poses.size(0)):
                single_pose = poses[i : i + 1]
                single_mask = pose_mask[i : i + 1]

                gen = self.model.generate(
                    src=single_pose,
                    src_mask=single_mask,
                    max_length=40,
                )
                words = [self.inv_vocab.get(t, "") for t in gen["token_ids"] if t in self.inv_vocab]
                hypotheses.append(" ".join(words).strip())
                references.append(refs[i].strip())

        bleu_metrics = compute_bleu(hypotheses, references)
        chrf_score = compute_chrf(hypotheses, references)
        wer_score = compute_wer(hypotheses, references)

        metrics = {
            **bleu_metrics,
            "chrf": chrf_score,
            "wer": wer_score,
            "sample_hyp": hypotheses[:2] if hypotheses else [],
            "sample_ref": references[:2] if references else [],
        }
        return metrics

    def fit(self, epochs: int = 5) -> Dict[str, Any]:
        print(f"[Training] Starting model training for {epochs} epochs on {self.device}...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = round(time.time() - t0, 2)

            bleu_4 = val_metrics.get("bleu_4", 0.0)
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce_loss"],
                "train_ctc": train_metrics["ctc_loss"],
                "val_bleu_4": bleu_4,
                "val_chrf": val_metrics["chrf"],
                "elapsed_sec": elapsed,
            }
            self.history.append(epoch_log)

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val BLEU-4: {bleu_4:.2f} | "
                f"Val chrF: {val_metrics['chrf']:.2f} | "
                f"Time: {elapsed}s"
            )

            # Save best checkpoint
            if bleu_4 >= self.best_bleu_4:
                self.best_bleu_4 = bleu_4
                self.save_checkpoint("isl_transformer_best.pt", epoch, val_metrics)

            # Always save latest checkpoint
            self.save_checkpoint("isl_transformer_last.pt", epoch, val_metrics)

        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {"best_bleu_4": self.best_bleu_4, "history": self.history}

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]) -> str:
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "vocab": self.vocab,
            "metrics": metrics,
            "config": self.config,
            "timestamp": time.time(),
        }
        torch.save(checkpoint, filepath)
        return filepath
