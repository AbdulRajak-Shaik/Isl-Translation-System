"""
Continuous Indian Sign Language Seq2Seq Transformer Model.
Combines Spatio-Temporal Pose Encoder, Autoregressive Decoder, and Auxiliary CTC Head.
Implements calibrated uncertainty estimation and generation.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.encoder import TemporalPoseEncoder
from ml.models.decoder import TranslationDecoder


class ISLSeq2SeqTransformer(nn.Module):
    """
    End-to-end continuous sign-to-text sequence model with joint CTC-Attention heads.
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 225,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        max_target_len: int = 200,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # 1. Temporal Pose Encoder
        self.encoder = TemporalPoseEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 2. Auxiliary CTC Projection Head
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # 3. Autoregressive Translation Decoder
        self.decoder = TranslationDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_target_len=max_target_len,
            pad_idx=pad_idx,
        )

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes frame sequence into memory representations."""
        # Convert boolean valid mask (True for valid) to PyTorch key_padding_mask (True for pad)
        src_key_padding_mask = ~src_mask if src_mask is not None else None
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        src: (B, T, input_dim)
        tgt: (B, U)
        Returns:
            decoder_logits: (B, U, vocab_size)
            ctc_logits: (B, T, vocab_size)
        """
        src_key_padding_mask = ~src_mask if src_mask is not None else None
        tgt_key_padding_mask = ~tgt_mask if tgt_mask is not None else None

        # 1. Encode
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 2. Auxiliary CTC Logits (over frame sequence)
        ctc_logits = self.ctc_head(memory)

        # 3. Decode Autoregressively
        decoder_logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return {
            "decoder_logits": decoder_logits,
            "ctc_logits": ctc_logits,
            "memory": memory,
        }

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
        confidence_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Autoregressively decodes a pose sequence into token IDs with uncertainty estimation.
        Returns:
            token_ids: List of decoded token indices
            mean_confidence: Calibrated sequence confidence [0.0, 1.0]
            is_uncertain: True if confidence < confidence_threshold
            token_entropies: List of per-token entropy scores
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Encode input sequence
        memory = self.encode(src, src_mask)

        # Initialize target sequence with <bos> token
        generated_tokens = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
        token_entropies: List[float] = []

        for step in range(max_length):
            logits = self.decoder(
                tgt=generated_tokens,
                memory=memory,
                memory_key_padding_mask=(~src_mask if src_mask is not None else None),
            )
            # Logits for current step: (B, vocab_size)
            next_token_logits = logits[:, -1, :].clone() / max(temperature, 1e-4)

            # Repetition penalty for already emitted tokens
            for b in range(batch_size):
                gen_tokens = generated_tokens[b].tolist()
                for tid in set(gen_tokens):
                    if tid not in (self.bos_idx, self.eos_idx, self.pad_idx):
                        if next_token_logits[b, tid] > 0:
                            next_token_logits[b, tid] /= 1.25
                        else:
                            next_token_logits[b, tid] *= 1.25

                # Prevent trigram repetition (loops like "good morning . good morning .")
                if len(gen_tokens) >= 3:
                    prefix = tuple(gen_tokens[-2:])
                    for i in range(len(gen_tokens) - 2):
                        if (gen_tokens[i], gen_tokens[i + 1]) == prefix:
                            blocked = gen_tokens[i + 2]
                            next_token_logits[b, blocked] = -1e9

            # Raw probabilities for true confidence calibration
            raw_probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-4), dim=-1)
            entropy = -(raw_probs * torch.log(raw_probs + 1e-9)).sum(dim=-1).item()
            token_entropies.append(entropy)

            probs = F.softmax(next_token_logits, dim=-1)

            # Greedy next token
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Check if all reached <eos>
            if next_token.item() == self.eos_idx:
                break

        # Calculate calibrated confidence: exp(-mean_entropy)
        mean_entropy = sum(token_entropies) / max(len(token_entropies), 1)
        mean_confidence = round(float(torch.exp(torch.tensor(-mean_entropy)).item()), 3)
        # Normalize confidence to [0, 1] range safely
        calibrated_confidence = max(0.0, min(1.0, mean_confidence))

        out_tokens = generated_tokens[0].tolist()
        # Remove <bos> and <eos>
        cleaned_tokens = [t for t in out_tokens if t not in (self.bos_idx, self.eos_idx, self.pad_idx)]

        is_uncertain = calibrated_confidence < confidence_threshold

        return {
            "token_ids": cleaned_tokens,
            "confidence": calibrated_confidence,
            "is_uncertain": is_uncertain,
            "mean_entropy": round(mean_entropy, 3),
        }
