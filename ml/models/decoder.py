"""
Autoregressive Transformer Decoder for Sign Language Translation.
Decodes contextual encoder representations into natural language token sequences.
"""

import torch
import torch.nn as nn
from ml.models.positional_encoding import PositionalEncoding


class TranslationDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder with causal self-attention and cross-attention.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_target_len: int = 200,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_target_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generates causal upper-triangular mask (True for masked positions)."""
        mask = torch.triu(torch.ones(sz, sz, device=device) == 1, diagonal=1)
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        tgt: (B, U)
        memory: (B, T, d_model)
        Returns: logits of shape (B, U, vocab_size)
        """
        U = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(U, tgt.device)

        # Embedding scaled by sqrt(d_model)
        x = self.tok_embedding(tgt) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_proj(output)
        return logits
