"""
Spatio-Temporal Sequence Transformer Encoder.
Projects raw or normalized landmark frames and extracts contextual sign representations.
"""

import torch
import torch.nn as nn
from ml.models.positional_encoding import PositionalEncoding


class TemporalPoseEncoder(nn.Module):
    """
    Encoder for continuous skeletal motion features.
    """

    def __init__(
        self,
        input_dim: int = 225,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        src: (B, T, input_dim)
        src_key_padding_mask: (B, T) - True for padding positions (PyTorch convention)
        Returns: memory of shape (B, T, d_model)
        """
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory
