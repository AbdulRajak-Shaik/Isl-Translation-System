"""
Sinusoidal Positional Encoding for continuous temporal sequences.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Dynamically compute sinusoidal encoding for arbitrary extended sequence lengths
            d_model = x.size(-1)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float, device=x.device)
                * (-math.log(10000.0) / d_model)
            )
            dyn_pe = torch.zeros(1, seq_len, d_model, device=x.device)
            dyn_pe[0, :, 0::2] = torch.sin(position * div_term)
            dyn_pe[0, :, 1::2] = torch.cos(position * div_term)
            x = x + dyn_pe
        else:
            x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
