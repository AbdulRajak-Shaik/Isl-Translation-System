"""
Unit Tests for Model Forward, Losses, CTC Auxiliary Head, and Autoregressive Generation.
"""

import torch
import pytest

from ml.models.positional_encoding import PositionalEncoding
from ml.models.encoder import TemporalPoseEncoder
from ml.models.decoder import TranslationDecoder
from ml.models.seq2seq_transformer import ISLSeq2SeqTransformer


class TestModelArchitecture:

    def test_positional_encoding_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.zeros(2, 25, 64)
        out = pe(x)
        assert out.shape == (2, 25, 64)

    def test_encoder_decoder_forward(self):
        B, T, U, V = 2, 30, 10, 50
        input_dim = 225
        d_model = 64

        model = ISLSeq2SeqTransformer(
            vocab_size=V,
            input_dim=input_dim,
            d_model=d_model,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
            max_seq_len=100,
            max_target_len=50,
        )

        src = torch.randn(B, T, input_dim)
        tgt = torch.randint(0, V, (B, U))

        out = model(src, tgt)
        assert "decoder_logits" in out
        assert "ctc_logits" in out

        # Decoder logits should be (B, U, V)
        assert out["decoder_logits"].shape == (B, U, V)
        # CTC logits should be (B, T, V)
        assert out["ctc_logits"].shape == (B, T, V)

    def test_generation_and_uncertainty(self):
        V = 30
        model = ISLSeq2SeqTransformer(
            vocab_size=V,
            input_dim=225,
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=128,
        )

        src = torch.randn(1, 20, 225)
        gen = model.generate(src, max_length=15, confidence_threshold=0.60)

        assert "token_ids" in gen
        assert "confidence" in gen
        assert "is_uncertain" in gen
        assert "mean_entropy" in gen
        assert 0.0 <= gen["confidence"] <= 1.0
        assert isinstance(gen["is_uncertain"], bool)
