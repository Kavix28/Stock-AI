"""
Market-Only Bidirectional LSTM Architecture in PyTorch.

Processes 20-session sequential market indicator vectors (16 features per session)
and predicts next-day stock price direction (Up vs. Down).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class MarketBiLSTM(nn.Module):
    """
    Two-layer Bidirectional LSTM for market time-series sequence modeling.
    Can be used as standalone classifier or as the market encoder in GMU fusion.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        feature_dim: int = 16,
        lstm_hidden_1: int = 64,
        lstm_hidden_2: int = 32,
        proj_dim: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim

        # Layer 1: BiLSTM (output dim: 2 * lstm_hidden_1 = 128)
        self.lstm1 = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: BiLSTM (output dim: 2 * lstm_hidden_2 = 64)
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1 * 2,
            hidden_size=lstm_hidden_2,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        # Projection layer to latent space
        self.fc_proj = nn.Linear(lstm_hidden_2 * 2, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.relu = nn.ReLU()

        # Classification head for market-only prediction
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 64-dimensional normalized latent representation h_m.
        x: Tensor of shape (batch_size, sequence_length, feature_dim)
        """
        # x: (B, T, D)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        # Take the final temporal state: out2[:, -1, :]
        last_step = out2[:, -1, :]
        last_step = self.dropout2(last_step)

        # Project and normalize
        h_m = self.fc_proj(last_step)
        h_m = self.layer_norm(h_m)
        h_m = self.relu(h_m)
        return h_m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward classification probability.
        Returns: Tensor of shape (batch_size, 1) with sigmoid output in (0, 1).
        """
        h_m = self.extract_features(x)
        logits = self.classifier(h_m)
        return torch.sigmoid(logits)


def build_market_bilstm(
    sequence_length: int = 20,
    feature_dim: int = 16,
    proj_dim: int = 64,
    dropout_rate: float = 0.2
) -> MarketBiLSTM:
    """Factory function for MarketBiLSTM."""
    return MarketBiLSTM(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        proj_dim=proj_dim,
        dropout_rate=dropout_rate
    )
