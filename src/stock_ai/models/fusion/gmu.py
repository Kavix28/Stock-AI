"""
Gated Multimodal Unit (GMU) Fusion Architecture in PyTorch.

Combines sequential market representations (via BiLSTM) with contextual
financial news embeddings (via FinBERT) using learned convex gating:
    h_m = LayerNorm(ReLU(W_m h_bilstm + b_m))  in R^64
    h_n = LayerNorm(ReLU(W_n h_finbert + b_n)) in R^64
    z   = Sigmoid(W_z [h_m; h_n] + b_z)        in R^64
    h_f = z * h_m + (1 - z) * h_n             in R^64
    y   = Sigmoid(W_c h_f + b_c)              in (0, 1)
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

try:
    from stock_ai.models.market.bilstm import MarketBiLSTM
except ImportError:
    from src.stock_ai.models.market.bilstm import MarketBiLSTM


class GatedMultimodalUnit(nn.Module):
    """
    Learned Gated Multimodal Unit fusion network for multimodal stock trend prediction.
    """

    def __init__(
        self,
        seq_len: int = 20,
        market_feat_dim: int = 16,
        news_embed_dim: int = 768,
        proj_dim: int = 64,
        head_hidden_dim: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.market_feat_dim = market_feat_dim
        self.news_embed_dim = news_embed_dim
        self.proj_dim = proj_dim

        # 1. Market Branch (BiLSTM)
        self.market_encoder = MarketBiLSTM(
            sequence_length=seq_len,
            feature_dim=market_feat_dim,
            proj_dim=proj_dim,
            dropout_rate=dropout_rate
        )

        # 2. News Branch Projection (FinBERT 768 -> proj_dim 64)
        self.news_proj = nn.Sequential(
            nn.Linear(news_embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 3. GMU Gating Mechanism
        # Concatenates h_m (proj_dim) and h_n (proj_dim) -> 2 * proj_dim
        self.gate_fc = nn.Linear(proj_dim * 2, proj_dim)
        self.gate_act = nn.Sigmoid()

        # 4. Multimodal Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1)
        )

    def compute_gate(self, h_m: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        """Computes element-wise gate vector z in (0, 1)^proj_dim."""
        concat_rep = torch.cat([h_m, h_n], dim=-1)
        gate = self.gate_act(self.gate_fc(concat_rep))
        return gate

    def fuse_representations(
        self,
        h_m: torch.Tensor,
        h_n: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes gated convex combination:
        h_fused = z * h_m + (1 - z) * h_n
        Returns: (h_fused, gate)
        """
        gate = self.compute_gate(h_m, h_n)
        h_fused = (gate * h_m) + ((1.0 - gate) * h_n)
        return h_fused, gate

    def forward(
        self,
        market_input: torch.Tensor,
        news_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning predicted probability in (0, 1).
        
        Args:
            market_input: Tensor of shape (batch_size, 20, 16)
            news_input: Tensor of shape (batch_size, 768)
        Returns:
            probability: Tensor of shape (batch_size, 1)
        """
        # Market branch representation
        h_m = self.market_encoder.extract_features(market_input)

        # News branch representation
        h_n = self.news_proj(news_input)

        # Gated fusion
        h_fused, _ = self.fuse_representations(h_m, h_n)

        # Classification
        logits = self.classifier(h_fused)
        return torch.sigmoid(logits)

    def forward_with_gate_weights(
        self,
        market_input: torch.Tensor,
        news_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both predicted probability and the gate tensor z."""
        h_m = self.market_encoder.extract_features(market_input)
        h_n = self.news_proj(news_input)
        h_fused, gate = self.fuse_representations(h_m, h_n)
        logits = self.classifier(h_fused)
        return torch.sigmoid(logits), gate

    def forward_market_only(self, market_input: torch.Tensor) -> torch.Tensor:
        """
        Market-only fallback prediction using zero news input.
        Ensures consistent dimensions and evaluation without fabricated news.
        """
        device = market_input.device
        batch_size = market_input.size(0)
        zero_news = torch.zeros((batch_size, self.news_embed_dim), device=device)
        return self.forward(market_input, zero_news)


def build_gmu_fusion_model(
    seq_len: int = 20,
    market_feat_dim: int = 16,
    news_embed_dim: int = 768,
    proj_dim: int = 64,
    head_hidden_dim: int = 32,
    dropout_rate: float = 0.2
) -> GatedMultimodalUnit:
    """Factory function for GatedMultimodalUnit."""
    return GatedMultimodalUnit(
        seq_len=seq_len,
        market_feat_dim=market_feat_dim,
        news_embed_dim=news_embed_dim,
        proj_dim=proj_dim,
        head_hidden_dim=head_hidden_dim,
        dropout_rate=dropout_rate
    )
