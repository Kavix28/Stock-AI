import pytest
import torch
import numpy as np

from src.stock_ai.models.market.bilstm import MarketBiLSTM, build_market_bilstm
from src.stock_ai.models.fusion.gmu import GatedMultimodalUnit, build_gmu_fusion_model


def test_market_bilstm_forward_shape():
    model = build_market_bilstm(sequence_length=20, feature_dim=16, proj_dim=64)
    x = torch.randn(4, 20, 16)
    out = model(x)

    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_market_bilstm_extract_features():
    model = build_market_bilstm(sequence_length=20, feature_dim=16, proj_dim=64)
    x = torch.randn(3, 20, 16)
    h_m = model.extract_features(x)

    assert h_m.shape == (3, 64)


def test_gmu_forward_shape():
    model = build_gmu_fusion_model(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    x_m = torch.randn(5, 20, 16)
    x_n = torch.randn(5, 768)

    prob = model(x_m, x_n)
    assert prob.shape == (5, 1)
    assert torch.all(prob >= 0.0) and torch.all(prob <= 1.0)


def test_gmu_gate_weights():
    model = build_gmu_fusion_model(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    x_m = torch.randn(2, 20, 16)
    x_n = torch.randn(2, 768)

    prob, gate = model.forward_with_gate_weights(x_m, x_n)
    assert gate.shape == (2, 64)
    # Sigmoid ensures gate is strictly between 0 and 1
    assert torch.all(gate > 0.0) and torch.all(gate < 1.0)


def test_gmu_perturbation_changes_prediction():
    torch.manual_seed(42)
    model = build_gmu_fusion_model(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    model.eval()

    x_m = torch.randn(1, 20, 16)
    x_n_bullish = torch.ones(1, 768) * 2.0
    x_n_bearish = torch.ones(1, 768) * -2.0

    with torch.no_grad():
        prob_bullish = model(x_m, x_n_bullish).item()
        prob_bearish = model(x_m, x_n_bearish).item()

    assert prob_bullish != prob_bearish


def test_gmu_market_only_fallback():
    model = build_gmu_fusion_model(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    model.eval()
    x_m = torch.randn(3, 20, 16)

    with torch.no_grad():
        out = model.forward_market_only(x_m)

    assert out.shape == (3, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_gmu_gradient_flow():
    model = build_gmu_fusion_model(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    model.train()

    x_m = torch.randn(4, 20, 16, requires_grad=True)
    x_n = torch.randn(4, 768, requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    prob = model(x_m, x_n)
    loss = torch.nn.functional.binary_cross_entropy(prob, target)
    loss.backward()

    # Verify gradients computed for market encoder, news proj, and gate
    assert model.market_encoder.fc_proj.weight.grad is not None
    assert model.news_proj[0].weight.grad is not None
    assert model.gate_fc.weight.grad is not None
    assert model.classifier[0].weight.grad is not None
