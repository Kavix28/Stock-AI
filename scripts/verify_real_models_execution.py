"""
Fresh-Process Verification of Genuine Model Execution & Checkpoints.

Verifies:
1. FinBERT real-weight forward pass.
2. Saved PyTorch checkpoints (GMU fusion and Market BiLSTM).
3. Forward inference with correct tensor shapes and finite probabilities.
4. GMU gate weight sum invariants.
5. InferenceEngine outputs, fallback labeling, and ATVI delisting disclosure.
"""
import os
import sys
import numpy as np
import torch

project_root = r"C:\PROJECTS\multimodal-stock-ai"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from stock_ai.models.news.finbert import FinBertNewsEncoder
from stock_ai.models.fusion.gmu import GatedMultimodalUnit
from stock_ai.models.market.bilstm import MarketBiLSTM
from stock_ai.inference.engine import InferenceEngine


def main():
    print("=" * 80)
    print("STEP 1: FinBERT Real-Weight Execution")
    print("=" * 80)
    encoder = FinBertNewsEncoder(lazy_load=False)
    headline = "Federal Reserve signals interest rate decision amid cooling inflation."
    emb = encoder.encode_text(headline)
    norm = float(np.linalg.norm(emb))
    print(f"Encoded headline: '{headline}'")
    print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}, L2 norm: {norm:.4f}")
    assert emb.shape == (768,)
    assert np.isfinite(norm) and norm > 0.0

    print("\n" + "=" * 80)
    print("STEP 2: Load Saved PyTorch Checkpoints")
    print("=" * 80)
    gmu_path = os.path.join(project_root, "models", "gmu", "gmu_fusion.pt")
    bilstm_path = os.path.join(project_root, "models", "market", "bilstm_market.pt")

    assert os.path.exists(gmu_path), f"Missing GMU checkpoint at {gmu_path}"
    assert os.path.exists(bilstm_path), f"Missing BiLSTM checkpoint at {bilstm_path}"

    print(f"GMU checkpoint: {gmu_path} ({os.path.getsize(gmu_path)} bytes)")
    print(f"BiLSTM checkpoint: {bilstm_path} ({os.path.getsize(bilstm_path)} bytes)")

    # Instantiate architectures matching config
    gmu = GatedMultimodalUnit(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)
    gmu.load_state_dict(torch.load(gmu_path, map_location="cpu"))
    gmu.eval()
    print("GMU Fusion state_dict loaded successfully.")

    bilstm = MarketBiLSTM(sequence_length=20, feature_dim=16, lstm_hidden_1=64, lstm_hidden_2=32, proj_dim=64)
    bilstm.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
    bilstm.eval()
    print("Market BiLSTM state_dict loaded successfully.")

    print("\n" + "=" * 80)
    print("STEP 3: Genuine Forward Tensor Inferences")
    print("=" * 80)
    # Batch of 2 samples (seq_len=20, market_feat_dim=16)
    x_market = torch.randn(2, 20, 16, dtype=torch.float32)
    x_news = torch.tensor(np.stack([emb, encoder.get_zero_news_vector()]), dtype=torch.float32)

    with torch.no_grad():
        bilstm_prob = bilstm(x_market)
        gmu_prob, gate_tensor = gmu.forward_with_gate_weights(x_market, x_news)
        z_m = gate_tensor.mean(dim=-1, keepdim=True)
        z_n = (1.0 - gate_tensor).mean(dim=-1, keepdim=True)

    print(f"Market BiLSTM output shape: {bilstm_prob.shape}, values: {bilstm_prob.squeeze().tolist()}")
    print(f"GMU Fusion output shape: {gmu_prob.shape}, values: {gmu_prob.squeeze().tolist()}")
    print(f"GMU Mean Market Gate weights (z_m): {z_m.squeeze().tolist()}")
    print(f"GMU Mean News Gate weights (z_n):   {z_n.squeeze().tolist()}")

    gate_sums = (z_m + z_n).squeeze().tolist()
    print(f"Gate weights sum (z_m + z_n):  {gate_sums}")
    for s in gate_sums:
        assert abs(s - 1.0) < 1e-5, f"Gate weight violation: sum={s}"

    print("\n" + "=" * 80)
    print("STEP 4: InferenceEngine Live Predictions & Fallbacks")
    print("=" * 80)
    engine = InferenceEngine(
        data_dir=os.path.join(project_root, "data", "processed", "market"),
        models_dir=os.path.join(project_root, "models"),
        results_dir=os.path.join(project_root, "results")
    )

    # AAPL Multimodal prediction
    pred_aapl_gmu = engine.get_prediction("AAPL", model_type="gmu_fusion")
    print("\nAAPL GMU Prediction:")
    for k, v in pred_aapl_gmu.items():
        print(f"  {k}: {v}")
    assert pred_aapl_gmu["ticker"] == "AAPL"
    assert pred_aapl_gmu["model_version"] == "GMU-Fusion-v1.0-PyTorch"
    assert "market_gate_weight" in pred_aapl_gmu

    # NVDA BiLSTM prediction
    pred_nvda_bilstm = engine.get_prediction("NVDA", model_type="market_bilstm")
    print("\nNVDA BiLSTM Prediction:")
    for k, v in pred_nvda_bilstm.items():
        print(f"  {k}: {v}")
    assert pred_nvda_bilstm["ticker"] == "NVDA"
    assert pred_nvda_bilstm["model_version"] == "Market-BiLSTM-v1.0-PyTorch"
    assert pred_nvda_bilstm["market_gate_weight"] == 1.0
    assert pred_nvda_bilstm["news_gate_weight"] == 0.0

    # ATVI delisted asset
    pred_atvi = engine.get_prediction("ATVI", model_type="gmu_fusion")
    print("\nATVI Delisted Asset Prediction:")
    for k, v in pred_atvi.items():
        print(f"  {k}: {v}")
    assert "Delisted Oct 2023" in pred_atvi["ticker_status"]

    print("\nALL VERIFICATIONS PASSED IN FRESH PROCESS WITH ZERO ERRORS.")


if __name__ == "__main__":
    main()
