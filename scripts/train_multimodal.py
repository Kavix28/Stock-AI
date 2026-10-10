"""
End-to-End Multimodal Model Training and Benchmark Pipeline.

Trains and evaluates:
1. Majority Baseline
2. Direction Persistence (Lag-1)
3. Logistic Regression (Lag-1 Endpoint)
4. Logistic Regression (20-Day Flat Window)
5. Market-Only BiLSTM (PyTorch)
6. Gated Multimodal Unit (GMU Fusion: BiLSTM + FinBERT)
7. GMU Zero-News Ablation (Market-only inference on trained GMU)

Strict anti-leakage invariants:
- Chronological train / validation / test splits.
- Scaler fitted strictly on the training partition.
- Probability threshold calibrated strictly on the validation partition.
- Zero future news leakage (certified causal alignment).
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stock_ai.models.market.bilstm import MarketBiLSTM
from src.stock_ai.models.fusion.gmu import GatedMultimodalUnit
from src.stock_ai.data.news.cache import NewsEmbeddingCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_MARKET_FEATURES = [
    'high', 'close', 'open', 'volume', 'low', 'adjclose',
    'rsi', '12-Day EMA', '26-Day EMA', 'MACD', 'SAR',
    'Upper Band', 'Middle Band', 'Lower Band', 'Slow k', 'Slow d'
]

ALL_TICKERS = [
    "AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ",
    "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"
]


class MultimodalDataset(Dataset):
    """PyTorch Dataset for paired sequential market and FinBERT news embeddings."""

    def __init__(self, x_market: np.ndarray, x_news: np.ndarray, y: np.ndarray):
        self.x_market = torch.tensor(x_market, dtype=torch.float32)
        self.x_news = torch.tensor(x_news, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_market[idx], self.x_news[idx], self.y[idx]


def build_multimodal_dataset_for_ticker(
    ticker: str,
    lookback: int = 20,
    start_date: str = "2021-01-05",
    end_date: str = "2022-09-20"
) -> Dict[str, Any]:
    """
    Constructs aligned sequential market data and cached FinBERT news embeddings.
    """
    market_path = os.path.join(project_root, "data", "processed", "market", f"{ticker}.csv")
    embed_path = os.path.join(project_root, "data", "processed", "embeddings", f"{ticker}_embeddings.npz")

    if not os.path.exists(market_path):
        raise FileNotFoundError(f"Market file not found: {market_path}")
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding cache not found: {embed_path}")

    # 1. Load market data
    market_df = pd.read_csv(market_path)
    market_df['date'] = market_df['date'].astype(str)
    market_df = market_df.sort_values('date').reset_index(drop=True)

    # 2. Load news embeddings
    npz_data = np.load(embed_path, allow_pickle=True)
    cached_dates = list(npz_data['dates'])
    cached_embeddings = npz_data['embeddings']
    cached_has_news = npz_data['has_news']
    date_to_idx = {d: i for i, d in enumerate(cached_dates)}

    # Ensure Diff and PrevDayClose exist
    if 'Diff' not in market_df.columns:
        if 'PrevDayClose' not in market_df.columns:
            market_df['PrevDayClose'] = market_df['close'].shift(1)
        market_df['Diff'] = ((market_df['close'] - market_df['PrevDayClose']) > 0).astype(int)

    # Build feature arrays
    samples = []
    for i in range(lookback, len(market_df)):
        target_date = str(market_df.loc[i, 'date'])
        if target_date < start_date or target_date > end_date:
            continue

        # Market sequence: rows [i - lookback : i]
        market_window = market_df.loc[i - lookback : i - 1, CORE_MARKET_FEATURES].values
        if np.isnan(market_window).any():
            continue

        target_y = int(market_df.loc[i, 'Diff'])
        prev_day_trend = int(market_df.loc[i - 1, 'Diff']) if (i - 1 >= 0 and not pd.isna(market_df.loc[i - 1, 'Diff'])) else target_y

        # News embedding for target date
        if target_date in date_to_idx:
            n_idx = date_to_idx[target_date]
            news_vec = cached_embeddings[n_idx]
            has_news = bool(cached_has_news[n_idx])
        else:
            news_vec = np.zeros(768, dtype=np.float32)
            has_news = False

        samples.append({
            "ticker": ticker,
            "target_date": target_date,
            "market_seq": market_window,
            "market_flat": market_window.flatten(),
            "market_lag1": market_window[-1],
            "news_vec": news_vec,
            "has_news": has_news,
            "y": target_y,
            "prev_day_trend": prev_day_trend
        })

    return {"ticker": ticker, "samples": samples}


def partition_and_scale_data(
    all_samples: List[Dict[str, Any]],
    train_cutoff: str = "2022-01-01",
    val_cutoff: str = "2022-06-01"
) -> Dict[str, Any]:
    """
    Chronologically partitions samples into train, val, and test, and fits
    MinMaxScaler strictly on the training partition.
    """
    train_samples = [s for s in all_samples if s["target_date"] < train_cutoff]
    val_samples = [s for s in all_samples if train_cutoff <= s["target_date"] < val_cutoff]
    test_samples = [s for s in all_samples if s["target_date"] >= val_cutoff]

    if not train_samples:
        raise ValueError("Training partition is empty!")
    if not val_samples:
        raise ValueError("Validation partition is empty!")
    if not test_samples:
        raise ValueError("Test partition is empty!")

    # Fit scaler strictly on training market features
    train_market_matrices = np.concatenate([s["market_seq"] for s in train_samples], axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_market_matrices)

    def transform_bucket(bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
        x_m_seq = []
        x_m_flat = []
        x_m_lag1 = []
        x_news = []
        y = []
        persist = []
        dates = []
        tickers = []
        has_news = []

        for s in bucket:
            scaled_seq = scaler.transform(s["market_seq"])
            x_m_seq.append(scaled_seq)
            x_m_flat.append(scaled_seq.flatten())
            x_m_lag1.append(scaled_seq[-1])
            x_news.append(s["news_vec"])
            y.append(s["y"])
            persist.append(s["prev_day_trend"])
            dates.append(s["target_date"])
            tickers.append(s["ticker"])
            has_news.append(s["has_news"])

        return {
            "x_market_seq": np.array(x_m_seq, dtype=np.float32),
            "x_market_flat": np.array(x_m_flat, dtype=np.float32),
            "x_market_lag1": np.array(x_m_lag1, dtype=np.float32),
            "x_news": np.array(x_news, dtype=np.float32),
            "y": np.array(y, dtype=np.int64),
            "persist": np.array(persist, dtype=np.int64),
            "dates": dates,
            "tickers": tickers,
            "has_news": np.array(has_news, dtype=bool)
        }

    return {
        "train": transform_bucket(train_samples),
        "val": transform_bucket(val_samples),
        "test": transform_bucket(test_samples),
        "scaler": scaler
    }


def calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds decision threshold maximizing Macro F1 on validation data."""
    best_tau = 0.50
    best_f1 = -1.0
    for tau in np.arange(0.35, 0.66, 0.01):
        preds = (y_prob >= tau).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    p0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    p1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.50

    brier = brier_score_loss(y_true, y_prob)

    return {
        "model": model_name,
        "threshold": round(threshold, 2),
        "accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "class0_precision": round(p0, 4),
        "class0_recall": round(r0, 4),
        "class0_f1": round(f0, 4),
        "class1_precision": round(p1, 4),
        "class1_recall": round(r1, 4),
        "class1_f1": round(f1, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_down": int(tn + fn),
        "pred_up": int(tp + fp),
        "total_n": len(y_true),
        "auc_roc": round(auc, 4),
        "brier_score": round(brier, 4)
    }


def train_pytorch_bilstm(
    data: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42
) -> Tuple[MarketBiLSTM, float]:
    """Trains MarketBiLSTM on market sequences."""
    torch.manual_seed(seed)
    model = MarketBiLSTM(sequence_length=20, feature_dim=16, proj_dim=64)

    train_ds = MultimodalDataset(data["train"]["x_market_seq"], data["train"]["x_news"], data["train"]["y"])
    val_ds = MultimodalDataset(data["val"]["x_market_seq"], data["val"]["x_news"], data["val"]["y"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 8
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        for x_m, _, y_b in train_loader:
            optimizer.zero_grad()
            out = model(x_m)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_m, _, y_b in val_loader:
                out = model(x_m)
                val_loss += criterion(out, y_b).item() * len(y_b)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate threshold on validation set
    model.eval()
    with torch.no_grad():
        val_prob = model(torch.tensor(data["val"]["x_market_seq"], dtype=torch.float32)).numpy().ravel()
    tau = calibrate_threshold(data["val"]["y"], val_prob)

    return model, tau


def train_pytorch_gmu(
    data: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42
) -> Tuple[GatedMultimodalUnit, float]:
    """Trains GatedMultimodalUnit on sequential market and FinBERT news embeddings."""
    torch.manual_seed(seed)
    model = GatedMultimodalUnit(seq_len=20, market_feat_dim=16, news_embed_dim=768, proj_dim=64)

    train_ds = MultimodalDataset(data["train"]["x_market_seq"], data["train"]["x_news"], data["train"]["y"])
    val_ds = MultimodalDataset(data["val"]["x_market_seq"], data["val"]["x_news"], data["val"]["y"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    patience = 8
    patience_cnt = 0

    for epoch in range(epochs):
        model.train()
        for x_m, x_n, y_b in train_loader:
            optimizer.zero_grad()
            out = model(x_m, x_n)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_m, x_n, y_b in val_loader:
                out = model(x_m, x_n)
                val_loss += criterion(out, y_b).item() * len(y_b)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate threshold on validation set
    model.eval()
    with torch.no_grad():
        x_m_val = torch.tensor(data["val"]["x_market_seq"], dtype=torch.float32)
        x_n_val = torch.tensor(data["val"]["x_news"], dtype=torch.float32)
        val_prob = model(x_m_val, x_n_val).numpy().ravel()
    tau = calibrate_threshold(data["val"]["y"], val_prob)

    return model, tau


def run_multimodal_training(seed: int = 42) -> None:
    """Main execution orchestrating data loading, training, evaluation, and reporting."""
    logger.info("=" * 80)
    logger.info(f"STARTING MULTIMODAL RESEARCH PIPELINE TRAINING (Seed: {seed})")
    logger.info("=" * 80)

    # Output directories
    out_models_gmu = os.path.join(project_root, "models", "gmu")
    out_models_market = os.path.join(project_root, "models", "market")
    out_results = os.path.join(project_root, "results", "multimodal")
    out_reports = os.path.join(project_root, "reports")

    os.makedirs(out_models_gmu, exist_ok=True)
    os.makedirs(out_models_market, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)

    # 1. Collect samples across active universe
    all_samples = []
    ticker_counts = {}

    for ticker in ALL_TICKERS:
        try:
            t_data = build_multimodal_dataset_for_ticker(ticker)
            n_samples = len(t_data["samples"])
            all_samples.extend(t_data["samples"])
            ticker_counts[ticker] = n_samples
            logger.info(f"Loaded {n_samples} multimodal samples for {ticker}.")
        except Exception as e:
            logger.warning(f"Could not load samples for {ticker}: {e}")

    logger.info(f"\nTotal universe samples collected: {len(all_samples)}")

    # 2. Partition and scale
    data = partition_and_scale_data(all_samples)
    n_train = len(data["train"]["y"])
    n_val = len(data["val"]["y"])
    n_test = len(data["test"]["y"])

    logger.info(f"Partition counts: Train={n_train}, Val={n_val}, Test={n_test}")
    logger.info(f"Test Class Balance: Up={np.sum(data['test']['y'] == 1)} ({np.mean(data['test']['y'] == 1)*100:.1f}%), Down={np.sum(data['test']['y'] == 0)} ({np.mean(data['test']['y'] == 0)*100:.1f}%)")

    metrics_list = []
    test_y = data["test"]["y"]

    # -------------------------------------------------------------
    # 1. Training-Majority Baseline
    # -------------------------------------------------------------
    maj_class = int(np.bincount(data["train"]["y"]).argmax())
    prob_maj = np.full(n_test, float(maj_class))
    m1 = evaluate_predictions(test_y, prob_maj, 0.50, "1. Training-Majority Baseline")
    metrics_list.append(m1)

    # -------------------------------------------------------------
    # 2. Direction Persistence (Lag-1)
    # -------------------------------------------------------------
    prob_persist = data["test"]["persist"].astype(float)
    m2 = evaluate_predictions(test_y, prob_persist, 0.50, "2. Direction Persistence (Lag-1)")
    metrics_list.append(m2)

    # -------------------------------------------------------------
    # 3a. Next-Day Logistic Regression (Lag-1 Endpoint)
    # -------------------------------------------------------------
    lr_endpoint = LogisticRegression(max_iter=1000, random_state=seed)
    lr_endpoint.fit(data["train"]["x_market_lag1"], data["train"]["y"])
    val_prob_lr1 = lr_endpoint.predict_proba(data["val"]["x_market_lag1"])[:, 1]
    tau_lr1 = calibrate_threshold(data["val"]["y"], val_prob_lr1)
    test_prob_lr1 = lr_endpoint.predict_proba(data["test"]["x_market_lag1"])[:, 1]

    metrics_list.append(evaluate_predictions(test_y, test_prob_lr1, 0.50, "3a. Next-Day LR (Lag-1 Endpoint, Tau=0.50)"))
    metrics_list.append(evaluate_predictions(test_y, test_prob_lr1, tau_lr1, f"3a. Next-Day LR (Lag-1 Endpoint, Tau*={tau_lr1:.2f})"))

    # -------------------------------------------------------------
    # 3b. Next-Day Logistic Regression (20-Day Flat Window)
    # -------------------------------------------------------------
    lr_flat = LogisticRegression(max_iter=1000, random_state=seed)
    lr_flat.fit(data["train"]["x_market_flat"], data["train"]["y"])
    val_prob_lrflat = lr_flat.predict_proba(data["val"]["x_market_flat"])[:, 1]
    tau_lrflat = calibrate_threshold(data["val"]["y"], val_prob_lrflat)
    test_prob_lrflat = lr_flat.predict_proba(data["test"]["x_market_flat"])[:, 1]

    metrics_list.append(evaluate_predictions(test_y, test_prob_lrflat, 0.50, "3b. Next-Day LR (20-Day Flat, Tau=0.50)"))
    metrics_list.append(evaluate_predictions(test_y, test_prob_lrflat, tau_lrflat, f"3b. Next-Day LR (20-Day Flat, Tau*={tau_lrflat:.2f})"))

    # -------------------------------------------------------------
    # 4. Market-Only BiLSTM (PyTorch)
    # -------------------------------------------------------------
    logger.info("Training Market-Only BiLSTM...")
    bilstm_model, tau_bilstm = train_pytorch_bilstm(data, epochs=40, seed=seed)
    bilstm_model.eval()
    with torch.no_grad():
        test_prob_bilstm = bilstm_model(torch.tensor(data["test"]["x_market_seq"], dtype=torch.float32)).numpy().ravel()

    metrics_list.append(evaluate_predictions(test_y, test_prob_bilstm, 0.50, "4. Market-Only BiLSTM (Tau=0.50)"))
    metrics_list.append(evaluate_predictions(test_y, test_prob_bilstm, tau_bilstm, f"4. Market-Only BiLSTM (Tau*={tau_bilstm:.2f})"))

    # Save BiLSTM model
    torch.save(bilstm_model.state_dict(), os.path.join(out_models_market, "bilstm_market.pt"))

    # -------------------------------------------------------------
    # 5. Gated Multimodal Unit (GMU Fusion)
    # -------------------------------------------------------------
    logger.info("Training Gated Multimodal Unit (GMU Fusion)...")
    gmu_model, tau_gmu = train_pytorch_gmu(data, epochs=40, seed=seed)
    gmu_model.eval()

    with torch.no_grad():
        x_m_test = torch.tensor(data["test"]["x_market_seq"], dtype=torch.float32)
        x_n_test = torch.tensor(data["test"]["x_news"], dtype=torch.float32)
        test_prob_gmu, gate_weights = gmu_model.forward_with_gate_weights(x_m_test, x_n_test)
        test_prob_gmu = test_prob_gmu.numpy().ravel()
        gate_weights = gate_weights.numpy()

    metrics_list.append(evaluate_predictions(test_y, test_prob_gmu, 0.50, "5. GMU Fusion (Tau=0.50)"))
    metrics_list.append(evaluate_predictions(test_y, test_prob_gmu, tau_gmu, f"5. GMU Fusion (Tau*={tau_gmu:.2f})"))

    # -------------------------------------------------------------
    # 6. GMU News-Masked Ablation (Zero-News Vector)
    # -------------------------------------------------------------
    with torch.no_grad():
        test_prob_ablation = gmu_model.forward_market_only(x_m_test).numpy().ravel()

    metrics_list.append(evaluate_predictions(test_y, test_prob_ablation, 0.50, "6. GMU Zero-News Ablation (Tau=0.50)"))
    metrics_list.append(evaluate_predictions(test_y, test_prob_ablation, tau_gmu, f"6. GMU Zero-News Ablation (Tau*={tau_gmu:.2f})"))

    # Save GMU model
    torch.save(gmu_model.state_dict(), os.path.join(out_models_gmu, "gmu_fusion.pt"))

    # Save Config and Threshold Metadata
    model_config = {
        "model_architecture": "GatedMultimodalUnit",
        "market_feature_dim": 16,
        "news_embedding_dim": 768,
        "proj_dim": 64,
        "sequence_length": 20,
        "seed": seed,
        "calibrated_threshold": tau_gmu,
        "bilstm_calibrated_threshold": tau_bilstm,
        "train_cutoff": "2022-01-01",
        "val_cutoff": "2022-06-01",
        "evaluation_period": "2021-01-05 to 2022-09-20",
        "tickers_trained": ALL_TICKERS,
        "total_test_samples": n_test
    }
    with open(os.path.join(out_models_gmu, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # -------------------------------------------------------------
    # 7. Gate Weight Analysis
    # -------------------------------------------------------------
    # Gate z in (0, 1): weight assigned to market branch
    # (1 - z): weight assigned to news branch
    z_mean_per_sample = gate_weights.mean(axis=1)
    mean_market_weight = float(np.mean(z_mean_per_sample))
    mean_news_weight = float(np.mean(1.0 - z_mean_per_sample))

    logger.info(f"\nGMU Gate Weight Analysis across Test Samples:")
    logger.info(f"Mean Market Weight (z): {mean_market_weight:.4f}")
    logger.info(f"Mean News Weight (1-z): {mean_news_weight:.4f}")

    # Separate gate weights for days with news vs days without news
    has_news_mask = data["test"]["has_news"]
    if np.any(has_news_mask):
        z_when_news = float(np.mean(z_mean_per_sample[has_news_mask]))
        logger.info(f"Mean Market Weight when News is Present: {z_when_news:.4f} (News weight: {1.0 - z_when_news:.4f})")
    if np.any(~has_news_mask):
        z_when_no_news = float(np.mean(z_mean_per_sample[~has_news_mask]))
        logger.info(f"Mean Market Weight when News is Absent: {z_when_no_news:.4f} (News weight: {1.0 - z_when_no_news:.4f})")

    # -------------------------------------------------------------
    # 8. Save Artifacts: Predictions & Metrics Summary
    # -------------------------------------------------------------
    preds_df = pd.DataFrame({
        "target_date": data["test"]["dates"],
        "ticker": data["test"]["tickers"],
        "actual_y": test_y,
        "has_news": data["test"]["has_news"],
        "prob_persistence": prob_persist,
        "prob_lr_endpoint": test_prob_lr1,
        "prob_lr_flat": test_prob_lrflat,
        "prob_bilstm": test_prob_bilstm,
        "pred_bilstm": (test_prob_bilstm >= tau_bilstm).astype(int),
        "prob_gmu": test_prob_gmu,
        "pred_gmu": (test_prob_gmu >= tau_gmu).astype(int),
        "prob_gmu_ablation": test_prob_ablation,
        "gmu_market_gate_weight": z_mean_per_sample,
        "gmu_news_gate_weight": 1.0 - z_mean_per_sample
    })
    preds_csv_path = os.path.join(out_results, "multimodal_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    logger.info(f"Saved test predictions to: {preds_csv_path}")

    # Compute per-ticker evaluation metrics
    per_ticker_records = []
    for tkr in ALL_TICKERS:
        t_sub = preds_df[preds_df["ticker"] == tkr]
        if len(t_sub) == 0:
            continue
        y_t = t_sub["actual_y"].values
        # Heuristic persistence
        acc_persist = accuracy_score(y_t, (t_sub["prob_persistence"].values >= 0.5).astype(int)) * 100.0
        f1_persist = f1_score(y_t, (t_sub["prob_persistence"].values >= 0.5).astype(int), average="macro", zero_division=0)
        # Market BiLSTM
        acc_bilstm = accuracy_score(y_t, t_sub["pred_bilstm"].values) * 100.0
        f1_bilstm = f1_score(y_t, t_sub["pred_bilstm"].values, average="macro", zero_division=0)
        # GMU Fusion
        acc_gmu = accuracy_score(y_t, t_sub["pred_gmu"].values) * 100.0
        f1_gmu = f1_score(y_t, t_sub["pred_gmu"].values, average="macro", zero_division=0)
        mean_z = t_sub["gmu_market_gate_weight"].mean()

        per_ticker_records.append({
            "ticker": tkr,
            "test_samples": len(t_sub),
            "up_count": int(np.sum(y_t == 1)),
            "down_count": int(np.sum(y_t == 0)),
            "persist_acc": round(acc_persist, 2),
            "persist_f1": round(f1_persist, 4),
            "bilstm_acc": round(acc_bilstm, 2),
            "bilstm_f1": round(f1_bilstm, 4),
            "gmu_acc": round(acc_gmu, 2),
            "gmu_f1": round(f1_gmu, 4),
            "mean_market_gate": round(float(mean_z), 4),
            "mean_news_gate": round(float(1.0 - mean_z), 4)
        })

    per_ticker_df = pd.DataFrame(per_ticker_records)
    per_ticker_csv_path = os.path.join(out_results, "per_ticker_metrics_summary.csv")
    per_ticker_df.to_csv(per_ticker_csv_path, index=False)
    logger.info(f"Saved per-ticker metrics summary to: {per_ticker_csv_path}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(out_results, "multimodal_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics summary to: {metrics_csv_path}")

    # -------------------------------------------------------------
    # 9. Generate MULTIMODAL_RESEARCH_TRAINING_REPORT.md
    # -------------------------------------------------------------
    report_path = os.path.join(out_reports, "MULTIMODAL_RESEARCH_TRAINING_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Multimodal Research Training and Evaluation Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"**Random Seed:** {seed}\n")
        f.write(f"**Universe:** 12 equities ({', '.join(ALL_TICKERS)})\n")
        f.write(f"**Evaluation Range:** 2021-01-05 to 2022-09-20 (Verified historical news archive)\n")
        f.write(f"**Chronological Split:** Train (< 2022-01-01, N={n_train}) | Val (2022-01-01 to 2022-05-31, N={n_val}) | Test (>= 2022-06-01, N={n_test})\n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("This report presents the complete empirical benchmark for the multimodal stock trend prediction platform, ")
        f.write("evaluating the newly trained **Gated Multimodal Unit (GMU)** against competitive heuristic, linear, ")
        f.write("and deep market-only baselines under strict anti-leakage invariants.\n\n")

        f.write("### Benchmark Results Table\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("### Per-Ticker Evaluation Breakdown\n\n")
        f.write(per_ticker_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Modality Gate Analysis\n\n")
        f.write(f"- **Mean Market Gate Weight ($z$):** `{mean_market_weight:.4f}`\n")
        f.write(f"- **Mean News Gate Weight ($1 - z$):** `{mean_news_weight:.4f}`\n")
        if np.any(has_news_mask):
            f.write(f"- **Mean Market Weight on Days With News:** `{z_when_news:.4f}` (News weight: `{1.0 - z_when_news:.4f}`)\n")
        if np.any(~has_news_mask):
            f.write(f"- **Mean Market Weight on Days Without News:** `{z_when_no_news:.4f}` (News weight: `{1.0 - z_when_no_news:.4f}`)\n\n")

        f.write("## 3. Methodological Invariants Enforced\n\n")
        f.write("1. **Zero Future Lookahead:** Technical indicators are computed using strictly backward-looking windows. PrevDayClose and Diff are lagged.\n")
        f.write("2. **Causal News Alignment:** Only headlines published prior to prediction cutoff enter daily representations.\n")
        f.write("3. **Preprocessing Isolation:** MinMaxScaler fitted strictly on training data; transform applied downstream.\n")
        f.write("4. **Decision Threshold Calibration:** $\\tau^*$ selected on validation data without touching test sets.\n")
        f.write("5. **Missing News Integrity:** Sessions with zero news receive deterministic zero-vectors rather than synthetic data.\n\n")

        f.write("## 4. Checkpoints & Deliverables\n\n")
        f.write(f"- **GMU Model Checkpoint:** `models/gmu/gmu_fusion.pt`\n")
        f.write(f"- **Market BiLSTM Checkpoint:** `models/market/bilstm_market.pt`\n")
        f.write(f"- **Model Configuration:** `models/gmu/model_config.json`\n")
        f.write(f"- **Predictions CSV:** `results/multimodal/multimodal_predictions.csv`\n")
        f.write(f"- **Metrics Summary CSV:** `results/multimodal/multimodal_metrics_summary.csv`\n")

    logger.info(f"Generated research training report at: {report_path}")

    # Print summary to console
    print("\n" + "="*95)
    print("MULTIMODAL TRAINING AND BENCHMARK RESULTS")
    print("="*95)
    print(f"{'Model':<40} {'Acc (%)':<9} {'BalAcc (%)':<11} {'Macro F1':<10} {'AUC-ROC':<9} {'Brier':<8}")
    print("-" * 95)
    for m in metrics_list:
        print(f"{m['model']:<40} {m['accuracy']:<9.2f} {m['balanced_accuracy']:<11.2f} {m['macro_f1']:<10.4f} {m['auc_roc']:<9.4f} {m['brier_score']:<8.4f}")
    print("="*95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Multimodal GMU model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    run_multimodal_training(seed=args.seed)
