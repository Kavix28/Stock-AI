"""
Unit tests verifying dataset coverage and integrity across all 12 tickers.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ALL_TICKERS = [
    "AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ",
    "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"
]


def test_market_data_coverage_for_all_tickers():
    market_dir = os.path.join(PROJECT_ROOT, "data", "processed", "market")
    assert os.path.exists(market_dir), f"Market data directory missing: {market_dir}"

    for ticker in ALL_TICKERS:
        csv_path = os.path.join(market_dir, f"{ticker}.csv")
        assert os.path.exists(csv_path), f"Market CSV missing for {ticker}: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) > 500, f"Insufficient rows for {ticker}: {len(df)}"
        assert "date" in df.columns
        assert "close" in df.columns
        assert "Diff" in df.columns


def test_embedding_caches_exist_for_all_tickers():
    embed_dir = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings")
    assert os.path.exists(embed_dir), f"Embeddings directory missing: {embed_dir}"

    for ticker in ALL_TICKERS:
        npz_path = os.path.join(embed_dir, f"{ticker}_embeddings.npz")
        man_path = os.path.join(embed_dir, f"{ticker}_manifest.json")

        assert os.path.exists(npz_path), f"Embedding npz missing for {ticker}: {npz_path}"
        assert os.path.exists(man_path), f"Manifest JSON missing for {ticker}: {man_path}"

        with open(man_path, "r") as f:
            man = json.load(f)

        assert man["ticker"] == ticker
        assert man["embedding_dim"] == 768
        assert man["total_dates"] > 0
        assert man["dates_with_news"] > 0

        # Load npz arrays
        npz = np.load(npz_path, allow_pickle=True)
        assert "embeddings" in npz
        assert "dates" in npz
        assert "has_news" in npz
        assert npz["embeddings"].shape[1] == 768


def test_multimodal_evaluation_results_exist():
    res_dir = os.path.join(PROJECT_ROOT, "results", "multimodal")
    preds_csv = os.path.join(res_dir, "multimodal_predictions.csv")
    metrics_csv = os.path.join(res_dir, "multimodal_metrics_summary.csv")
    per_ticker_csv = os.path.join(res_dir, "per_ticker_metrics_summary.csv")

    assert os.path.exists(preds_csv), f"multimodal_predictions.csv missing: {preds_csv}"
    assert os.path.exists(metrics_csv), f"multimodal_metrics_summary.csv missing: {metrics_csv}"
    assert os.path.exists(per_ticker_csv), f"per_ticker_metrics_summary.csv missing: {per_ticker_csv}"

    preds_df = pd.read_csv(preds_csv)
    for ticker in ALL_TICKERS:
        t_sub = preds_df[preds_df["ticker"] == ticker]
        assert len(t_sub) == 77, f"Expected 77 test samples for {ticker}, got {len(t_sub)}"

    per_ticker_df = pd.read_csv(per_ticker_csv)
    assert len(per_ticker_df) == 12
    assert set(per_ticker_df["ticker"].tolist()) == set(ALL_TICKERS)
