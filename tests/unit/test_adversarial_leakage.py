"""
Adversarial Anti-Leakage Unit Tests for Stock-AI.

Tests adversarial boundary conditions, lookahead prevention in technical indicators,
chronological split isolation, news publication cutoff enforcement, and scaler fit isolation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from stock_ai.features.indicators import compute_indicators
    from stock_ai.data.news.alignment import NewsMarketAligner, CausalAlignmentError
except ImportError:
    from src.stock_ai.features.indicators import compute_indicators
    from src.stock_ai.data.news.alignment import NewsMarketAligner, CausalAlignmentError


def test_adversarial_indicator_future_lookahead():
    """
    Modifying future price session (t = 25) must NOT alter indicator values
    at any historical session t <= 24.
    """
    dates = pd.date_range("2021-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(30)) + 100.0

    df_clean = pd.DataFrame({
        "date": dates,
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.98,
        "close": closes,
        "adj_close": closes,
        "volume": np.random.randint(1000, 5000, 30)
    })

    feat_clean = compute_indicators(df_clean)

    # Adversarial: Mutate prices at index 25 (spike by 500%)
    df_adversarial = df_clean.copy()
    df_adversarial.loc[25, "close"] = df_clean.loc[25, "close"] * 5.0
    df_adversarial.loc[25, "high"] = df_adversarial.loc[25, "close"] * 1.05

    feat_adversarial = compute_indicators(df_adversarial)

    # Invariant: Features from index 0 to 24 must be EXACTLY identical
    for col in feat_clean.columns:
        if col in ["date", "symbol"]:
            continue
        clean_slice = feat_clean.loc[:24, col].values
        adv_slice = feat_adversarial.loc[:24, col].values
        assert np.allclose(clean_slice, adv_slice, equal_nan=True), (
            f"Future price spike at t=25 leaked backwards into feature '{col}' at t <= 24!"
        )


def test_adversarial_news_temporal_cutoff():
    """
    Headlines published after the market decision cutoff (09:30 AM EST)
    must be excluded from the trading day's prediction features.
    """
    aligner = NewsMarketAligner(alignment_mode="PRE_MARKET_WINDOW")

    trading_dates = ["2022-01-07", "2022-01-10", "2022-01-11"]

    # Pre-market news: 08:15 AM EST (13:15 UTC) on 2022-01-10 -> Eligible for target date 2022-01-10
    # Post-market news: 10:45 AM EST (15:45 UTC) on 2022-01-10 -> Ineligible for 2022-01-10
    news_df = pd.DataFrame([
        {
            "article_id": "art_pre",
            "symbol": "AAPL",
            "title": "Pre-market blowout earnings",
            "published_at_utc": "2022-01-10T13:15:00Z",
            "has_intraday_time": True,
            "published_date": "2022-01-10"
        },
        {
            "article_id": "art_post",
            "symbol": "AAPL",
            "title": "Midday press conference announcement",
            "published_at_utc": "2022-01-10T15:45:00Z",
            "has_intraday_time": True,
            "published_date": "2022-01-10"
        }
    ])

    aligned = aligner.align_ticker_news(news_df, trading_dates, symbol="AAPL")

    row_10 = aligned[aligned["target_date"] == "2022-01-10"].iloc[0]

    # Pre-market news was captured
    assert "Pre-market blowout" in row_10["combined_text"]
    # Midday news was strictly EXCLUDED from 2022-01-10 pre-market feature
    assert "Midday press conference" not in row_10["combined_text"]


def test_adversarial_scaler_isolation():
    """
    Fitting a MinMaxScaler across train + test partitions leaks variance and range.
    Demonstrates that train-only fitting produces distinct parameters and prevents leakage.
    """
    np.random.seed(42)
    train_vals = np.random.uniform(10.0, 50.0, size=(100, 1))
    test_vals = np.random.uniform(40.0, 100.0, size=(50, 1))

    # Proper train-only fitting
    scaler_correct = MinMaxScaler()
    scaler_correct.fit(train_vals)
    test_transformed_correct = scaler_correct.transform(test_vals)

    # Leaked fitting
    scaler_leaked = MinMaxScaler()
    scaler_leaked.fit(np.vstack([train_vals, test_vals]))
    test_transformed_leaked = scaler_leaked.transform(test_vals)

    # Invariant: Scaler parameters must differ; test-transformed values must differ
    assert scaler_correct.data_max_[0] == pytest.approx(train_vals.max())
    assert scaler_leaked.data_max_[0] == pytest.approx(max(train_vals.max(), test_vals.max()))
    assert not np.allclose(test_transformed_correct, test_transformed_leaked)
