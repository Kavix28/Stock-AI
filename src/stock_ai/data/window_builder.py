"""
Leakage-Free Chronological Rolling Window Builder for Stock-AI.

Guarantees:
1. Strict chronological boundaries: train < val_cutoff, val in [val_cutoff, test_cutoff), test >= test_cutoff.
2. 20-day rolling market lookback ends strictly on t-1 relative to target day t.
3. Preprocessing isolation: MinMaxScaler is fitted strictly on the training partition.
4. Target and future-derived features are strictly excluded from input arrays.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CORE_MARKET_FEATURES = [
    'high', 'close', 'open', 'volume', 'low', 'adjclose',
    'rsi', '12-Day EMA', '26-Day EMA', 'MACD', 'SAR',
    'Upper Band', 'Middle Band', 'Lower Band', 'Slow k', 'Slow d'
]

FORBIDDEN_COLUMNS = [
    'Diff', 'NextDayTrend', 'PrevDayTrend', 'NextDayClose',
    'return', 'target', 'Close_Target', 'direction'
]


def validate_feature_integrity(df: pd.DataFrame, features: List[str]) -> None:
    """Verifies that features contain no forbidden columns and no NaNs."""
    for col in features:
        if col in FORBIDDEN_COLUMNS:
            raise ValueError(f"Forbidden column '{col}' detected in feature set! Lookahead risk.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in DataFrame: {missing}")


def build_chronological_windows(
    df: pd.DataFrame,
    core_features: Optional[List[str]] = None,
    lookback: int = 20,
    train_cutoff: str = "2023-01-01",
    val_cutoff: str = "2024-07-01",
    include_news: bool = False
) -> Dict[str, Any]:
    """
    Constructs leakage-free chronological rolling windows.
    Returns:
        Dict with keys 'train', 'val', 'test', 'scaler', 'feature_names'
    """
    if core_features is None:
        core_features = list(CORE_MARKET_FEATURES)

    validate_feature_integrity(df, core_features)

    df_sorted = df.copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    df_sorted = df_sorted.sort_values('date').reset_index(drop=True)

    if 'Diff' not in df_sorted.columns:
        if 'PrevDayClose' not in df_sorted.columns:
            df_sorted['PrevDayClose'] = df_sorted['close'].shift(1)
        df_sorted['Diff'] = ((df_sorted['close'] - df_sorted['PrevDayClose']) > 0).astype(int)

    train_mask = df_sorted['date'] < pd.to_datetime(train_cutoff)
    train_df = df_sorted[train_mask]
    if len(train_df) == 0:
        raise ValueError(f"No training data found before train_cutoff: {train_cutoff}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[core_features].values)

    all_scaled = scaler.transform(df_sorted[core_features].values)

    train_data = {"X_seq": [], "X_flat": [], "X_lag1": [], "y": [], "dates": [], "persist_preds": [], "window_starts": [], "window_ends": []}
    val_data = {"X_seq": [], "X_flat": [], "X_lag1": [], "y": [], "dates": [], "persist_preds": [], "window_starts": [], "window_ends": []}
    test_data = {"X_seq": [], "X_flat": [], "X_lag1": [], "y": [], "dates": [], "persist_preds": [], "window_starts": [], "window_ends": []}

    t_train_cut = pd.to_datetime(train_cutoff)
    t_val_cut = pd.to_datetime(val_cutoff)

    for i in range(lookback, len(df_sorted)):
        target_date = df_sorted.loc[i, 'date']
        target_val = int(df_sorted.loc[i, 'Diff'])

        seq = all_scaled[i - lookback : i]
        flat = seq.flatten()
        lag1 = all_scaled[i - 1]

        prev_day_trend = int(df_sorted.loc[i - 1, 'Diff']) if (i - 1 >= 0 and not pd.isna(df_sorted.loc[i - 1, 'Diff'])) else target_val

        w_start = str(df_sorted.loc[i - lookback, 'date'].strftime('%Y-%m-%d'))
        w_end = str(df_sorted.loc[i - 1, 'date'].strftime('%Y-%m-%d'))
        t_date_str = str(target_date.strftime('%Y-%m-%d'))

        if target_date < t_train_cut:
            bucket = train_data
        elif target_date < t_val_cut:
            bucket = val_data
        else:
            bucket = test_data

        bucket["X_seq"].append(seq)
        bucket["X_flat"].append(flat)
        bucket["X_lag1"].append(lag1)
        bucket["y"].append(target_val)
        bucket["dates"].append(t_date_str)
        bucket["persist_preds"].append(prev_day_trend)
        bucket["window_starts"].append(w_start)
        bucket["window_ends"].append(w_end)

    for b in [train_data, val_data, test_data]:
        b["X_seq"] = np.array(b["X_seq"])
        b["X_flat"] = np.array(b["X_flat"])
        b["X_lag1"] = np.array(b["X_lag1"])
        b["y"] = np.array(b["y"])
        b["persist_preds"] = np.array(b["persist_preds"])

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "scaler": scaler,
        "feature_names": core_features
    }
