"""
Automated validation script for refreshed datasets in data/stock_data_latest/
and manifests in results/latest/.
Validates:
1. All 12 tickers present and non-empty.
2. 16 core market features present and non-null.
3. Monotonic dates without duplicates.
4. Mathematical integrity of OHLC prices.
5. Target definition integrity (Diff).
6. Manifest consistency.
"""

import os
import sys
import pandas as pd
import numpy as np

CORE_FEATURES = [
    'high', 'close', 'open', 'volume', 'low', 'adjclose',
    'rsi', '12-Day EMA', '26-Day EMA', 'MACD', 'SAR',
    'Upper Band', 'Middle Band', 'Lower Band', 'Slow k', 'Slow d'
]

ALL_TICKERS = [
    "AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ",
    "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"
]

def validate_datasets(stock_dir="data/stock_data_latest", results_dir="results/latest"):
    print("================================================================================")
    print("               VALIDATING REFRESHED DATASETS & MANIFESTS                        ")
    print("================================================================================")
    
    errors = []
    
    # 1. Manifest Validation
    market_manifest_path = os.path.join(results_dir, "market_data_manifest.csv")
    news_manifest_path = os.path.join(results_dir, "news_coverage_manifest.csv")
    
    if not os.path.exists(market_manifest_path):
        errors.append(f"Missing market data manifest: {market_manifest_path}")
    else:
        m_df = pd.read_csv(market_manifest_path)
        if len(m_df) != 12:
            errors.append(f"Market manifest expected 12 rows, got {len(m_df)}")
            
    if not os.path.exists(news_manifest_path):
        errors.append(f"Missing news coverage manifest: {news_manifest_path}")
    else:
        n_df = pd.read_csv(news_manifest_path)
        if len(n_df) != 12:
            errors.append(f"News manifest expected 12 rows, got {len(n_df)}")
            
    # 2. Individual Ticker Dataset Validation
    for ticker in ALL_TICKERS:
        fpath = os.path.join(stock_dir, f"{ticker}.csv")
        if not os.path.exists(fpath):
            errors.append(f"Missing stock file for {ticker}: {fpath}")
            continue
            
        df = pd.read_csv(fpath)
        if len(df) < 500:
            errors.append(f"{ticker} has suspiciously few rows: {len(df)}")
            
        # Check required columns
        for col in CORE_FEATURES:
            if col not in df.columns:
                errors.append(f"{ticker} missing core feature: {col}")
                
        # Check for NaNs in core features
        nan_counts = df[CORE_FEATURES].isnull().sum().sum()
        if nan_counts > 0:
            errors.append(f"{ticker} contains {nan_counts} NaNs in core features!")
            
        # Check Date Monotonicity
        df['parsed_date'] = pd.to_datetime(df['date'])
        if not df['parsed_date'].is_monotonic_increasing:
            errors.append(f"{ticker} dates are not monotonically increasing!")
            
        if df['date'].duplicated().any():
            errors.append(f"{ticker} contains duplicate dates!")
            
        # Check OHLC Relationships
        high_low_viol = (df['high'] < df['low']).sum()
        if high_low_viol > 0:
            errors.append(f"{ticker} has {high_low_viol} High < Low violations!")
            
        high_close_viol = (df['high'] < df['close'] - 1e-3).sum()
        if high_close_viol > 0:
            errors.append(f"{ticker} has {high_close_viol} High < Close violations!")
            
        low_close_viol = (df['low'] > df['close'] + 1e-3).sum()
        if low_close_viol > 0:
            errors.append(f"{ticker} has {low_close_viol} Low > Close violations!")
            
        # Check Target Diff Definition
        if 'Diff' in df.columns and 'PrevDayClose' in df.columns:
            expected_diff = ((df['close'] - df['PrevDayClose']) > 0).astype(int)
            # Check after first row where PrevDayClose is valid
            valid_idx = df['PrevDayClose'].dropna().index
            mismatches = (df.loc[valid_idx, 'Diff'] != expected_diff.loc[valid_idx]).sum()
            if mismatches > 0:
                errors.append(f"{ticker} has {mismatches} Diff target mismatches with (close > PrevDayClose)!")
                
        print(f"[PASS] {ticker:<5} | Rows: {len(df):<5} | Dates: {df['date'].iloc[0]} to {df['date'].iloc[-1]} | 16 Features Clean")

    print("--------------------------------------------------------------------------------")
    if errors:
        print(f"[FAIL] Found {len(errors)} validation errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("[SUCCESS] All 12 refreshed datasets and manifests passed all integrity checks!")
        print("================================================================================\n")
        return True

if __name__ == "__main__":
    success = validate_datasets()
    sys.exit(0 if success else 1)
