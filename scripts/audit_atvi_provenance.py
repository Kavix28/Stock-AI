"""
Audit ATVI market data files, provenance, date coverage, OHLC validity, and paper period alignment.
"""
import os
import pandas as pd
import json

ROOT = r"C:\PROJECTS\multimodal-stock-ai"

raw_atvi = os.path.join(ROOT, "data", "raw", "market", "ATVI.csv")
proc_atvi = os.path.join(ROOT, "data", "processed", "market", "ATVI.csv")

print("=== ATVI FILE AUDIT ===")
print("Raw exists:", os.path.exists(raw_atvi))
print("Processed exists:", os.path.exists(proc_atvi))

if os.path.exists(proc_atvi):
    df_proc = pd.read_csv(proc_atvi)
    print(f"\nProcessed ATVI shape: {df_proc.shape}")
    print(f"Columns: {list(df_proc.columns)}")
    print(f"Date min: {df_proc['date'].min()}, Date max: {df_proc['date'].max()}")
    print(f"Duplicate dates: {df_proc['date'].duplicated().sum()}")
    print(f"Null counts:\n{df_proc.isnull().sum()[df_proc.isnull().sum() > 0]}")
    
    # Check OHLC bounds
    invalid_hl = (df_proc['high'] < df_proc['low']).sum()
    invalid_ho = (df_proc['high'] < df_proc['open']).sum()
    invalid_hc = (df_proc['high'] < df_proc['close']).sum()
    invalid_lo = (df_proc['low'] > df_proc['open']).sum()
    invalid_lc = (df_proc['low'] > df_proc['close']).sum()
    print(f"\nOHLC boundary violations:")
    print(f"  high < low: {invalid_hl}")
    print(f"  high < open: {invalid_ho}")
    print(f"  high < close: {invalid_hc}")
    print(f"  low > open: {invalid_lo}")
    print(f"  low > close: {invalid_lc}")

if os.path.exists(raw_atvi):
    df_raw = pd.read_csv(raw_atvi)
    print(f"\nRaw ATVI shape: {df_raw.shape}")
    print(f"Date min: {df_raw['date'].min() if 'date' in df_raw.columns else df_raw['Date'].min()}")
    print(f"Date max: {df_raw['date'].max() if 'date' in df_raw.columns else df_raw['Date'].max()}")

# Check fallback provider implementation
fallback_src = os.path.join(ROOT, "src", "stock_ai", "data", "market", "fallback_provider.py")
if os.path.exists(fallback_src):
    print("\n=== FALLBACK PROVIDER CODE ===")
    with open(fallback_src, "r") as f:
        print(f.read())
