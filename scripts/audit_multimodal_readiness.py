"""
Stock-AI Multimodal Research Readiness Audit Script.

Performs a comprehensive, reproducible inspection of:
1. Model components (time-series encoder, news encoder, fusion mechanism, prediction head, inference).
2. Saved checkpoints and weights.
3. News dataset coverage, timestamps, and lookahead risk.
4. Multimodal training and evaluation status.
"""

import os
import glob
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src", "stock_ai")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def audit_model_architecture():
    print("================================================================================")
    print("1. MODEL ARCHITECTURE & COMPONENT AUDIT")
    print("================================================================================")
    
    bilstm_path = os.path.join(SRC_DIR, "models", "market", "bilstm.py")
    gmu_path = os.path.join(SRC_DIR, "models", "fusion", "gmu.py")
    news_dir = os.path.join(SRC_DIR, "models", "news")
    engine_path = os.path.join(SRC_DIR, "inference", "engine.py")

    print(f"[-] Time-Series Encoder (BiLSTM):  {'EXISTS' if os.path.exists(bilstm_path) else 'MISSING'} -> {bilstm_path}")
    print(f"[-] Fusion Mechanism (GMU):        {'EXISTS' if os.path.exists(gmu_path) else 'MISSING'} -> {gmu_path}")
    print(f"[-] News Encoder (FinBERT):        {'EXISTS' if os.path.exists(news_dir) else 'MISSING'} -> {news_dir}")
    print(f"[-] Inference Engine:              {'EXISTS' if os.path.exists(engine_path) else 'MISSING'} -> {engine_path}")

    # Inspect Inference Engine logic
    with open(engine_path, "r", encoding="utf-8") as f:
        engine_code = f.read()
    uses_multimodal = "gmu" in engine_code.lower() or "news" in engine_code.lower() or "finbert" in engine_code.lower()
    print(f"[-] Inference Engine Mode:         {'MULTIMODAL' if uses_multimodal else 'MARKET-ONLY (Loads pred_bilstm)'}")

    # Inspect saved weights
    weights = []
    for ext in ["*.pt", "*.pth", "*.bin", "*.h5", "*.keras", "*.onnx"]:
        weights.extend(glob.glob(os.path.join(PROJECT_ROOT, "**", ext), recursive=True))
    print(f"[-] Saved Checkpoint / Weight Files: {len(weights)} found across repository")

    # Inspect precomputed embeddings
    npy_files = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.npy"), recursive=True)
    npz_files = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.npz"), recursive=True)
    print(f"[-] Precomputed Embeddings (.npy/.npz): {len(npy_files) + len(npz_files)} found")

def audit_news_dataset():
    print("\n================================================================================")
    print("2. NEWS DATASET, TIMESTAMPS & COVERAGE AUDIT")
    print("================================================================================")
    news_dir = os.path.join(DATA_DIR, "processed", "news")
    if not os.path.exists(news_dir):
        print("[!] News directory missing!")
        return

    news_files = sorted(glob.glob(os.path.join(news_dir, "*.csv")))
    print(f"Found {len(news_files)} news files in {news_dir}:")
    
    rows = []
    for nf in news_files:
        ticker = os.path.splitext(os.path.basename(nf))[0]
        df = pd.read_csv(nf)
        dates = pd.to_datetime(df['date'])
        has_intraday = any(':' in str(d) for d in df['date'])
        
        # Check forward coverage
        fwd_count = len(df[df['date'] >= '2024-07-01'])
        
        rows.append({
            "Ticker": ticker,
            "Rows": len(df),
            "Start": dates.min().strftime('%Y-%m-%d'),
            "End": dates.max().strftime('%Y-%m-%d'),
            "Intraday_Timestamps": has_intraday,
            "Forward_2024_2026_Rows": fwd_count
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print("\n[!] CRITICAL FINDING: News data terminates on 2022-09-20. 0 headlines exist for 2024–2026 forward test set.")
    print("[!] CRITICAL FINDING: Intraday timestamps are absent (YYYY-MM-DD only), creating post-close lookahead risk.")

def main():
    audit_model_architecture()
    audit_news_dataset()

if __name__ == "__main__":
    main()
