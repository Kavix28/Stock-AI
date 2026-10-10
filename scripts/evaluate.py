"""
Master benchmark runner for refreshed 12-ticker stock prediction project.
Evaluates:
1. Training-Majority Baseline
2. Direction Persistence (Lag-1)
3. Next-Day LR (Lag-1 Endpoint)
4. Next-Day LR (20-Day Flat)
5. Individual Market-Only BiLSTM (Multi-seed: 42, 123, 7)
6. Shared Multi-Ticker Market-Only BiLSTM (Evaluated per-ticker and pooled)

Strict leakage-free protocols:
- Chronological train/val/test partitions.
- Scaler fitted strictly on training data.
- Threshold calibrated strictly on validation data.
- Exactly one score per eligible target date.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

# TensorFlow / Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.window_builder import build_chronological_windows

CORE_MARKET_FEATURES = [
    'high', 'close', 'open', 'volume', 'low', 'adjclose',
    'rsi', '12-Day EMA', '26-Day EMA', 'MACD', 'SAR',
    'Upper Band', 'Middle Band', 'Lower Band', 'Slow k', 'Slow d'
]

ALL_TICKERS = [
    ("AAPL", False),
    ("AMT", False),
    ("ATVI", True),  # Delisted Oct 2023
    ("BIO", False),
    ("JNJ", False),
    ("NDAQ", False),
    ("NFLX", False),
    ("NVDA", False),
    ("PLD", False),
    ("SBUX", False),
    ("SCHW", False),
    ("TSLA", False)
]

def calibrate_threshold_on_val(y_val, val_probs):
    """Finds optimal decision threshold on validation set maximizing Macro F1."""
    best_tau = 0.50
    best_f1 = -1.0
    for tau in np.arange(0.30, 0.71, 0.02):
        preds = (val_probs >= tau).astype(int)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    d_count = int(np.sum(y_pred == 0))
    u_count = int(np.sum(y_pred == 1))
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "down_preds": d_count,
        "up_preds": u_count
    }

def build_bilstm_model():
    model = Sequential([
        Input(shape=(20, 16)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def run_benchmarks(args):
    os.makedirs(args.results_dir, exist_ok=True)
    pred_dir = os.path.join(args.results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    print("================================================================================")
    print("      EXECUTING REFRESHED CHRONOLOGICAL BENCHMARKS ACROSS 12 TICKERS             ")
    print(f"      Train Cutoff: {args.train_cutoff} | Val Cutoff: {args.val_cutoff}         ")
    print(f"      Seeds: {args.seeds} | Epochs: {args.epochs} | Batch Size: {args.batch_size} ")
    print("================================================================================")

    ticker_window_cache = {}
    partition_manifest_rows = []
    all_ticker_metrics = []

    # 1. Build windows and evaluate individual baselines for each ticker
    for ticker, is_delisted in ALL_TICKERS:
        fpath = os.path.join(args.stock_dir, f"{ticker}.csv")
        if not os.path.exists(fpath):
            print(f"[!] Warning: Missing stock file for {ticker}, skipping.")
            continue

        df = pd.read_csv(fpath)
        
        # Determine split cutoffs
        if is_delisted:
            tr_cut = "2021-10-01"
            val_cut = "2022-01-01"
            notes = "Delisted Oct 2023; evaluated on historical 180-target test partition"
        else:
            tr_cut = args.train_cutoff
            val_cut = args.val_cutoff
            notes = "Standard refreshed forward test horizon"

        w_data = build_chronological_windows(
            df,
            core_features=CORE_MARKET_FEATURES,
            lookback=20,
            train_cutoff=tr_cut,
            val_cutoff=val_cut
        )
        ticker_window_cache[ticker] = w_data

        tr = w_data["train"]
        val = w_data["val"]
        te = w_data["test"]

        partition_manifest_rows.append({
            "ticker": ticker,
            "train_samples": len(tr["y"]),
            "train_start": str(tr["dates"][0]),
            "train_end": str(tr["dates"][-1]),
            "val_samples": len(val["y"]),
            "val_start": str(val["dates"][0]),
            "val_end": str(val["dates"][-1]),
            "test_samples": len(te["y"]),
            "test_start": str(te["dates"][0]),
            "test_end": str(te["dates"][-1]),
            "test_down_count": int(np.sum(te["y"] == 0)),
            "test_up_count": int(np.sum(te["y"] == 1)),
            "notes": notes
        })

        print(f"\n[*] Evaluating Ticker: {ticker} (Test N = {len(te['y'])}, {te['dates'][0]} to {te['dates'][-1]})")

        # Predictions dictionary for CSV export
        pred_export = {
            "target_date": te["dates"],
            "ground_truth": te["y"]
        }

        # --- Baseline 1: Training Majority Baseline ---
        tr_majority = int(np.bincount(tr["y"]).argmax())
        b_maj_preds = np.full(len(te["y"]), tr_majority, dtype=int)
        pred_export["pred_majority"] = b_maj_preds
        m_maj = compute_metrics(te["y"], b_maj_preds)
        all_ticker_metrics.append({
            "ticker": ticker,
            "model_type": "Majority Baseline",
            "seed": "N/A",
            "threshold": 0.50,
            **m_maj
        })
        print(f"    [1] Majority Baseline:     Acc: {m_maj['accuracy']*100:.2f}% | Macro F1: {m_maj['macro_f1']:.4f}")

        # --- Baseline 2: Direction Persistence (Lag-1) ---
        b_persist_preds = te["persist_preds"]
        pred_export["pred_persistence"] = b_persist_preds
        m_persist = compute_metrics(te["y"], b_persist_preds)
        all_ticker_metrics.append({
            "ticker": ticker,
            "model_type": "Direction Persistence (Lag-1)",
            "seed": "N/A",
            "threshold": 0.50,
            **m_persist
        })
        print(f"    [2] Direction Persistence: Acc: {m_persist['accuracy']*100:.2f}% | Macro F1: {m_persist['macro_f1']:.4f}")

        # --- Baseline 3: Next-Day LR (t-1 Endpoint) ---
        lr_endpoint = LogisticRegression(max_iter=1000, random_state=42)
        lr_endpoint.fit(tr["X_lag1"], tr["y"])
        b_lr_end_preds = lr_endpoint.predict(te["X_lag1"])
        pred_export["pred_lr_endpoint"] = b_lr_end_preds
        m_lr_end = compute_metrics(te["y"], b_lr_end_preds)
        all_ticker_metrics.append({
            "ticker": ticker,
            "model_type": "Next-Day LR (t-1 Endpoint)",
            "seed": "N/A",
            "threshold": 0.50,
            **m_lr_end
        })
        print(f"    [3] LR Endpoint (t-1):     Acc: {m_lr_end['accuracy']*100:.2f}% | Macro F1: {m_lr_end['macro_f1']:.4f}")

        # --- Baseline 4: Next-Day LR (20-Day Flat) ---
        lr_flat = LogisticRegression(max_iter=1000, random_state=42)
        lr_flat.fit(tr["X_flat"], tr["y"])
        b_lr_flat_preds = lr_flat.predict(te["X_flat"])
        pred_export["pred_lr_flat"] = b_lr_flat_preds
        m_lr_flat = compute_metrics(te["y"], b_lr_flat_preds)
        all_ticker_metrics.append({
            "ticker": ticker,
            "model_type": "Next-Day LR (20-Day Flat)",
            "seed": "N/A",
            "threshold": 0.50,
            **m_lr_flat
        })
        print(f"    [4] LR Flat (20-Day):      Acc: {m_lr_flat['accuracy']*100:.2f}% | Macro F1: {m_lr_flat['macro_f1']:.4f}")

        # --- Individual Market-Only BiLSTM (Multi-Seed) ---
        seed_accs = []
        seed_f1s = []
        seed_baccs = []
        test_prob_ensemble = np.zeros(len(te["y"]))

        for seed in args.seeds:
            tf.keras.utils.set_random_seed(seed)
            np.random.seed(seed)
            model = build_bilstm_model()
            es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            model.fit(
                tr["X_seq"], tr["y"],
                validation_data=(val["X_seq"], val["y"]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[es],
                verbose=0
            )

            val_probs = model.predict(val["X_seq"], verbose=0).flatten()
            te_probs = model.predict(te["X_seq"], verbose=0).flatten()
            test_prob_ensemble += te_probs / len(args.seeds)
            pred_export[f"prob_bilstm_seed_{seed}"] = te_probs

            # Calibrate threshold on validation only
            tau_star = calibrate_threshold_on_val(val["y"], val_probs)
            te_preds = (te_probs >= tau_star).astype(int)
            pred_export[f"pred_bilstm_seed_{seed}"] = te_preds

            m_seed = compute_metrics(te["y"], te_preds)
            seed_accs.append(m_seed["accuracy"])
            seed_f1s.append(m_seed["macro_f1"])
            seed_baccs.append(m_seed["balanced_accuracy"])

            all_ticker_metrics.append({
                "ticker": ticker,
                "model_type": "Individual BiLSTM",
                "seed": seed,
                "threshold": tau_star,
                **m_seed
            })

        # Save individual prediction CSV
        pred_df = pd.DataFrame(pred_export)
        pred_csv_path = os.path.join(pred_dir, f"{ticker}_predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        mean_f1 = np.mean(seed_f1s)
        std_f1 = np.std(seed_f1s)
        print(f"    [5] Individual BiLSTM:     Acc: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%) | Macro F1: {mean_f1:.4f} (±{std_f1:.4f}) across {len(args.seeds)} seeds")

    # Save Partition Manifest
    part_df = pd.DataFrame(partition_manifest_rows)
    part_csv_path = os.path.join(args.results_dir, "train_validation_test_manifest.csv")
    part_df.to_csv(part_csv_path, index=False)
    print(f"\n[+] Saved Partition Manifest -> {part_csv_path}")

    # =========================================================================
    # Phase 8: Shared Multi-Ticker Market-Only BiLSTM
    # =========================================================================
    print("\n================================================================================")
    print("      TRAINING SHARED MULTI-TICKER MARKET-ONLY BiLSTM (PHASE 8)                 ")
    print("================================================================================")

    # Eligible active tickers for shared pooling
    active_tickers = [t for t, is_del in ALL_TICKERS if not is_del and t in ticker_window_cache]
    print(f"Pooling {len(active_tickers)} active tickers: {active_tickers}")

    pooled_tr_X = np.concatenate([ticker_window_cache[t]["train"]["X_seq"] for t in active_tickers], axis=0)
    pooled_tr_y = np.concatenate([ticker_window_cache[t]["train"]["y"] for t in active_tickers], axis=0)
    pooled_val_X = np.concatenate([ticker_window_cache[t]["val"]["X_seq"] for t in active_tickers], axis=0)
    pooled_val_y = np.concatenate([ticker_window_cache[t]["val"]["y"] for t in active_tickers], axis=0)

    print(f"Pooled Training Set:   {pooled_tr_X.shape[0]} sequences (20x16)")
    print(f"Pooled Validation Set: {pooled_val_X.shape[0]} sequences (20x16)")

    # Train shared model on seed 42
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    shared_model = build_bilstm_model()
    es_shared = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    shared_model.fit(
        pooled_tr_X, pooled_tr_y,
        validation_data=(pooled_val_X, pooled_val_y),
        epochs=min(args.epochs, 10),
        batch_size=args.batch_size * 2,
        callbacks=[es_shared],
        verbose=1
    )

    # Calibrate shared threshold on pooled validation set
    pooled_val_probs = shared_model.predict(pooled_val_X, verbose=0).flatten()
    shared_tau = calibrate_threshold_on_val(pooled_val_y, pooled_val_probs)
    print(f"[+] Shared Model Calibrated Threshold: tau* = {shared_tau:.2f}")

    pooled_test_y_all = []
    pooled_test_preds_all = []

    # Evaluate shared model per ticker on held-out test horizon
    print("\n--- Shared Model Evaluation Across Tickers ---")
    for t in active_tickers:
        te = ticker_window_cache[t]["test"]
        te_probs = shared_model.predict(te["X_seq"], verbose=0).flatten()
        te_preds = (te_probs >= shared_tau).astype(int)
        
        m_shared_t = compute_metrics(te["y"], te_preds)
        all_ticker_metrics.append({
            "ticker": t,
            "model_type": "Shared BiLSTM (Pooled)",
            "seed": 42,
            "threshold": shared_tau,
            **m_shared_t
        })
        print(f"    {t:<5} | Acc: {m_shared_t['accuracy']*100:.2f}% | Macro F1: {m_shared_t['macro_f1']:.4f} | Balanced: {m_shared_t['balanced_accuracy']*100:.2f}%")

        pooled_test_y_all.extend(te["y"].tolist())
        pooled_test_preds_all.extend(te_preds.tolist())

    # Overall pooled metrics
    pooled_metrics = compute_metrics(np.array(pooled_test_y_all), np.array(pooled_test_preds_all))
    print(f"\n[+] OVERALL POOLED TEST PERFORMANCE (N = {len(pooled_test_y_all)}):")
    print(f"    Accuracy:          {pooled_metrics['accuracy']*100:.2f}%")
    print(f"    Balanced Accuracy: {pooled_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1:          {pooled_metrics['macro_f1']:.4f}")
    print(f"    Confusion Matrix:  TN={pooled_metrics['tn']}, FP={pooled_metrics['fp']}, FN={pooled_metrics['fn']}, TP={pooled_metrics['tp']}")
    print(f"    Predicted:         Down={pooled_metrics['down_preds']}, Up={pooled_metrics['up_preds']}")

    # Save Metrics CSVs
    metrics_df = pd.DataFrame(all_ticker_metrics)
    metrics_path = os.path.join(args.results_dir, "metrics_by_ticker.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[+] Saved Ticker Metrics -> {metrics_path}")

    pooled_df = pd.DataFrame([{
        "evaluation_scope": "Pooled Across 11 Active Tickers",
        "total_test_observations": len(pooled_test_y_all),
        "threshold": shared_tau,
        **pooled_metrics
    }])
    pooled_path = os.path.join(args.results_dir, "pooled_metrics.csv")
    pooled_df.to_csv(pooled_path, index=False)
    print(f"[+] Saved Pooled Metrics -> {pooled_path}")

    print("================================================================================")
    print("                      ALL BENCHMARKS COMPLETED SUCCESSFULLY                     ")
    print("================================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks across refreshed ticker datasets")
    parser.add_argument("--stock-dir", default="data/stock_data_latest", help="Stock CSV directory")
    parser.add_argument("--results-dir", default="results/latest", help="Results directory")
    parser.add_argument("--train-cutoff", default="2023-01-01", help="Train partition cutoff")
    parser.add_argument("--val-cutoff", default="2024-07-01", help="Validation partition cutoff")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_benchmarks(args)

if __name__ == "__main__":
    main()
