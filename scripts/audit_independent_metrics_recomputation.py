"""
Independent Recomputation and Audit of Multimodal Benchmark Metrics.
Examines ground-truth labels, predictions, probability outputs, confusion matrices,
class distributions, calibration thresholds, and ablation behavior.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    brier_score_loss, confusion_matrix
)

project_root = r"C:\PROJECTS\multimodal-stock-ai"
preds_path = os.path.join(project_root, "results", "multimodal", "multimodal_predictions.csv")

if not os.path.exists(preds_path):
    print(f"ERROR: Predictions file not found at {preds_path}")
    sys.exit(1)

df = pd.read_csv(preds_path)
print(f"Loaded predictions: {len(df)} rows across {df['ticker'].nunique()} tickers.")
print(f"Date range: {df['target_date'].min()} to {df['target_date'].max()}")
print(f"Unique tickers: {sorted(df['ticker'].unique())}")

y_true = df["actual_y"].values
n_total = len(y_true)
n_up = int(np.sum(y_true == 1))
n_down = int(np.sum(y_true == 0))
pct_up = n_up / n_total * 100.0
pct_down = n_down / n_total * 100.0

print("\n" + "=" * 90)
print("1. GROUND TRUTH CLASS DISTRIBUTION")
print("=" * 90)
print(f"Total test samples: {n_total}")
print(f"Class 1 (Up):   {n_up} ({pct_up:.2f}%)")
print(f"Class 0 (Down): {n_down} ({pct_down:.2f}%)")

# Check for NaNs or infinites
for col in ["prob_persistence", "prob_lr_endpoint", "prob_lr_flat", "prob_bilstm", "prob_gmu", "prob_gmu_ablation"]:
    n_nans = df[col].isna().sum()
    n_infs = np.isinf(df[col]).sum()
    min_v, max_v = df[col].min(), df[col].max()
    print(f"Column {col:<20}: NaNs={n_nans}, Infs={n_infs}, Min={min_v:.4f}, Max={max_v:.4f}")

models = {
    "Direction Persistence": {
        "prob": df["prob_persistence"].values,
        "pred": (df["prob_persistence"].values >= 0.5).astype(int),
        "tau": 0.50
    },
    "Logistic Regression (Flat)": {
        "prob": df["prob_lr_flat"].values,
        "pred": (df["prob_lr_flat"].values >= 0.56).astype(int),
        "tau": 0.56
    },
    "Logistic Regression (Endpoint)": {
        "prob": df["prob_lr_endpoint"].values,
        "pred": (df["prob_lr_endpoint"].values >= 0.54).astype(int),
        "tau": 0.54
    },
    "Market-Only BiLSTM": {
        "prob": df["prob_bilstm"].values,
        "pred": df["pred_bilstm"].values,
        "tau": 0.53
    },
    "GMU Fusion": {
        "prob": df["prob_gmu"].values,
        "pred": df["pred_gmu"].values,
        "tau": 0.52
    },
    "GMU Zero-News Ablation": {
        "prob": df["prob_gmu_ablation"].values,
        "pred": (df["prob_gmu_ablation"].values >= 0.52).astype(int),
        "tau": 0.52
    },
    "Training-Majority (Down=0)": {
        "prob": np.zeros(n_total),
        "pred": np.zeros(n_total, dtype=int),
        "tau": 0.50
    }
}

print("\n" + "=" * 90)
print("2. INDEPENDENT RECOMPUTATION OF BENCHMARK METRICS")
print("=" * 90)

results = []
for name, m in models.items():
    p = m["prob"]
    y_pred = m["pred"]

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, p)
    except Exception:
        auc = 0.5

    brier = brier_score_loss(y_true, p)
    cm = confusion_matrix(y_true, y_pred)

    results.append({
        "Model": name,
        "Threshold": m["tau"],
        "Accuracy (%)": round(acc, 2),
        "Bal Acc (%)": round(bal_acc, 2),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Down Prec/Rec/F1": f"{prec_0:.2f} / {rec_0:.2f} / {f1_0:.4f}",
        "Up Prec/Rec/F1": f"{prec_1:.2f} / {rec_1:.2f} / {f1_1:.4f}",
        "ROC-AUC": round(auc, 4),
        "Brier": round(brier, 4),
        "TN": cm[0, 0], "FP": cm[0, 1],
        "FN": cm[1, 0], "TP": cm[1, 1],
        "Total Pred Up": int(np.sum(y_pred == 1)),
        "Total Pred Down": int(np.sum(y_pred == 0))
    })

res_df = pd.DataFrame(results)
for r in results:
    print(f"\n--- {r['Model']} (Tau={r['Threshold']}) ---")
    print(f"  Accuracy: {r['Accuracy (%)']}% | Balanced Acc: {r['Bal Acc (%)']}% | Macro F1: {r['Macro F1']} | Weighted F1: {r['Weighted F1']}")
    print(f"  ROC-AUC: {r['ROC-AUC']} | Brier Score: {r['Brier']}")
    print(f"  Down Class: {r['Down Prec/Rec/F1']} | Up Class: {r['Up Prec/Rec/F1']}")
    print(f"  Confusion Matrix: TN={r['TN']}, FP={r['FP']}, FN={r['FN']}, TP={r['TP']}")
    print(f"  Predicted Distribution: Down={r['Total Pred Down']} ({r['Total Pred Down']/n_total*100:.1f}%), Up={r['Total Pred Up']} ({r['Total Pred Up']/n_total*100:.1f}%)")

print("\n" + "=" * 90)
print("3. INVESTIGATION: WHY MARKET-ONLY AND GMU ZERO-NEWS MACRO F1 ARE IDENTICAL")
print("=" * 90)
diff_preds = np.sum(df["pred_bilstm"].values != (df["prob_gmu_ablation"].values >= 0.52).astype(int))
print(f"Number of prediction differences between BiLSTM and GMU Ablation: {diff_preds} / {n_total}")
print(f"BiLSTM predictions: Down={np.sum(df['pred_bilstm'].values == 0)}, Up={np.sum(df['pred_bilstm'].values == 1)}")
gmu_ablation_pred = (df["prob_gmu_ablation"].values >= 0.52).astype(int)
print(f"GMU Ablation predictions: Down={np.sum(gmu_ablation_pred == 0)}, Up={np.sum(gmu_ablation_pred == 1)}")

print("\n" + "=" * 90)
print("4. INVESTIGATION: WHY GMU FUSION UNDERPERFORMS PERSISTENCE BASELINE")
print("=" * 90)
print("Analyzing the test period (June 2022 to September 2022 bear market regime):")
print(f"Persistence predicted: Down={np.sum(models['Direction Persistence']['pred'] == 0)}, Up={np.sum(models['Direction Persistence']['pred'] == 1)}")
print(f"GMU Fusion predicted:  Down={np.sum(models['GMU Fusion']['pred'] == 0)}, Up={np.sum(models['GMU Fusion']['pred'] == 1)}")
