"""
Evaluation metrics and validation-only threshold calibration.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def calibrate_threshold_on_val(y_val: np.ndarray, val_probs: np.ndarray) -> float:
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    d_count = int(np.sum(y_pred == 0))
    u_count = int(np.sum(y_pred == 1))
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "down_preds": d_count,
        "up_preds": u_count
    }
