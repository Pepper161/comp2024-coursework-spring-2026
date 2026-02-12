"""Metric helpers for IDS binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected_features: int,
    total_features: int,
    runtime_sec: float,
    fit_time_sec: float,
    predict_time_sec: float,
) -> dict[str, Any]:
    """Compute metrics required by coursework rubric and optimization protocol."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    denom = fp + tn
    fpr = float(fp / denom) if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr,
        "selected_features": int(selected_features),
        "selected_feature_ratio": float(selected_features / max(total_features, 1)),
        "runtime_sec": float(runtime_sec),
        "fit_time_sec": float(fit_time_sec),
        "predict_time_sec": float(predict_time_sec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

