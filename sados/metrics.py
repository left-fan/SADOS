"""Evaluation metrics for binary graph anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "auroc": safe_roc_auc(y_true, y_score),
        "auprc": safe_average_precision(y_true, y_score),
        "f1": safe_f1(y_true, y_pred),
    }


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size < 2 and np.unique(y_pred).size < 2:
        return float("nan")
    return float(f1_score(y_true, y_pred, zero_division=0))
