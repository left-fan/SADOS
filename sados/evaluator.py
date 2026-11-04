"""Evaluation helpers."""

from __future__ import annotations

import numpy as np

from .metrics import binary_classification_metrics


def evaluate_scores(y_true: np.ndarray, anomaly_score: np.ndarray) -> dict[str, float]:
    return binary_classification_metrics(y_true, anomaly_score)
