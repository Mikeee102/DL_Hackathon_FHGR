"""Metriken für binäre Klassifikation bei Imbalance.

Primärmetriken: PR-AUC (Average Precision) und F1. Accuracy nur ergänzend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    threshold: float
    confusion: list[list[int]]


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return BinaryMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        roc_auc=float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        threshold=float(threshold),
        confusion=cm,
    )


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Maximiert F1 entlang der PR-Kurve. Gibt (threshold, f1) zurück."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve liefert n+1 Werte vs. n Thresholds — letzten ignorieren.
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])
