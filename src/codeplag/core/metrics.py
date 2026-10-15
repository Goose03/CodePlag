"""Binary classification metrics shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute threshold-dependent and ranking metrics.

    Args:
        labels: Ground-truth labels (0/1).
        scores: Predicted scores in ``[0, 1]``; higher means more likely positive.
        threshold: A pair is predicted positive when ``score >= threshold``.

    Returns:
        Accuracy, precision, recall, F1, FPR, ROC-AUC, PR-AUC and confusion counts. AUCs are
        ``nan`` when only one class is present.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = len(np.unique(y)) == 2
    return {
        "n": float(len(y)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def threshold_for_fpr(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, max_fpr: float
) -> float:
    """Smallest threshold whose false-positive rate does not exceed ``max_fpr``.

    Choosing the smallest admissible threshold maximizes recall under the FPR budget.

    Args:
        labels: Ground-truth labels.
        scores: Scores.
        max_fpr: False-positive-rate budget in ``[0, 1]``.

    Returns:
        The threshold (``score >= threshold`` means positive).
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    fpr, _, thresholds = roc_curve(y, s)
    admissible = thresholds[fpr <= max_fpr]
    admissible = admissible[np.isfinite(admissible)]
    if admissible.size == 0:
        return 1.0
    return float(min(1.0, admissible.min()))


def threshold_for_f1(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> float:
    """Threshold that maximizes F1 on the given data."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(s, 4)):
        f1 = binary_metrics(y, s, float(t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t
