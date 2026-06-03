"""
Evaluation utilities — accuracy, precision/recall/F1, ROC-AUC, FAR/FRR, EER,
and a paired-bootstrapped 95% CI. Designed for the model zoo + security tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class EvalReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    far: float         # false acceptance rate (spoof accepted as real)
    frr: float         # false rejection rate  (real rejected as spoof)
    eer: float         # equal error rate
    threshold: float   # decision threshold that gives the EER


# -----------------------------------------------------------------------------
# Binary classification
# -----------------------------------------------------------------------------
def evaluate_binary(
    y_true: NDArray[np.int32],
    scores: NDArray[np.float32],
    *,
    pos_label: int = 1,
) -> EvalReport:
    """scores: higher = more likely `pos_label`."""
    y = (np.asarray(y_true) == pos_label).astype(np.int32)
    auc = float(roc_auc_score(y, scores)) if y.sum() and (len(y) - y.sum()) else 0.0
    # FAR/FRR sweep
    fpr, tpr, thresh = roc_curve(y, scores, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    thr_eer = float(thresh[i]) if i < len(thresh) else 0.5
    yhat = (scores >= thr_eer).astype(np.int32)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)
    return EvalReport(
        accuracy=float(accuracy_score(y, yhat)),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        f1=float(f1_score(y, yhat, zero_division=0)),
        auc=auc,
        far=float(far),
        frr=float(frr),
        eer=eer,
        threshold=thr_eer,
    )


# -----------------------------------------------------------------------------
# Face-recognition evaluation: TAR@FAR
# -----------------------------------------------------------------------------
def tar_at_far(
    genuine: NDArray[np.float32],
    impostor: NDArray[np.float32],
    far_target: float = 1e-3,
) -> tuple[float, float]:
    """TAR@FAR — true accept rate at a fixed false-accept rate."""
    far_target = max(1e-5, min(0.5, float(far_target)))
    if genuine.size == 0 or impostor.size == 0:
        return 0.0, 0.0
    all_scores = np.concatenate([genuine, impostor])
    is_genuine = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    fpr, tpr, thr = roc_curve(is_genuine, all_scores, pos_label=1)
    if fpr.size == 0:
        return 0.0, 0.0
    # Pick the lowest threshold where FPR <= target
    idx = int(np.searchsorted(fpr, far_target, side="right"))
    idx = min(max(0, idx - 1), len(fpr) - 1)
    return float(tpr[idx]), float(thr[idx])


# -----------------------------------------------------------------------------
# Bootstrap CI
# -----------------------------------------------------------------------------
def bootstrap_ci(
    values: NDArray[np.float32],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric bootstrap percentile CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = float(values[idx].mean())
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
