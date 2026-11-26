"""Detection metrics for the faithfulness gate vs. human labels.

detection P/R/F1, Cohen's kappa, determinism (variance across repeated runs),
cost (API $ / latency). All pure functions; no LLM calls.
"""
from __future__ import annotations
import numpy as np


def prf(y_true: list[int], y_pred: list[int]) -> dict:
    """Binary detection: 1 = hallucinated / not-grounded (positive class)."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / len(yt) if len(yt) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(yt)}


def cohen_kappa(y_true: list[int], y_pred: list[int]) -> float:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = len(yt)
    if n == 0:
        return 0.0
    po = float((yt == yp).mean())
    # expected agreement
    pe = 0.0
    for c in (0, 1):
        pe += (yt == c).mean() * (yp == c).mean()
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def determinism_variance(f1_runs: list[float]) -> float:
    """Variance of detection F1 across repeated identical runs.
    Deterministic scorers -> 0. LLM-judge -> > 0."""
    if not f1_runs:
        return float("nan")
    return float(np.var(np.asarray(f1_runs, dtype=float)))
