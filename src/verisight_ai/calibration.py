"""Temperature scaling and threshold / abstention policy. Fitted on validation data only."""

from __future__ import annotations

import numpy as np

from .config import SYNTHETIC_INDEX
from .schemas import DecisionStatus


def softmax_p_synthetic(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, SYNTHETIC_INDEX] / e.sum(axis=1)).astype(np.float64)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax_p_synthetic(logits, temperature)
    p = np.clip(np.where(labels == SYNTHETIC_INDEX, p, 1 - p), 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Fit a single temperature by minimising validation NLL; accept only if NLL improves."""
    from scipy.optimize import minimize_scalar  # train extra; keeps the serving image lean

    res = minimize_scalar(
        lambda log_t: _nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    t = float(np.exp(res.x))
    before, after = _nll(logits, labels, 1.0), _nll(logits, labels, t)
    return {
        "method": "temperature_scaling",
        "temperature": t,
        "val_nll_before": before,
        "val_nll_after": after,
        "applied": bool(np.isfinite(t) and after < before - 1e-6),
    }


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Pick the threshold maximising validation F1 and report the precision/recall trade-off."""
    y = (labels == SYNTHETIC_INDEX).astype(int)
    candidates = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 91), scores]), 6))
    best = {"threshold": 0.5, "f1": -1.0}
    curve = []
    for t in candidates:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        curve.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1})
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1}
    best["tradeoff_curve_points"] = len(curve)
    best["criterion"] = "max validation F1 (no assumed preference between FP and FN)"
    return best


def decide(score: float, threshold: float, margin: float) -> DecisionStatus:
    """Three-way decision with an abstain band around the threshold."""
    if score >= threshold + margin:
        return DecisionStatus.SYNTHETIC_LIKE
    if score <= threshold - margin:
        return DecisionStatus.NO_STRONG_SIGNAL
    return DecisionStatus.INCONCLUSIVE
