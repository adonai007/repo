"""Forecast-quality metrics for 1X2 predictions.

RPS (Ranked Probability Score) is the standard for football because the outcome
is ordinal (home / draw / away): predicting a draw when the home team wins is
"less wrong" than predicting an away win. Lower is better for all three metrics.
"""

from __future__ import annotations

import numpy as np

_ORDER = ("home", "draw", "away")


def _onehot(outcome: str) -> np.ndarray:
    v = np.zeros(3)
    v[_ORDER.index(outcome)] = 1.0
    return v


def rps_1x2(probs, outcome: str) -> float:
    """Ranked Probability Score for one ordinal 1X2 forecast (0 = perfect)."""
    p = np.asarray(probs, dtype=float)
    o = _onehot(outcome)
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(o)
    return float(np.sum((cum_p - cum_o) ** 2) / (len(p) - 1))


def log_loss_1x2(probs, outcome: str, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    return float(-np.log(p[_ORDER.index(outcome)]))


def brier_1x2(probs, outcome: str) -> float:
    p = np.asarray(probs, dtype=float)
    return float(np.sum((p - _onehot(outcome)) ** 2))


def summarize(forecasts: list[tuple], ) -> dict:
    """Mean RPS / log-loss / Brier / accuracy over ``(probs, outcome)`` pairs."""
    if not forecasts:
        return {"n": 0}
    rps = np.mean([rps_1x2(p, o) for p, o in forecasts])
    ll = np.mean([log_loss_1x2(p, o) for p, o in forecasts])
    br = np.mean([brier_1x2(p, o) for p, o in forecasts])
    acc = np.mean([_ORDER[int(np.argmax(p))] == o for p, o in forecasts])
    return {"n": len(forecasts), "rps": float(rps), "log_loss": float(ll),
            "brier": float(br), "accuracy": float(acc)}


def reliability(forecasts: list[tuple], n_bins: int = 10) -> list[dict]:
    """Calibration curve: for each predicted-probability bin, the empirical rate.

    Pools all three outcome probabilities (one-vs-rest) into bins.
    """
    preds, hits = [], []
    for probs, outcome in forecasts:
        for k, name in enumerate(_ORDER):
            preds.append(probs[k])
            hits.append(1.0 if outcome == name else 0.0)
    preds = np.asarray(preds)
    hits = np.asarray(hits)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(n_bins):
        mask = (preds >= edges[b]) & (preds < edges[b + 1] if b < n_bins - 1 else preds <= edges[b + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin": (edges[b] + edges[b + 1]) / 2,
            "predicted": float(preds[mask].mean()),
            "empirical": float(hits[mask].mean()),
            "n": int(mask.sum()),
        })
    return out
