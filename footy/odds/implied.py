"""Convert bookmaker odds to de-margined probabilities and blend with the model.

Closing odds are the hardest-to-beat benchmark in football forecasting, so the
framework always reports model / market / blend rather than trusting one blindly.
Odds may be given as American (signed ints, e.g. -175, +300) or decimal (>1.0).
"""

from __future__ import annotations

import numpy as np


def american_to_prob(odds: float) -> float:
    """Raw (with-margin) implied probability from American odds."""
    odds = float(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def decimal_to_prob(odds: float) -> float:
    """Raw (with-margin) implied probability from decimal odds."""
    return 1.0 / float(odds)


def _raw_prob(value: float) -> float:
    """Heuristic: decimal odds are > 1; American are <= -100 or >= +100."""
    v = float(value)
    if v >= 1.0 and v <= 100.0:
        # ambiguous small positives are treated as decimal (1.01..100)
        return decimal_to_prob(v)
    return american_to_prob(v)


# map common odds-key spellings -> canonical home/draw/away
_KEY_ALIASES = {
    "home": "home", "home_win": "home", "home_win_decimal": "home",
    "home_decimal": "home", "1": "home",
    "draw": "draw", "draw_decimal": "draw", "x": "draw", "tie": "draw",
    "away": "away", "away_win": "away", "away_win_decimal": "away",
    "away_decimal": "away", "2": "away",
}


def _canonical_odds(odds: dict) -> dict:
    """Map a market_odds dict (various key spellings) to {home, draw, away}."""
    out = {}
    for k, v in odds.items():
        canon = _KEY_ALIASES.get(str(k).strip().lower())
        if canon and canon not in out:
            out[canon] = v
    missing = {"home", "draw", "away"} - set(out)
    if missing:
        raise KeyError(f"market_odds missing outcomes: {missing}")
    return out


def implied_probabilities(odds: dict) -> dict:
    """De-margined 1X2 probabilities via the multiplicative (normalisation) method.

    ``odds`` keys may be ``home``/``draw``/``away`` or common variants
    (``home_win_decimal``, ``1``/``X``/``2`` ...). Returns probabilities that sum
    to 1, plus the implied ``overround``.
    """
    odds = _canonical_odds(odds)
    raw = {k: _raw_prob(odds[k]) for k in ("home", "draw", "away")}
    overround = sum(raw.values())
    probs = {k: v / overround for k, v in raw.items()}
    probs["overround"] = overround
    return probs


def shin_probabilities(odds: dict, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """De-margined probabilities via Shin's method (accounts for insider trading).

    Solves for z (proportion of informed money) so probabilities sum to 1.
    Falls back to the multiplicative method if it fails to converge.
    """
    odds = _canonical_odds(odds)
    raw = np.array([_raw_prob(odds[k]) for k in ("home", "draw", "away")], dtype=float)
    booksum = raw.sum()
    pi = raw / booksum  # initial guess

    z = 0.0
    for _ in range(max_iter):
        denom = booksum  # normalisation
        sqrt_term = np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / denom)
        p = (sqrt_term - z) / (2.0 * (1.0 - z)) if z < 1.0 else pi
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return implied_probabilities(odds)
        new_z = z + (s - 1.0) * 0.5
        new_z = float(np.clip(new_z, 0.0, 0.2))
        if abs(new_z - z) < tol:
            z = new_z
            break
        z = new_z

    p = p / p.sum()
    return {
        "home": float(p[0]),
        "draw": float(p[1]),
        "away": float(p[2]),
        "overround": float(booksum),
        "z": float(z),
    }


def blend_probabilities(
    model: tuple[float, float, float],
    market: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    """Convex blend ``weight`` toward the market (0 = pure model, 1 = pure market)."""
    w = float(np.clip(weight, 0.0, 1.0))
    blended = [(1 - w) * m + w * k for m, k in zip(model, market)]
    s = sum(blended)
    return tuple(b / s for b in blended)
