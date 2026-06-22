"""Knockout advancement: 90' -> extra time -> penalty shootout.

World Cup knockout rules: a draw after 90' goes to 30' of extra time (not sudden
death), then a best-of-five penalty shootout. We model::

    P(home advances) = P(home win 90')
                     + P(draw 90') * [ P(home win ET)
                                       + P(draw ET) * P(home win shootout) ]

Extra time reuses the Dixon-Coles matrix with lambdas scaled by 30/90.
The shootout is a single Bernoulli driven by relative ``penalty_strength``.
"""

from __future__ import annotations

import numpy as np

from .dixon_coles import score_matrix


def _windrawloss(M: np.ndarray) -> tuple[float, float, float]:
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())
    return p_home, p_draw, p_away


def shootout_home_win_prob(penalty_strength: dict | None) -> float:
    """P(home wins the shootout) from relative penalty strength (default 0.5)."""
    if not penalty_strength:
        return 0.5
    h = float(penalty_strength.get("home", 0.5))
    a = float(penalty_strength.get("away", 0.5))
    if h + a <= 0:
        return 0.5
    return h / (h + a)


def advancement_probability(
    lam_home: float,
    lam_away: float,
    rho: float,
    penalty_strength: dict | None = None,
    max_goals: int = 10,
    et_fraction: float = 30.0 / 90.0,
) -> dict:
    """Return advancement probabilities and the stage-by-stage breakdown."""
    M90 = score_matrix(lam_home, lam_away, rho, max_goals)
    h90, d90, a90 = _windrawloss(M90)

    Met = score_matrix(lam_home * et_fraction, lam_away * et_fraction, rho, max_goals)
    het, det, aet = _windrawloss(Met)

    p_home_pens = shootout_home_win_prob(penalty_strength)

    p_home_adv = h90 + d90 * (het + det * p_home_pens)
    p_away_adv = 1.0 - p_home_adv

    return {
        "p_home_advance": p_home_adv,
        "p_away_advance": p_away_adv,
        "regulation": {"home": h90, "draw": d90, "away": a90},
        "extra_time": {"home": het, "draw": det, "away": aet},
        "shootout_home_win": p_home_pens,
    }
