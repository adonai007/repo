"""Expected-goals (lambda) computation from strength indices.

Faithful port of the original ``modelo_bra_mar.py`` formula::

    lambda_home = base * atk_home * adj_home_atk * def_away * adj_away_def * ha_home
    lambda_away = base * atk_away * adj_away_atk * def_home * adj_home_def * ha_away

i.e. each side's scoring rate = baseline x its attack x the opponent's defence,
modulated by context adjustments and home advantage. The ``def`` index is
"goals conceded relative to average" (<1 = elite defence), so a strong opponent
defence *lowers* your lambda.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from ..schema import MatchParams


def expected_goals(p: MatchParams) -> tuple[float, float]:
    """Return ``(lambda_home, lambda_away)`` for a parsed match.

    If ``expected_goals_override`` is present it bypasses the strength formula
    (the SOTA "supremacy/total" parameterisation).
    """
    if p.expected_goals_override is not None:
        return (
            float(p.expected_goals_override["home"]),
            float(p.expected_goals_override["away"]),
        )

    lam_home = (
        p.base_rate
        * p.strength_home.atk * p.adj_home_atk
        * p.strength_away.deff * p.adj_away_def
        * p.ha_home
    )
    lam_away = (
        p.base_rate
        * p.strength_away.atk * p.adj_away_atk
        * p.strength_home.deff * p.adj_home_def
        * p.ha_away
    )
    return lam_home, lam_away


def poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    """PMF of a Poisson(lam) truncated to ``0..max_goals`` (not renormalised)."""
    k = np.arange(max_goals + 1)
    return poisson.pmf(k, lam)
