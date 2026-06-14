"""End-to-end prediction for a single match from validated :class:`MatchParams`.

Pipeline: params -> lambdas -> Dixon-Coles matrix -> analytic markets ->
Monte Carlo validation -> market comparison (de-margined) + blend ->
knockout advancement (if applicable).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import MatchParams
from .model.poisson import expected_goals
from .model.dixon_coles import score_matrix
from .model.markets import outcomes_from_matrix
from .model.knockout import advancement_probability
from .simulate.monte_carlo import simulate_from_matrix
from .odds.implied import implied_probabilities, blend_probabilities


def predict(
    params: MatchParams,
    max_goals: int = 10,
    n_sims: int = 200_000,
    seed: int = 42,
    run_mc: bool = True,
) -> dict[str, Any]:
    """Run the full numeric pipeline and return a JSON-serialisable result."""
    lam_home, lam_away = expected_goals(params)
    M = score_matrix(lam_home, lam_away, params.rho, max_goals)
    outcome = outcomes_from_matrix(M)

    result: dict[str, Any] = {
        "match": {
            "home": params.match.home,
            "away": params.match.away,
            "date": params.match.date,
            "stage": params.match.stage,
            "neutral": params.match.neutral,
        },
        "lambda": {"home": lam_home, "away": lam_away, "total": lam_home + lam_away},
        "rho": params.rho,
        "analytic": outcome.to_dict(),
        "score_matrix": M.tolist(),
    }

    if run_mc:
        result["monte_carlo"] = simulate_from_matrix(M, n_sims=n_sims, seed=seed)

    # Market comparison + blend
    if params.market_odds:
        try:
            implied = implied_probabilities(params.market_odds)
            market = (implied["home"], implied["draw"], implied["away"])
            blended = blend_probabilities(outcome.as_1x2(), market, params.blend_weight)
            result["market"] = {
                "implied": {"home": market[0], "draw": market[1], "away": market[2]},
                "overround": implied["overround"],
                "blend_weight": params.blend_weight,
                "blended": {"home": blended[0], "draw": blended[1], "away": blended[2]},
            }
        except (KeyError, ZeroDivisionError, ValueError):
            result["market"] = {"error": "could not parse market_odds"}

    # Knockout advancement
    if params.match.is_knockout:
        result["knockout"] = advancement_probability(
            lam_home, lam_away, params.rho, params.penalty_strength, max_goals
        )

    # Ensemble across scenarios (uncertainty band on 1X2)
    if params.scenarios:
        result["ensemble"] = _scenario_ensemble(params, max_goals)

    return result


def _scenario_ensemble(params: MatchParams, max_goals: int) -> dict[str, Any]:
    """Run each named scenario and report the spread of 1X2 probabilities."""
    rows = []
    for sc in params.scenarios:
        st = sc.get("strength", {})
        try:
            lh = (
                params.base_rate
                * float(st["home"]["atk"]) * params.adj_home_atk
                * float(st["away"]["def"]) * params.adj_away_def
                * params.ha_home
            )
            la = (
                params.base_rate
                * float(st["away"]["atk"]) * params.adj_away_atk
                * float(st["home"]["def"]) * params.adj_home_def
                * params.ha_away
            )
            rho = float(sc.get("rho", params.rho))
            M = score_matrix(lh, la, rho, max_goals)
            o = outcomes_from_matrix(M)
            rows.append({"name": sc.get("name", "scenario"), **dict(zip(("home", "draw", "away"), o.as_1x2()))})
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return {}
    arr = np.array([[r["home"], r["draw"], r["away"]] for r in rows])
    return {
        "scenarios": rows,
        "range": {
            "home": [float(arr[:, 0].min()), float(arr[:, 0].max())],
            "draw": [float(arr[:, 1].min()), float(arr[:, 1].max())],
            "away": [float(arr[:, 2].min()), float(arr[:, 2].max())],
        },
    }
