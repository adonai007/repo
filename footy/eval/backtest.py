"""Backtest the model on a past tournament with data-as-of (no look-ahead).

Fits the prior as-of the day before kickoff, predicts every match, grades against
actual results, and compares mean RPS to trivial baselines (uniform 1/3/1/3, and
the historical home/draw/away base rate). This is the honesty check: a model that
can't beat the base-rate baseline has no skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ratings.prior_fit import fit_prior, load_results
from ..research.calibrate import baseline_params
from ..schema import MatchParams
from ..predict import predict
from .metrics import summarize, _ORDER


def _result(hs: float, as_: float) -> str:
    return "home" if hs > as_ else "away" if as_ > hs else "draw"


def backtest_tournament(
    tournament: str = "FIFA World Cup",
    year: int = 2022,
    window_years: int = 8,
    max_matches: int | None = None,
) -> dict:
    """Backtest one tournament edition. Returns metrics vs baselines."""
    allres = load_results()
    mask = (allres["tournament"].astype(str).str.contains(tournament, case=False)
            & (allres["date"].dt.year == year)
            & allres["home_score"].notna())
    matches = allres[mask].sort_values("date").reset_index(drop=True)
    if matches.empty:
        raise ValueError(f"no matches found for {tournament} {year}")
    if max_matches:
        matches = matches.head(max_matches)

    start = matches["date"].min()
    prior = fit_prior(asof=start - pd.Timedelta(days=1), window_years=window_years)

    forecasts, skipped = [], 0
    for _, m in matches.iterrows():
        match = {"home": m["home_team"], "away": m["away_team"],
                 "date": str(m["date"].date()),
                 "stage": "group", "neutral": bool(m["neutral"])}
        try:
            params = baseline_params(match, prior)
            out = predict(MatchParams.parse(params), run_mc=False)
            a = out["analytic"]
            forecasts.append(((a["p_home"], a["p_draw"], a["p_away"]),
                              _result(m["home_score"], m["away_score"])))
        except (KeyError, Exception):
            skipped += 1

    model = summarize(forecasts)

    # baselines
    outcomes = [o for _, o in forecasts]
    base_rate = np.array([outcomes.count(k) for k in _ORDER], dtype=float)
    base_rate = base_rate / base_rate.sum() if base_rate.sum() else np.array([1, 1, 1]) / 3
    uniform = summarize([((1 / 3, 1 / 3, 1 / 3), o) for o in outcomes])
    baserate = summarize([(tuple(base_rate), o) for o in outcomes])

    return {
        "tournament": f"{tournament} {year}", "n": model["n"], "skipped": skipped,
        "prior_asof": prior.asof,
        "model": model,
        "baseline_uniform": uniform,
        "baseline_baserate": baserate,
        "beats_uniform": model.get("rps", 1) < uniform.get("rps", 1),
        "beats_baserate": model.get("rps", 1) < baserate.get("rps", 1),
    }
