"""The prior fit must recover known strengths from synthetic data."""

import numpy as np
import pandas as pd
import pytest

from footy.ratings.prior_fit import fit_prior


def _synthetic(n_teams=20, n_matches=4000, seed=0):
    rng = np.random.default_rng(seed)
    true_atk = rng.normal(0, 0.4, n_teams)
    true_def = rng.normal(0, 0.4, n_teams)
    mu = np.log(1.3)
    rows = []
    base = pd.Timestamp("2026-01-01")
    for k in range(n_matches):
        i, j = rng.integers(0, n_teams, 2)
        if i == j:
            continue
        lh = np.exp(mu + true_atk[i] - true_def[j])
        la = np.exp(mu + true_atk[j] - true_def[i])
        rows.append({
            "date": base - pd.Timedelta(days=int(rng.integers(0, 700))),
            "home_team": f"T{i}", "away_team": f"T{j}",
            "home_score": rng.poisson(lh), "away_score": rng.poisson(la),
            "tournament": "Friendly", "neutral": True,
        })
    return pd.DataFrame(rows), true_atk, true_def


def test_prior_recovers_synthetic_ratings():
    df, true_atk, true_def = _synthetic()
    prior = fit_prior(asof="2026-01-02", window_years=8, xi=0.0, reg=0.01, df=df)

    fitted_atk = np.array([np.log(prior.teams[f"T{i}"]["atk"]) for i in range(20)])
    fitted_def = np.array([-np.log(prior.teams[f"T{i}"]["def"]) for i in range(20)])

    # centring is arbitrary; compare against centred truth via correlation
    r_atk = np.corrcoef(fitted_atk, true_atk - true_atk.mean())[0, 1]
    r_def = np.corrcoef(fitted_def, true_def - true_def.mean())[0, 1]
    assert r_atk > 0.9
    assert r_def > 0.9


def test_prior_global_params_sane():
    df, _, _ = _synthetic()
    prior = fit_prior(asof="2026-01-02", window_years=8, xi=0.0, reg=0.01, df=df)
    assert 0.8 < prior.base_rate < 2.0
    assert -0.2 <= prior.rho <= 0.0
