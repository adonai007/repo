"""Property tests for the numeric core (independent of the Bra-Mar golden values)."""

import numpy as np
import pytest

from footy.model.dixon_coles import score_matrix, tau
from footy.model.markets import outcomes_from_matrix
from footy.model.knockout import advancement_probability
from footy.odds.implied import (
    american_to_prob,
    implied_probabilities,
    shin_probabilities,
    blend_probabilities,
)


def test_matrix_sums_to_one():
    M = score_matrix(1.6, 1.1, -0.08, max_goals=12)
    assert M.sum() == pytest.approx(1.0, abs=1e-12)


def test_rho_zero_is_independent_poisson():
    lh, la = 1.4, 1.0
    M = score_matrix(lh, la, 0.0, max_goals=15)
    from scipy.stats import poisson
    k = np.arange(16)
    expected = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    expected /= expected.sum()
    assert np.allclose(M, expected, atol=1e-12)


def test_tau_increases_draw_mass_for_negative_rho():
    lh, la = 1.2, 1.0
    draws0 = np.trace(score_matrix(lh, la, 0.0))
    draws_neg = np.trace(score_matrix(lh, la, -0.1))
    assert draws_neg > draws0  # negative rho inflates 0-0 and 1-1


def test_tau_cell_values():
    assert tau(0, 0, 1.5, 1.0, -0.1) == pytest.approx(1 - 1.5 * 1.0 * -0.1)
    assert tau(1, 1, 1.5, 1.0, -0.1) == pytest.approx(1 - -0.1)
    assert tau(2, 3, 1.5, 1.0, -0.1) == 1.0  # untouched outside the four cells


def test_markets_are_coherent():
    M = score_matrix(1.5, 1.2, -0.05)
    o = outcomes_from_matrix(M)
    assert o.p_home + o.p_draw + o.p_away == pytest.approx(1.0, abs=1e-9)
    assert o.p_over25 + o.p_under25 == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= o.p_btts <= 1.0
    assert o.exp_total == pytest.approx(o.exp_goals_home + o.exp_goals_away)


def test_knockout_advancement_sums_to_one():
    ko = advancement_probability(1.4, 1.1, -0.07, {"home": 0.55, "away": 0.5})
    assert ko["p_home_advance"] + ko["p_away_advance"] == pytest.approx(1.0, abs=1e-9)
    # advancing is always at least as likely as winning in regulation
    assert ko["p_home_advance"] >= ko["regulation"]["home"]


def test_american_to_prob():
    assert american_to_prob(-175) == pytest.approx(175 / 275)
    assert american_to_prob(300) == pytest.approx(100 / 400)


def test_implied_probabilities_demargined():
    p = implied_probabilities({"home": -175, "draw": 300, "away": 425})
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-12)
    assert p["overround"] > 1.0
    assert p["home"] == pytest.approx(0.59, abs=0.02)


def test_shin_sums_to_one():
    p = shin_probabilities({"home": -175, "draw": 300, "away": 425})
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-9)


def test_blend():
    model = (0.38, 0.32, 0.30)
    market = (0.59, 0.26, 0.15)
    b = blend_probabilities(model, market, 0.5)
    assert sum(b) == pytest.approx(1.0, abs=1e-12)
    assert model[0] < b[0] < market[0]  # blended toward market on home
