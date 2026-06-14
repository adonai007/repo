"""Tournament Monte Carlo: valid probabilities, monotonic rounds, stronger wins more."""

from itertools import combinations

import pandas as pd
import pytest

from footy.ratings.prior_fit import PriorRatings
from footy.simulate.tournament import simulate_tournament, _seed_positions, _standings_one


def _prior():
    # Group 1 has a clear powerhouse (A); group 2 a clear powerhouse (E).
    teams = {t: {"atk": atk, "def": dfn, "n_matches": 80}
             for t, atk, dfn in [
                 ("A", 1.8, 0.5), ("B", 1.0, 1.0), ("C", 0.95, 1.05), ("D", 0.8, 1.2),
                 ("E", 1.7, 0.55), ("F", 1.0, 1.0), ("G", 0.95, 1.05), ("H", 0.8, 1.2),
             ]}
    return PriorRatings(asof="2026-06-01", base_rate=1.2, home_advantage=1.2,
                        rho=-0.05, teams=teams)


def _fixture():
    groups = {"Group 1": ["A", "B", "C", "D"], "Group 2": ["E", "F", "G", "H"]}
    rows = []
    for label, ts in groups.items():
        for h, a in combinations(ts, 2):
            rows.append({"stage": "group", "group": label, "home": h, "away": a,
                         "teams_known": True, "played": False,
                         "home_score": None, "away_score": None})
    return pd.DataFrame(rows)


def test_seed_positions_keeps_top_seeds_apart():
    pos = _seed_positions(8)
    assert sorted(pos) == list(range(1, 9))
    # seeds 1 and 2 land in opposite halves so they can only meet in the final
    assert pos.index(1) < 4 <= pos.index(2)


def test_standings_orders_by_points_then_gd():
    # A beats everyone; B and C both beat D; B beats C -> order A, B, C, D
    res = [("A", "B", 1, 0), ("A", "C", 1, 0), ("A", "D", 2, 0),
           ("B", "C", 1, 0), ("B", "D", 1, 0), ("C", "D", 1, 0)]
    assert _standings_one(["A", "B", "C", "D"], res) == ["A", "B", "C", "D"]


def test_tournament_probs_valid_and_monotonic():
    res = simulate_tournament(_prior(), _fixture(), n_sims=3000, seed=7)
    teams = res["teams"]
    champ_total = sum(p["champion"] for p in teams.values())
    assert champ_total == pytest.approx(1.0, abs=1e-9)
    for t, p in teams.items():
        # reaching a later round is never more likely than an earlier one
        assert p["champion"] <= p["final"] <= p["sf"] <= p["advance"] + 1e-9
        assert 0.0 <= p["champion"] <= 1.0
    # the two powerhouses should be the title favorites
    fav = [f["team"] for f in res["favorites"][:2]]
    assert set(fav) == {"A", "E"}
