"""Golden test: the engine must reproduce the original modelo_bra_mar.py output.

These numbers (lambdas ~1.15/0.98; 1X2 ~38.3/32.2/29.6; 1-1 ~14.7%, 0-0 ~13.1%,
1-0 ~12.3%) are the validated result the user already trusts. Any drift here is a
regression in the numeric core.
"""

import json
from pathlib import Path

import pytest

from footy.schema import MatchParams
from footy.predict import predict

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bra_mar" / "params.json"


@pytest.fixture(scope="module")
def result():
    params = MatchParams.parse(json.loads(EXAMPLE.read_text()))
    return predict(params, n_sims=200_000, seed=42)


def test_lambdas(result):
    assert result["lambda"]["home"] == pytest.approx(1.154, abs=0.005)
    assert result["lambda"]["away"] == pytest.approx(0.984, abs=0.005)


def test_1x2(result):
    a = result["analytic"]
    assert a["p_home"] == pytest.approx(0.383, abs=0.005)
    assert a["p_draw"] == pytest.approx(0.322, abs=0.005)
    assert a["p_away"] == pytest.approx(0.296, abs=0.005)
    assert a["p_home"] + a["p_draw"] + a["p_away"] == pytest.approx(1.0, abs=1e-9)


def test_top_scorelines(result):
    top = {s["score"]: s["p"] for s in result["analytic"]["top_scores"]}
    assert top["1-1"] == pytest.approx(0.147, abs=0.004)
    assert top["0-0"] == pytest.approx(0.131, abs=0.004)
    assert top["1-0"] == pytest.approx(0.123, abs=0.004)
    # 1-1 is the single most likely scoreline
    assert result["analytic"]["top_scores"][0]["score"] == "1-1"


def test_derived_markets(result):
    a = result["analytic"]
    assert a["p_under25"] == pytest.approx(0.639, abs=0.01)
    assert a["p_btts"] == pytest.approx(0.442, abs=0.01)


def test_monte_carlo_agrees_with_analytic(result):
    a, mc = result["analytic"], result["monte_carlo"]
    assert mc["p_home"] == pytest.approx(a["p_home"], abs=0.01)
    assert mc["p_draw"] == pytest.approx(a["p_draw"], abs=0.01)
    assert mc["p_away"] == pytest.approx(a["p_away"], abs=0.01)


def test_market_comparison_present(result):
    mk = result["market"]
    # de-margined market: Brazil clear favourite (~59%)
    assert mk["implied"]["home"] == pytest.approx(0.59, abs=0.03)
    assert sum(mk["implied"].values()) == pytest.approx(1.0, abs=1e-9)
