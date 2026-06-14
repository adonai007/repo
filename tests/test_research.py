"""Research baseline + engine tests (offline, no LLM / network)."""

import json

import pytest

from footy.ratings.prior_fit import PriorRatings
from footy.research.calibrate import baseline_params
from footy.research.engine import research_match, _extract, _clip_to_prior
from footy.schema import MatchParams
from footy.predict import predict


def _prior():
    return PriorRatings(
        asof="2026-06-01", base_rate=1.1, home_advantage=1.25, rho=-0.06,
        teams={
            "Brazil": {"atk": 1.5, "def": 0.7, "n_matches": 90},
            "Morocco": {"atk": 1.1, "def": 0.55, "n_matches": 100},
            "Mexico": {"atk": 1.2, "def": 0.8, "n_matches": 120},
        },
    )


def test_baseline_params_valid_and_predicts():
    p = baseline_params({"home": "Brazil", "away": "Morocco", "date": "2026-06-13"}, _prior())
    assert p["match"]["neutral"] is True
    mp = MatchParams.parse(p)
    a = predict(mp, run_mc=False)["analytic"]
    assert a["p_home"] + a["p_draw"] + a["p_away"] == pytest.approx(1.0, abs=1e-9)


def test_host_gets_home_advantage():
    p = baseline_params({"home": "Mexico", "away": "Morocco"}, _prior())
    assert p["match"]["neutral"] is False
    assert p["home_advantage"]["home"] > 1.0


def test_research_falls_back_to_baseline_without_runner():
    res = research_match({"home": "Brazil", "away": "Morocco"}, _prior(), runner=None)
    assert res["engine"] == "prior-baseline"
    MatchParams.parse(res["params"])  # valid


def test_research_uses_runner_output_and_clips():
    prior = _prior()
    base = baseline_params({"home": "Brazil", "away": "Morocco"}, prior)

    def fake_runner(prompt):
        params = json.loads(json.dumps(base))
        # LLM tries to push Brazil attack way up (should be clipped to +40%)
        params["strength"]["home"]["atk"] = 99.0
        params["citations"] = ["http://example.com"]
        return "## Dossier\nstuff\n```json\n" + json.dumps(params) + "\n```"

    res = research_match({"home": "Brazil", "away": "Morocco"}, prior, runner=fake_runner)
    assert res["engine"].startswith("claude")
    # clipped to prior * 1.4
    assert res["params"]["strength"]["home"]["atk"] == pytest.approx(1.5 * 1.4, abs=1e-6)


def test_extract_json_block():
    params, dossier = _extract("text before\n```json\n{\"a\": 1}\n```\n")
    assert params == {"a": 1}
    assert "text before" in dossier
