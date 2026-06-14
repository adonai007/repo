"""Validation / guard tests for the params.json contract."""

import pytest

from footy.schema import MatchParams, ParamsError

MINIMAL = {
    "match": {"home": "A", "away": "B"},
    "base_rate": 1.3,
    "strength": {"home": {"atk": 1.2, "def": 0.9}, "away": {"atk": 1.0, "def": 1.0}},
    "rho": -0.1,
    "adjustments": {},
    "home_advantage": {},
}


def test_minimal_parses_with_defaults():
    p = MatchParams.parse(MINIMAL)
    assert p.adj_home_atk == 1.0 and p.ha_home == 1.0
    assert p.match.is_knockout is False
    assert p.blend_weight == 0.0


def test_missing_strength_rejected():
    bad = {**MINIMAL, "strength": {"home": {"atk": 1.2, "def": 0.9}}}
    with pytest.raises(ParamsError):
        MatchParams.parse(bad)


def test_out_of_range_rho_rejected():
    bad = {**MINIMAL, "rho": -0.9}
    with pytest.raises(ParamsError):
        MatchParams.parse(bad)


def test_out_of_range_strength_rejected():
    bad = {**MINIMAL, "strength": {"home": {"atk": 99, "def": 0.9}, "away": {"atk": 1.0, "def": 1.0}}}
    with pytest.raises(ParamsError):
        MatchParams.parse(bad)


def test_knockout_stage_flag():
    p = MatchParams.parse({**MINIMAL, "match": {"home": "A", "away": "B", "stage": "final"}})
    assert p.match.is_knockout is True
