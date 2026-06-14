"""Ledger upsert / grading / immutability tests (offline)."""

import pandas as pd
import pytest

from footy.worldcup.results_store import upsert_ledger, grade_ledger, load_ledger, match_slug


def _row(mn, home, away, ph, pd_, pa):
    return {
        "match_number": mn, "date": "2026-06-14", "stage": "group", "group": "A",
        "home": home, "away": away, "p_home": ph, "p_draw": pd_, "p_away": pa,
        "blend_home": None, "blend_draw": None, "blend_away": None,
        "mkt_home": None, "mkt_draw": None, "mkt_away": None,
        "top_score": "1-1", "lambda_home": 1.2, "lambda_away": 1.0,
        "engine": "prior-baseline", "confidence": "baseline",
        "generated_at": "2026-06-14T00:00:00Z", "code_sha": "abc",
        "actual_home": None, "actual_away": None, "actual_result": None,
        "rps": None, "graded": False,
    }


def test_slug():
    assert match_slug(9, "Côte d'Ivoire", "Ecuador").startswith("009-")


def test_upsert_and_grade(tmp_path):
    upsert_ledger([_row(1, "A", "B", 0.5, 0.3, 0.2), _row(2, "C", "D", 0.4, 0.3, 0.3)], tmp_path)
    led = load_ledger(tmp_path)
    assert len(led) == 2 and not led["graded"].any()

    fixture = pd.DataFrame([
        {"match_number": 1, "home_score": 2, "away_score": 0, "played": True},
        {"match_number": 2, "home_score": 1, "away_score": 1, "played": True},
    ])
    led = grade_ledger(fixture, tmp_path)
    assert led["graded"].all()
    r1 = led[led.match_number == 1].iloc[0]
    assert r1["actual_result"] == "home" and r1["rps"] is not None and r1["rps"] >= 0


def test_graded_rows_are_immutable(tmp_path):
    upsert_ledger([_row(1, "A", "B", 0.5, 0.3, 0.2)], tmp_path)
    fixture = pd.DataFrame([{"match_number": 1, "home_score": 2, "away_score": 0, "played": True}])
    grade_ledger(fixture, tmp_path)
    # re-running with a different prediction must NOT overwrite the graded row
    upsert_ledger([_row(1, "A", "B", 0.1, 0.1, 0.8)], tmp_path)
    led = load_ledger(tmp_path)
    assert led[led.match_number == 1].iloc[0]["p_home"] == 0.5  # original kept
    assert led[led.match_number == 1].iloc[0]["graded"]
