from datetime import date

import pandas as pd

from footy.worldcup.refresh import select_matches_for_refresh, write_pipeline_run


def _fixture():
    return pd.DataFrame(
        [
            {
                "match_number": 1,
                "datetime_utc": pd.Timestamp("2026-06-22T13:00:00Z"),
                "date": date(2026, 6, 22),
                "home": "A",
                "away": "B",
                "teams_known": True,
                "played": False,
            },
            {
                "match_number": 2,
                "datetime_utc": pd.Timestamp("2026-06-22T17:00:00Z"),
                "date": date(2026, 6, 22),
                "home": "C",
                "away": "D",
                "teams_known": True,
                "played": False,
            },
            {
                "match_number": 3,
                "datetime_utc": pd.Timestamp("2026-06-23T17:00:00Z"),
                "date": date(2026, 6, 23),
                "home": "E",
                "away": "F",
                "teams_known": True,
                "played": True,
            },
            {
                "match_number": 4,
                "datetime_utc": pd.Timestamp("2026-06-24T17:00:00Z"),
                "date": date(2026, 6, 24),
                "home": "1A",
                "away": "2B",
                "teams_known": False,
                "played": False,
            },
        ]
    )


def test_select_matches_skips_started_and_non_predictable():
    selected, skipped = select_matches_for_refresh(
        _fixture(),
        from_day="2026-06-22",
        now=pd.Timestamp("2026-06-22T15:00:00Z"),
    )

    assert [m["match_number"] for m in selected] == [2]
    assert [m["match_number"] for m in skipped] == [1]


def test_select_matches_can_include_started_for_backfill():
    selected, skipped = select_matches_for_refresh(
        _fixture(),
        from_day="2026-06-22",
        now=pd.Timestamp("2026-06-22T15:00:00Z"),
        include_started=True,
    )

    assert [m["match_number"] for m in selected] == [1, 2]
    assert skipped == []


def test_write_pipeline_run_updates_latest(tmp_path):
    summary = {"started_at": "2026-06-22T15:00:00+00:00", "status": "ok"}
    path = write_pipeline_run(summary, tmp_path)

    assert path.exists()
    assert (tmp_path / "latest.json").exists()
    assert "20260622T150000Z" in path.name
