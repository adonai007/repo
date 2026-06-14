"""Offline tests for fixture normalisation / filtering (no network)."""

import pandas as pd

from footy.data.fixture import _normalise, _is_placeholder, matches_on, predictable_matches

RAW = [
    {"MatchNumber": 1, "RoundNumber": 1, "DateUtc": "2026-06-11 19:00:00Z",
     "Location": "Mexico City", "HomeTeam": "Mexico", "AwayTeam": "South Africa",
     "Group": "Group A", "HomeTeamScore": 2, "AwayTeamScore": 0, "Winner": "Mexico"},
    {"MatchNumber": 9, "RoundNumber": 1, "DateUtc": "2026-06-14 23:00:00Z",
     "Location": "X", "HomeTeam": "Cote d'Ivoire", "AwayTeam": "Ecuador",
     "Group": "Group E", "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
    {"MatchNumber": 103, "RoundNumber": 8, "DateUtc": "2026-07-18 21:00:00Z",
     "Location": "Miami", "HomeTeam": "To be announced", "AwayTeam": "To be announced",
     "Group": None, "HomeTeamScore": None, "AwayTeamScore": None, "Winner": ""},
]


def test_placeholder_detection():
    assert _is_placeholder("To be announced")
    assert _is_placeholder("Winner Group A")
    assert _is_placeholder(None)
    assert not _is_placeholder("Brazil")


def test_normalise_fields():
    df = _normalise(RAW)
    assert len(df) == 3
    row = df[df.match_number == 1].iloc[0]
    assert row.played and row.teams_known and row.winner == "Mexico"
    assert row.stage == "group"
    final = df[df.match_number == 103].iloc[0]
    assert not final.teams_known and final.stage == "final"


def test_matches_on_filter():
    df = _normalise(RAW)
    day = matches_on(df, "2026-06-14")
    assert set(day.match_number) == {9}


def test_predictable_excludes_tbd_and_played():
    df = _normalise(RAW)
    pred = predictable_matches(df)
    # match 1 is played, match 103 is TBD -> only match 9 is predictable
    assert set(pred.match_number) == {9}
