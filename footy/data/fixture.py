"""World Cup 2026 fixture ingestion.

Primary source is a clean public JSON feed (match number, UTC kickoff, venue,
group, teams, and live scores as they are played). Knockout matches whose teams
are not yet resolved show a placeholder ("To be announced") and are flagged so
the pipeline can skip them until the bracket fills in.

Robustness (the runner may be offline / the feed may move): every fetch writes a
dated CSV snapshot under ``data/snapshots/`` and ``load_fixture`` falls back to the
most recent snapshot when the network is unavailable.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

FIXTURE_SOURCE = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
SNAPSHOT_DIR = Path("data/snapshots")

# Strings that indicate the team slot is not yet resolved.
_PLACEHOLDERS = ("to be announced", "tbd", "winner", "runner", "loser", "1st", "2nd", "3rd", "place")


def _is_placeholder(name: str | None) -> bool:
    if not name:
        return True
    s = name.strip()
    # Bracket slots reference group positions: "1K", "2A", "3ABCDF" — they start
    # with a digit, which no real national-team name does.
    if s[0].isdigit():
        return True
    return any(tok in s.lower() for tok in _PLACEHOLDERS)


def _fetch_raw(url: str = FIXTURE_SOURCE, timeout: int = 30) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "footy/0.1 (+https://github.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _normalise(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        home, away = m.get("HomeTeam"), m.get("AwayTeam")
        hs, as_ = m.get("HomeTeamScore"), m.get("AwayTeamScore")
        dt = pd.to_datetime(m.get("DateUtc"), utc=True, errors="coerce")
        rows.append(
            {
                "match_number": m.get("MatchNumber"),
                "round": m.get("RoundNumber"),
                "datetime_utc": dt,
                "date": dt.date() if pd.notna(dt) else None,
                "location": m.get("Location"),
                "group": m.get("Group"),
                "home": home,
                "away": away,
                "home_score": hs,
                "away_score": as_,
                "winner": m.get("Winner") or None,
                "teams_known": not (_is_placeholder(home) or _is_placeholder(away)),
                "played": hs is not None and as_ is not None,
                "stage": _stage_for_round(m.get("RoundNumber")),
            }
        )
    df = pd.DataFrame(rows).sort_values("match_number").reset_index(drop=True)
    return df


def _stage_for_round(rnd: int | None) -> str:
    # Rounds 1-3 = group matchdays; 4-8 = knockout bracket.
    return {
        4: "round_of_32",
        5: "round_of_16",
        6: "quarter",
        7: "semi",
        8: "final",
    }.get(rnd, "group")


def load_fixture(
    season: str = "2026",
    refresh: bool = False,
    snapshot_dir: Path | str = SNAPSHOT_DIR,
) -> pd.DataFrame:
    """Return the fixture as a DataFrame, fetching + snapshotting or using cache.

    With ``refresh=False`` and no network, falls back to the latest snapshot.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    today_snap = snapshot_dir / f"fixture-{season}-{date_cls.today().isoformat()}.csv"

    if not refresh and today_snap.exists():
        return _read_snapshot(today_snap)

    try:
        df = _normalise(_fetch_raw())
        df.to_csv(today_snap, index=False)
        return df
    except Exception as exc:  # network/feed failure -> latest snapshot
        snaps = sorted(snapshot_dir.glob(f"fixture-{season}-*.csv"))
        if snaps:
            return _read_snapshot(snaps[-1])
        raise RuntimeError(
            f"could not fetch fixture from {FIXTURE_SOURCE} and no snapshot found"
        ) from exc


def _read_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def matches_on(df: pd.DataFrame, day: str | date_cls) -> pd.DataFrame:
    """All matches kicking off on a given calendar date (UTC)."""
    if isinstance(day, str):
        day = pd.to_datetime(day).date()
    return df[df["date"] == day].reset_index(drop=True)


def predictable_matches(df: pd.DataFrame, include_played: bool = False) -> pd.DataFrame:
    """Matches with both teams known (skip TBD knockout slots)."""
    mask = df["teams_known"]
    if not include_played:
        mask = mask & ~df["played"]
    return df[mask].reset_index(drop=True)
