"""One-shot World Cup refresh orchestration.

This is the production-facing pipeline used by automation:

1. refresh the fixture/results feed,
2. optionally refit the statistical prior,
3. predict every known, unplayed match from a chosen date onward,
4. grade newly played matches without rewriting graded predictions,
5. regenerate digest/report/tournament artifacts, and
6. write an auditable pipeline run record.
"""

from __future__ import annotations

import json
import traceback
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.fixture import load_fixture, predictable_matches
from ..ratings.prior_fit import fit_prior
from ..ratings.store import latest_prior, save_prior
from ..simulate.tournament import simulate_tournament
from .pipeline import run_match, write_digest, write_root_report
from .results_store import (
    PREDICTIONS_ROOT,
    code_sha,
    grade_ledger,
    load_ledger,
    upsert_ledger,
)

PIPELINE_RUNS_ROOT = Path("pipeline_runs")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _coerce_day(value: str | date_cls | None) -> date_cls:
    if value is None or value == "today":
        return _utc_now().date()
    if isinstance(value, date_cls):
        return value
    return pd.to_datetime(value).date()


def _utc_timestamp(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _match_day(match: dict) -> date_cls | None:
    if match.get("date") is not None and not pd.isna(match.get("date")):
        return pd.Timestamp(match["date"]).date()
    ko = _utc_timestamp(match.get("datetime_utc"))
    return None if ko is None else ko.date()


def select_matches_for_refresh(
    fixture: pd.DataFrame,
    from_day: str | date_cls | None = "today",
    *,
    now: pd.Timestamp | None = None,
    include_started: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return known, unplayed matches from ``from_day`` onward.

    Matches with kickoff already reached are skipped by default because they are
    no longer honest pre-match forecasts. They can be included explicitly for
    backfills with ``include_started=True``.
    """
    day = _coerce_day(from_day)
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    selected: list[dict] = []
    skipped_started: list[dict] = []
    for _, row in predictable_matches(fixture).iterrows():
        match = row.to_dict()
        match_day = _match_day(match)
        if match_day is None or match_day < day:
            continue
        kickoff = _utc_timestamp(match.get("datetime_utc"))
        if not include_started and kickoff is not None and kickoff <= now:
            skipped_started.append(
                {
                    "match_number": match.get("match_number"),
                    "home": match.get("home"),
                    "away": match.get("away"),
                    "kickoff_utc": kickoff.isoformat(),
                }
            )
            continue
        selected.append(match)
    return selected, skipped_started


def _ledger_counts(root: Path) -> dict:
    led = load_ledger(root)
    if led.empty:
        return {"rows": 0, "graded": 0, "ungraded": 0}
    graded = led["graded"].fillna(False).astype(bool)
    return {"rows": int(len(led)), "graded": int(graded.sum()), "ungraded": int((~graded).sum())}


def _run_filename(started_at: str) -> str:
    safe = (
        started_at.replace(":", "")
        .replace("-", "")
        .replace("+00:00", "Z")
        .replace("+0000", "Z")
    )
    return f"{safe}.json"


def write_pipeline_run(summary: dict, runs_root: Path | str = PIPELINE_RUNS_ROOT) -> Path:
    """Write an immutable run JSON and update ``latest.json``."""
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / _run_filename(str(summary["started_at"]))
    artifacts = dict(summary.get("artifacts") or {})
    artifacts["run"] = str(run_path)
    payload_summary = {**summary, "artifacts": artifacts}
    payload = json.dumps(payload_summary, ensure_ascii=False, indent=2, default=str)
    run_path.write_text(payload, encoding="utf-8")
    (root / "latest.json").write_text(payload, encoding="utf-8")
    return run_path


def refresh_predictions(
    *,
    from_day: str | date_cls | None = "today",
    predictions_root: Path | str = PREDICTIONS_ROOT,
    refit_prior: bool = True,
    asof: str | None = "today",
    runner: Callable | None = None,
    refresh_existing: bool = True,
    include_started: bool = False,
    n_sims: int = 50_000,
    tournament_sims: int = 20_000,
    tournament_seed: int = 0,
    runs_root: Path | str = PIPELINE_RUNS_ROOT,
) -> dict:
    """Run the full refresh pipeline and return its summary."""
    started = _utc_now()
    root = Path(predictions_root)
    day = _coerce_day(from_day)
    summary: dict = {
        "status": "running",
        "started_at": started.isoformat(),
        "finished_at": None,
        "from_day": day.isoformat(),
        "code_sha": code_sha(),
        "config": {
            "refit_prior": refit_prior,
            "asof": asof,
            "research": runner is not None,
            "refresh_existing": refresh_existing,
            "include_started": include_started,
            "n_sims": n_sims,
            "tournament_sims": tournament_sims,
            "tournament_seed": tournament_seed,
        },
        "fixture": {},
        "prior": {},
        "predictions": {"predicted": 0, "errors": [], "skipped_started": []},
        "ledger": {},
        "artifacts": {},
    }

    rows: list[dict] = []
    errors: list[dict] = []
    touched_days: set[str] = set()
    fixture = None

    try:
        if refit_prior:
            prior = fit_prior(asof=asof)
            prior_path = save_prior(prior)
        else:
            prior = latest_prior()
            prior_path = None
        if prior is None:
            raise RuntimeError("no prior found; run `footy fit` first")

        summary["prior"] = {
            "asof": prior.asof,
            "path": str(prior_path) if prior_path else None,
            "teams": len(prior.teams),
            "matches_fit": prior.n_matches_fit,
            "base_rate": prior.base_rate,
            "home_advantage": prior.home_advantage,
            "rho": prior.rho,
        }

        fixture = load_fixture(refresh=True)
        selected, skipped_started = select_matches_for_refresh(
            fixture, from_day=day, include_started=include_started
        )
        summary["fixture"] = {
            "matches": int(len(fixture)),
            "played": int(fixture["played"].fillna(False).astype(bool).sum()),
            "predictable_unplayed": int(len(predictable_matches(fixture))),
            "selected_for_prediction": int(len(selected)),
        }
        summary["predictions"]["skipped_started"] = skipped_started

        for match in selected:
            kickoff = _utc_timestamp(match.get("datetime_utc"))
            utc_date = (kickoff.date() if kickoff is not None else _match_day(match)).isoformat()
            day_dir = root / utc_date
            day_dir.mkdir(parents=True, exist_ok=True)
            touched_days.add(utc_date)
            try:
                row = run_match(match, prior, runner, day_dir, refresh_existing, n_sims)
                rows.append(row)
            except Exception as exc:  # keep the batch moving
                errors.append(
                    {
                        "date": utc_date,
                        "match_number": match.get("match_number"),
                        "match": f"{match.get('home')} vs {match.get('away')}",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        if rows:
            upsert_ledger(rows, root)
        ledger = grade_ledger(fixture, root)

        for touched_day in sorted(touched_days):
            day_dir = root / touched_day
            day_rows = [r for r in rows if str(r.get("date")) == touched_day]
            day_errors = [e for e in errors if e.get("date") == touched_day]
            write_digest(day_dir, day_rows, day_errors)

        report_path = write_root_report(root, fixture)
        summary["artifacts"]["report"] = str(report_path)

        if tournament_sims and tournament_sims > 0:
            tournament_path = Path("tournament") / f"{day.isoformat()}.json"
            tournament_path.parent.mkdir(parents=True, exist_ok=True)
            tournament = simulate_tournament(prior, fixture, n_sims=tournament_sims, seed=tournament_seed)
            tournament_path.write_text(
                json.dumps(tournament, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary["artifacts"]["tournament"] = str(tournament_path)
            summary["tournament"] = {
                "sims": tournament["n_sims"],
                "favorites": tournament.get("favorites", [])[:8],
            }

        summary["predictions"]["predicted"] = int(len(rows))
        summary["predictions"]["errors"] = errors
        summary["ledger"] = _ledger_counts(root)
        summary["status"] = "ok" if not errors else "partial"
        if ledger is not None and not ledger.empty:
            summary["ledger"]["graded"] = int(ledger["graded"].fillna(False).astype(bool).sum())
    except Exception as exc:
        summary["status"] = "failed"
        summary["fatal_error"] = {"error": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        summary["finished_at"] = _utc_now().isoformat()
        summary["artifacts"]["run"] = str(write_pipeline_run(summary, runs_root))

    return summary
