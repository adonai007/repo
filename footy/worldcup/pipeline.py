"""Daily World Cup pipeline: for each match of a date, research + predict + persist."""

from __future__ import annotations

import json
from datetime import date as date_cls
from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.fixture import load_fixture, matches_on, predictable_matches
from ..ratings.store import latest_prior
from ..research.engine import research_match
from ..schema import MatchParams
from ..predict import predict
from .results_store import (
    PREDICTIONS_ROOT, match_slug, save_match_artifacts, ledger_row,
    upsert_ledger, grade_ledger, load_ledger,
)


def run_match(match: dict, prior, runner: Callable | None, day_dir: Path,
              refresh: bool = False, n_sims: int = 50_000) -> dict:
    """Research + predict + persist a single match. Returns its ledger row."""
    slug = match_slug(match.get("match_number"), match["home"], match["away"])
    cache_dir = Path(day_dir) / slug
    research = research_match(match, prior, runner=runner, cache_dir=cache_dir, refresh=refresh)
    params = research["params"]
    mp = MatchParams.parse(params)
    result = predict(mp, n_sims=n_sims, run_mc=n_sims > 0)
    result["citations"] = params.get("citations", [])
    save_match_artifacts(day_dir, slug, result, params, research["dossier"])
    return ledger_row(match, result, params)


def run_day(
    day: str | date_cls,
    prior=None,
    runner: Callable | None = None,
    predictions_root: Path | str = PREDICTIONS_ROOT,
    refresh: bool = False,
    n_sims: int = 50_000,
) -> dict:
    """Process every predictable match on ``day``; persist artifacts + ledger + reports."""
    if isinstance(day, str):
        day = pd.to_datetime(day).date()
    prior = prior or latest_prior()
    if prior is None:
        raise RuntimeError("no prior found; run `footy fit` first")

    fixture = load_fixture()
    root = Path(predictions_root)
    day_dir = root / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    todays = predictable_matches(matches_on(fixture, day))
    rows, errors = [], []
    for _, m in todays.iterrows():
        match = m.to_dict()
        try:
            rows.append(run_match(match, prior, runner, day_dir, refresh, n_sims))
        except Exception as exc:  # isolation: one bad match never breaks the batch
            errors.append({"match": f"{match['home']} vs {match['away']}", "error": str(exc)})

    if rows:
        upsert_ledger(rows, root)
    grade_ledger(fixture, root)
    write_digest(day_dir, rows, errors)
    write_root_report(root, fixture)

    return {"date": day.isoformat(), "predicted": len(rows),
            "errors": errors, "day_dir": str(day_dir)}


def _pct(x) -> str:
    return f"{x*100:.0f}%" if x is not None else "—"


def write_digest(day_dir: Path, rows: list[dict], errors: list[dict]) -> Path:
    day = Path(day_dir).name
    lines = [f"# World Cup predictions — {day}", ""]
    if not rows:
        lines.append("_No predictable matches (all played or teams TBD)._")
    else:
        lines += ["| Match | Stage | Model (H/D/A) | Most likely |", "|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['home']} vs {r['away']} | {r['stage']} | "
                f"{_pct(r['p_home'])} / {_pct(r['p_draw'])} / {_pct(r['p_away'])} | {r['top_score']} |"
            )
    if errors:
        lines += ["", "## Skipped", ""] + [f"- {e['match']}: {e['error']}" for e in errors]
    lines += ["", "---", "*Informational model output — not betting advice.*"]
    path = Path(day_dir) / "digest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_root_report(root: Path, fixture: pd.DataFrame) -> Path:
    """Repo-root REPORT.md: upcoming predictions + the accuracy scoreboard."""
    from ..eval.metrics import summarize
    led = load_ledger(root)
    lines = ["# ⚽ footy — World Cup 2026 predictions", "",
             "Auto-generated. *Informational model output — not betting advice.*", ""]

    graded = led[led["graded"] == True] if not led.empty else led  # noqa: E712
    if not led.empty and len(graded):
        fc = [((r.p_home, r.p_draw, r.p_away), r.actual_result) for r in graded.itertuples()]
        s = summarize(fc)
        lines += ["## Scoreboard (graded matches)", "",
                  f"- Matches graded: **{s['n']}**",
                  f"- Mean RPS: **{s['rps']:.3f}**  ·  log-loss: {s['log_loss']:.3f}  ·  "
                  f"Brier: {s['brier']:.3f}",
                  f"- Hit rate (argmax): **{s['accuracy']*100:.0f}%**", ""]

    if not led.empty:
        upcoming = led[led["graded"] != True].sort_values("match_number").head(20)  # noqa: E712
        if len(upcoming):
            lines += ["## Upcoming predictions", "",
                      "| # | Date | Match | Model (H/D/A) | Pick |", "|--|--|--|--|--|"]
            for r in upcoming.itertuples():
                lines.append(
                    f"| {r.match_number} | {r.date} | {r.home} vs {r.away} | "
                    f"{_pct(r.p_home)} / {_pct(r.p_draw)} / {_pct(r.p_away)} | {r.top_score} |"
                )
            lines.append("")

    path = Path(root).parent / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
