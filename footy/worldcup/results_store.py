"""Durable, reproducible persistence of predictions (ephemeral env -> commit).

Layout::

    predictions/<date>/<slug>/  dossier.md params.json prediction.json report.md heatmap.png
    predictions/<date>/digest.md
    predictions/ledger.parquet          # append-only, one row per prediction (+ actual + RPS)

Pre-match predictions are frozen with ``generated_at`` + ``code_sha``; grading
later joins actual results in without recomputing the probabilities.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.metrics import rps_1x2

PREDICTIONS_ROOT = Path("predictions")
LEDGER_NAME = "ledger.parquet"

_LEDGER_COLS = [
    "match_number", "date", "stage", "group", "home", "away",
    "p_home", "p_draw", "p_away", "blend_home", "blend_draw", "blend_away",
    "mkt_home", "mkt_draw", "mkt_away", "top_score", "lambda_home", "lambda_away",
    "engine", "confidence", "generated_at", "code_sha",
    "actual_home", "actual_away", "actual_result", "rps", "graded",
]


def code_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def match_slug(match_number: int | None, home: str, away: str) -> str:
    def s(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")[:16]
    num = f"{int(match_number):03d}-" if match_number is not None else ""
    return f"{num}{s(home)}-vs-{s(away)}"


def save_match_artifacts(day_dir: Path, slug: str, result: dict, params: dict,
                         dossier: str) -> Path:
    """Write all per-match deliverables and return the match directory."""
    from ..report import match_report, score_heatmap

    mdir = Path(day_dir) / slug
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2))
    (mdir / "dossier.md").write_text(dossier or "")

    M = np.array(result["score_matrix"])
    score_heatmap(M, result["match"]["home"], result["match"]["away"], mdir / "heatmap.png")
    (mdir / "report.md").write_text(match_report(result, heatmap_rel="heatmap.png"))

    slim = {k: v for k, v in result.items() if k != "score_matrix"}
    slim["code_sha"] = code_sha()
    (mdir / "prediction.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2))
    return mdir


def ledger_row(match: dict, result: dict, params: dict) -> dict:
    a = result["analytic"]
    mk = result.get("market", {})
    bl = mk.get("blended", {})
    imp = mk.get("implied", {})
    return {
        "match_number": match.get("match_number"),
        "date": match.get("date"),
        "stage": match.get("stage", "group"),
        "group": match.get("group"),
        "home": match["home"], "away": match["away"],
        "p_home": a["p_home"], "p_draw": a["p_draw"], "p_away": a["p_away"],
        "blend_home": bl.get("home"), "blend_draw": bl.get("draw"), "blend_away": bl.get("away"),
        "mkt_home": imp.get("home"), "mkt_draw": imp.get("draw"), "mkt_away": imp.get("away"),
        "top_score": a["top_scores"][0]["score"],
        "lambda_home": result["lambda"]["home"], "lambda_away": result["lambda"]["away"],
        "engine": params.get("research_engine"), "confidence": params.get("confidence"),
        "generated_at": params.get("generated_at"), "code_sha": code_sha(),
        "actual_home": None, "actual_away": None, "actual_result": None,
        "rps": None, "graded": False,
    }


def load_ledger(root: Path | str = PREDICTIONS_ROOT) -> pd.DataFrame:
    path = Path(root) / LEDGER_NAME
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_LEDGER_COLS)


def upsert_ledger(rows: list[dict], root: Path | str = PREDICTIONS_ROOT) -> pd.DataFrame:
    """Insert/replace rows keyed by match_number; preserve graded actuals."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    led = load_ledger(root)
    new = pd.DataFrame(rows)
    if not led.empty:
        # keep already-graded rows untouched (immutability of played predictions)
        graded_keys = set(led.loc[led["graded"] == True, "match_number"])  # noqa: E712
        new = new[~new["match_number"].isin(graded_keys)]
        led = led[~led["match_number"].isin(set(new["match_number"]))]
    out = pd.concat([led, new], ignore_index=True)
    out = out.reindex(columns=_LEDGER_COLS)
    out = out.sort_values("match_number").reset_index(drop=True)
    out.to_parquet(root / LEDGER_NAME, index=False)
    return out


def grade_ledger(fixture: pd.DataFrame, root: Path | str = PREDICTIONS_ROOT) -> pd.DataFrame:
    """Fill actual results + RPS for predicted matches that have since been played."""
    root = Path(root)
    led = load_ledger(root)
    if led.empty:
        return led
    played = fixture[fixture["played"]].set_index("match_number")
    for i, row in led.iterrows():
        if row.get("graded"):
            continue
        mn = row["match_number"]
        if mn in played.index:
            hs = int(played.loc[mn, "home_score"])
            as_ = int(played.loc[mn, "away_score"])
            res = "home" if hs > as_ else "away" if as_ > hs else "draw"
            probs = (row["p_home"], row["p_draw"], row["p_away"])
            led.at[i, "actual_home"] = hs
            led.at[i, "actual_away"] = as_
            led.at[i, "actual_result"] = res
            led.at[i, "rps"] = rps_1x2(probs, res)
            led.at[i, "graded"] = True
    led.to_parquet(root / LEDGER_NAME, index=False)
    return led
