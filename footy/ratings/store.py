"""Persist / load fitted priors as dated JSON (committed for reproducibility)."""

from __future__ import annotations

import json
from pathlib import Path

from .prior_fit import PriorRatings

RATINGS_DIR = Path("ratings")


def save_prior(prior: PriorRatings, ratings_dir: Path | str = RATINGS_DIR) -> Path:
    ratings_dir = Path(ratings_dir)
    ratings_dir.mkdir(parents=True, exist_ok=True)
    path = ratings_dir / f"{prior.asof}.json"
    path.write_text(json.dumps(prior.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_prior(path: Path | str) -> PriorRatings:
    return PriorRatings.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def latest_prior(ratings_dir: Path | str = RATINGS_DIR) -> PriorRatings | None:
    ratings_dir = Path(ratings_dir)
    snaps = sorted(ratings_dir.glob("*.json"))
    return load_prior(snaps[-1]) if snaps else None
