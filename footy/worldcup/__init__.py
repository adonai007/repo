"""World Cup daily pipeline: fixture -> research -> predict -> persist."""

from .pipeline import run_day, run_match
from .results_store import (
    match_slug,
    save_match_artifacts,
    upsert_ledger,
    grade_ledger,
    load_ledger,
)

__all__ = [
    "run_day",
    "run_match",
    "match_slug",
    "save_match_artifacts",
    "upsert_ledger",
    "grade_ledger",
    "load_ledger",
]
