"""Deterministic numeric engine: expected goals, Dixon-Coles matrix, markets, knockout."""

from .poisson import expected_goals
from .dixon_coles import score_matrix, tau
from .markets import outcomes_from_matrix, MatchOutcome
from .knockout import advancement_probability

__all__ = [
    "expected_goals",
    "score_matrix",
    "tau",
    "outcomes_from_matrix",
    "MatchOutcome",
    "advancement_probability",
]
