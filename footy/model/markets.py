"""Derive betting-market probabilities from a score-probability matrix."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchOutcome:
    """Probabilities derived analytically from the score matrix."""

    p_home: float
    p_draw: float
    p_away: float
    p_btts: float
    p_over25: float
    p_under25: float
    exp_goals_home: float
    exp_goals_away: float
    exp_total: float
    top_scores: list[tuple[tuple[int, int], float]] = field(default_factory=list)

    @property
    def p_1x(self) -> float:
        return self.p_home + self.p_draw

    @property
    def p_x2(self) -> float:
        return self.p_draw + self.p_away

    def as_1x2(self) -> tuple[float, float, float]:
        return (self.p_home, self.p_draw, self.p_away)

    def to_dict(self) -> dict:
        return {
            "p_home": self.p_home,
            "p_draw": self.p_draw,
            "p_away": self.p_away,
            "p_btts": self.p_btts,
            "p_over25": self.p_over25,
            "p_under25": self.p_under25,
            "p_1x": self.p_1x,
            "p_x2": self.p_x2,
            "exp_goals_home": self.exp_goals_home,
            "exp_goals_away": self.exp_goals_away,
            "exp_total": self.exp_total,
            "top_scores": [
                {"score": f"{x}-{y}", "p": p} for (x, y), p in self.top_scores
            ],
        }


def outcomes_from_matrix(M: np.ndarray, top_n: int = 8) -> MatchOutcome:
    """Compute 1X2, BTTS, over/under, expected goals and the top scorelines."""
    n = M.shape[0]
    idx = np.arange(n)

    p_home = float(np.tril(M, -1).sum())  # home goals > away goals
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    # both teams to score: exclude row 0 and column 0
    p_btts = float(M[1:, 1:].sum())

    totals = idx[:, None] + idx[None, :]
    p_over25 = float(M[totals > 2].sum())
    p_under25 = 1.0 - p_over25

    eg_home = float((idx[:, None] * M).sum())
    eg_away = float((idx[None, :] * M).sum())

    # top scorelines (limit to a sensible window for readability)
    w = min(n, 6)
    flat = [((x, y), float(M[x, y])) for x in range(w) for y in range(w)]
    flat.sort(key=lambda t: t[1], reverse=True)

    return MatchOutcome(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_btts=p_btts,
        p_over25=p_over25,
        p_under25=p_under25,
        exp_goals_home=eg_home,
        exp_goals_away=eg_away,
        exp_total=eg_home + eg_away,
        top_scores=flat[:top_n],
    )
