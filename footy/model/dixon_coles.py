"""Dixon-Coles (1997) low-score correction and the score-probability matrix.

Independent Poisson under-counts 0-0 and 1-1 draws and over-counts 1-0 / 0-1.
The ``tau`` factor (governed by ``rho``) re-weights exactly those four cells.
``tau`` here is identical to the original script's ``tau()``.
"""

from __future__ import annotations

import numpy as np

from .poisson import poisson_pmf_vector


def tau(x: int, y: int, lam_x: float, lam_y: float, rho: float) -> float:
    """Dixon-Coles correction factor for a single scoreline ``(x, y)``."""
    if x == 0 and y == 0:
        return 1.0 - lam_x * lam_y * rho
    if x == 0 and y == 1:
        return 1.0 + lam_x * rho
    if x == 1 and y == 0:
        return 1.0 + lam_y * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam_home: float, lam_away: float, rho: float, max_goals: int = 10
) -> np.ndarray:
    """Normalised score-probability matrix.

    ``M[i, j]`` = P(home scores ``i``, away scores ``j``). Rows are the home
    team, columns the away team. Renormalised after truncation + correction so
    the matrix sums to 1.
    """
    ph = poisson_pmf_vector(lam_home, max_goals)
    pa = poisson_pmf_vector(lam_away, max_goals)
    M = np.outer(ph, pa)  # independent Poisson

    # Apply the four-cell Dixon-Coles correction.
    M[0, 0] *= 1.0 - lam_home * lam_away * rho
    M[0, 1] *= 1.0 + lam_home * rho
    M[1, 0] *= 1.0 + lam_away * rho
    M[1, 1] *= 1.0 - rho

    total = M.sum()
    if total <= 0:
        raise ValueError("score matrix collapsed to zero mass; check lambdas/rho")
    return M / total
