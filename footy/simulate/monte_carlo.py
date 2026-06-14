"""Monte Carlo simulation by sampling scorelines from the score matrix.

This mirrors the original script: rather than sampling two independent Poissons
(which would discard the Dixon-Coles dependence), we sample whole scorelines from
the corrected joint distribution ``M``. The MC estimates must therefore agree
with the analytic outcomes up to sampling noise — that agreement is a test.
"""

from __future__ import annotations

import numpy as np


def simulate_from_matrix(M: np.ndarray, n_sims: int = 200_000, seed: int = 42) -> dict:
    """Sample ``n_sims`` scorelines from ``M`` and return empirical 1X2 + goals."""
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    probs = M.flatten()
    probs = probs / probs.sum()
    idx = rng.choice(probs.size, size=n_sims, p=probs)
    gx = idx // n  # home goals
    gy = idx % n   # away goals

    return {
        "p_home": float(np.mean(gx > gy)),
        "p_draw": float(np.mean(gx == gy)),
        "p_away": float(np.mean(gx < gy)),
        "mean_goals": float(np.mean(gx + gy)),
        "n_sims": int(n_sims),
    }
