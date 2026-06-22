"""Score-matrix heatmap (same style as the original deliverable)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def score_heatmap(
    M: np.ndarray,
    home: str,
    away: str,
    out_path: str | Path,
    max_display: int = 6,
    subtitle: str = "Poisson + Dixon-Coles",
) -> Path:
    """Render the scoreline probability matrix to a PNG and return its path.

    Rows = home goals (Y axis), columns = away goals (X axis). Draw cells are
    outlined. Probabilities are shown as percentages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = min(M.shape[0], max_display)
    sub = M[:n, :n] * 100.0

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sub, origin="lower", cmap="OrRd", aspect="equal")

    for i in range(n):
        for j in range(n):
            val = sub[i, j]
            if val < 0.05:
                continue
            color = "white" if val > sub.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=10)

    # outline the draw diagonal
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="teal", linewidth=2.5))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel(f"Goals — {away}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Goals — {home}", fontsize=12, fontweight="bold")
    ax.set_title(f"Scoreline probability (%)\n{home} vs {away}  ({subtitle})",
                 fontsize=13, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Probability (%)")
    fig.text(0.99, 0.01, "Teal outline = draws", ha="right", color="teal", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
