"""Human-facing deliverables: per-match Markdown report and score-matrix heatmap."""

from .plots import score_heatmap
from .markdown import match_report

__all__ = ["score_heatmap", "match_report"]
