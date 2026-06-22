"""Research layer: turn a fixture into a params.json via prior + deep research."""

from .calibrate import baseline_params, HOSTS
from .engine import research_match, ClaudeCliRunner

__all__ = ["baseline_params", "HOSTS", "research_match", "ClaudeCliRunner"]
