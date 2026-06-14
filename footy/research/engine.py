"""Run the deep-research calibration for one match via Claude Code headless.

``research_match`` builds the prior baseline, asks the runner (``claude -p`` by
default) to produce a sourced dossier + refined params.json, validates it against
the schema, clips strengths to a sane band around the prior (anti-hallucination),
and caches both artifacts. On any failure it degrades gracefully to the prior
baseline so a single bad match never breaks the batch.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..schema import MatchParams, ParamsError
from .calibrate import baseline_params
from .prompts import build_prompt

# how far a researched strength may stray from the prior anchor
MAX_DEVIATION = 0.40

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ClaudeCliRunner:
    """Invoke Claude Code headless (`claude -p`) and return its stdout."""

    # Headless research needs live web tools and no interactive permission prompts.
    DEFAULT_ARGS = ["--dangerously-skip-permissions",
                    "--allowedTools", "WebSearch", "WebFetch"]

    def __init__(self, model: str | None = None, timeout: int = 900,
                 extra_args: list[str] | None = None):
        self.model = model
        self.timeout = timeout
        self.extra_args = self.DEFAULT_ARGS if extra_args is None else extra_args

    def __call__(self, prompt: str) -> str:
        cmd = ["claude", "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]
        cmd += self.extra_args
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout, check=True
        )
        return out.stdout


def _extract(raw: str) -> tuple[dict, str]:
    """Split runner output into (params dict, dossier markdown)."""
    matches = list(_JSON_BLOCK.finditer(raw))
    if not matches:
        raise ValueError("no json block in research output")
    block = matches[-1]
    params = json.loads(block.group(1))
    dossier = (raw[: block.start()]).strip() or raw.strip()
    return params, dossier


def _clip_to_prior(params: dict) -> dict:
    """Clamp researched strengths to +/-MAX_DEVIATION of the prior anchor."""
    prior = params.get("prior_strength")
    if not prior:
        return params
    for side in ("home", "away"):
        for k in ("atk", "def"):
            p = prior[side][k]
            lo, hi = p * (1 - MAX_DEVIATION), p * (1 + MAX_DEVIATION)
            v = params["strength"][side][k]
            params["strength"][side][k] = float(min(max(v, lo), hi))
    return params


def research_match(
    match: dict,
    prior,
    runner: Callable[[str], str] | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Return ``{"params": ..., "dossier": ..., "engine": ...}`` for one match."""
    baseline = baseline_params(match, prior)
    cache = Path(cache_dir) if cache_dir else None

    if cache and not refresh and (cache / "params.json").exists():
        params = json.loads((cache / "params.json").read_text())
        dossier = (cache / "dossier.md").read_text() if (cache / "dossier.md").exists() else ""
        return {"params": params, "dossier": dossier, "engine": params.get("research_engine", "cache")}

    params, dossier, engine = baseline, baseline["rationale"], "prior-baseline"

    if runner is not None:
        prompt = build_prompt(baseline)
        for attempt in range(max_retries + 1):
            try:
                raw = runner(prompt)
                cand, doss = _extract(raw)
                # preserve the prior anchor + provenance, then validate
                cand["prior_strength"] = baseline["prior_strength"]
                cand.setdefault("base_rate", baseline["base_rate"])
                cand = _clip_to_prior(cand)
                MatchParams.parse(cand)  # validate; raises on bad output
                cand["research_engine"] = "claude-code-cli/deep-research"
                params, dossier, engine = cand, doss, cand["research_engine"]
                break
            except (subprocess.SubprocessError, ValueError, KeyError, ParamsError) as exc:
                if attempt >= max_retries:
                    params.setdefault("research_note", f"research failed, using prior baseline: {exc}")
                continue

    params["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2))
        (cache / "dossier.md").write_text(dossier or "")

    return {"params": params, "dossier": dossier, "engine": engine}
