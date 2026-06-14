"""Contract between the research layer and the numeric engine: ``params.json``.

The research step (an LLM) emits a ``params.json`` describing a single match.
This module parses it into a typed, validated :class:`MatchParams` with sane
defaults so the *minimum required* core (``match``, ``base_rate``, ``strength``,
``rho``, ``adjustments``, ``home_advantage``) reproduces the original hand-rolled
script exactly, while every other field is optional.

Guards here are the anti-hallucination backstop: the LLM only *sets parameters*;
this module rejects values outside plausible ranges before they reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from . import SCHEMA_VERSION

# --- plausibility guards (reject obviously broken LLM output) -----------------
ATK_RANGE = (0.3, 3.0)        # 1.0 = average elite national side
DEF_RANGE = (0.3, 3.0)        # <1.0 concedes less than average
RHO_RANGE = (-0.2, 0.2)       # Dixon-Coles low-score dependence
ADJ_RANGE = (0.5, 1.5)        # context multipliers (injuries/form)
BASE_RATE_RANGE = (0.8, 2.0)  # baseline goals per elite side
HA_RANGE = (0.8, 1.4)         # home-advantage multiplier


class ParamsError(ValueError):
    """Raised when a params.json payload is structurally invalid or out of range."""


def _check(name: str, value: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ParamsError(f"{name!r} must be numeric, got {value!r}")
    if not (lo <= v <= hi):
        raise ParamsError(f"{name!r}={v} out of allowed range [{lo}, {hi}]")
    return v


@dataclass
class TeamStrength:
    atk: float
    deff: float  # 'def' is a reserved word

    @classmethod
    def parse(cls, side: str, d: dict[str, Any]) -> "TeamStrength":
        if not isinstance(d, dict) or "atk" not in d or "def" not in d:
            raise ParamsError(f"strength.{side} must have 'atk' and 'def'")
        return cls(
            atk=_check(f"strength.{side}.atk", d["atk"], *ATK_RANGE),
            deff=_check(f"strength.{side}.def", d["def"], *DEF_RANGE),
        )

    def to_dict(self) -> dict[str, float]:
        return {"atk": self.atk, "def": self.deff}


@dataclass
class MatchInfo:
    home: str
    away: str
    date: str | None = None
    neutral: bool = True
    stage: str = "group"          # group | round_of_32 | round_of_16 | quarter | semi | final
    venue: str | None = None

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "MatchInfo":
        if not isinstance(d, dict) or "home" not in d or "away" not in d:
            raise ParamsError("match must have 'home' and 'away'")
        return cls(
            home=str(d["home"]),
            away=str(d["away"]),
            date=d.get("date"),
            neutral=bool(d.get("neutral", True)),
            stage=str(d.get("stage", "group")),
            venue=d.get("venue"),
        )

    @property
    def is_knockout(self) -> bool:
        return self.stage not in ("group", "league")


@dataclass
class MatchParams:
    """Fully-parsed, validated inputs for one match prediction."""

    match: MatchInfo
    base_rate: float
    strength_home: TeamStrength
    strength_away: TeamStrength
    rho: float
    # adjustments: multipliers applied to each (team, atk/def) component of lambda
    adj_home_atk: float = 1.0
    adj_home_def: float = 1.0
    adj_away_atk: float = 1.0
    adj_away_def: float = 1.0
    ha_home: float = 1.0
    ha_away: float = 1.0
    expected_goals_override: dict[str, float] | None = None
    penalty_strength: dict[str, float] | None = None
    market_odds: dict[str, Any] | None = None
    blend_weight: float = 0.0
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    # provenance / audit
    prior_strength: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    confidence: str | None = None
    citations: list[str] = field(default_factory=list)
    rationale: str | None = None
    generated_at: str | None = None
    research_engine: str | None = None
    prior_source: str | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def parse(cls, d: dict[str, Any]) -> "MatchParams":
        if not isinstance(d, dict):
            raise ParamsError("params must be a JSON object")
        match = MatchInfo.parse(d.get("match", {}))
        strength = d.get("strength", {})
        if "home" not in strength or "away" not in strength:
            raise ParamsError("strength must have 'home' and 'away'")
        adj = d.get("adjustments", {}) or {}
        ha = d.get("home_advantage", {}) or {}

        ego = d.get("expected_goals_override")
        if ego is not None:
            ego = {
                "home": _check("expected_goals_override.home", ego["home"], 0.05, 6.0),
                "away": _check("expected_goals_override.away", ego["away"], 0.05, 6.0),
            }

        pen = None
        if isinstance(d.get("knockout"), dict):
            ps = d["knockout"].get("penalty_strength")
            if isinstance(ps, dict):
                pen = {
                    "home": _check("penalty_strength.home", ps.get("home", 0.5), 0.1, 0.9),
                    "away": _check("penalty_strength.away", ps.get("away", 0.5), 0.1, 0.9),
                }

        return cls(
            match=match,
            base_rate=_check("base_rate", d.get("base_rate", 1.30), *BASE_RATE_RANGE),
            strength_home=TeamStrength.parse("home", strength["home"]),
            strength_away=TeamStrength.parse("away", strength["away"]),
            rho=_check("rho", d.get("rho", 0.0), *RHO_RANGE),
            adj_home_atk=_check("adjustments.home_atk", adj.get("home_atk", 1.0), *ADJ_RANGE),
            adj_home_def=_check("adjustments.home_def", adj.get("home_def", 1.0), *ADJ_RANGE),
            adj_away_atk=_check("adjustments.away_atk", adj.get("away_atk", 1.0), *ADJ_RANGE),
            adj_away_def=_check("adjustments.away_def", adj.get("away_def", 1.0), *ADJ_RANGE),
            ha_home=_check("home_advantage.home", ha.get("home", 1.0), *HA_RANGE),
            ha_away=_check("home_advantage.away", ha.get("away", 1.0), *HA_RANGE),
            expected_goals_override=ego,
            penalty_strength=pen,
            market_odds=d.get("market_odds"),
            blend_weight=_check("blend_weight", d.get("blend_weight", 0.0), 0.0, 1.0),
            scenarios=list(d.get("scenarios", []) or []),
            prior_strength=d.get("prior_strength"),
            evidence=d.get("evidence"),
            context=d.get("context"),
            confidence=d.get("confidence"),
            citations=list(d.get("citations", []) or []),
            rationale=d.get("rationale"),
            generated_at=d.get("generated_at"),
            research_engine=d.get("research_engine"),
            prior_source=d.get("prior_source"),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
