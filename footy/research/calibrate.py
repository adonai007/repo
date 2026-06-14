"""Build a baseline params.json from the statistical prior.

This is both the data-driven fallback (no LLM needed) and the *anchor* the deep
research adjusts. Strengths come straight from the fitted prior; host nations get
the prior's home-advantage, everyone else is neutral.
"""

from __future__ import annotations

from typing import Any

from ..data.teams import normalize_team, resolve_strength

# 2026 hosts get a genuine home advantage even at "neutral" venues.
HOSTS = {"United States", "Mexico", "Canada", "USA"}


def _host_ha(team: str, prior) -> float:
    return prior.home_advantage if normalize_team(team) in HOSTS else 1.0


def baseline_params(match: dict, prior) -> dict[str, Any]:
    """Construct a params.json dict for ``match`` using only the prior.

    ``match`` keys: home, away, date, stage, group (optional), location (optional).
    Raises KeyError if a team has no prior rating (caller should skip/flag).
    """
    home, away = match["home"], match["away"]
    sh = resolve_strength(home, prior)
    sa = resolve_strength(away, prior)
    if sh is None or sa is None:
        raise KeyError(f"missing prior rating for {home if sh is None else away!r}")

    ha_home = _host_ha(home, prior)
    ha_away = _host_ha(away, prior)
    neutral = ha_home == 1.0 and ha_away == 1.0

    raw_date = match.get("date")
    date_str = str(raw_date)[:10] if raw_date not in (None, "") else None
    venue = match.get("location")
    if venue is not None and not isinstance(venue, str):
        venue = None  # guard against NaN

    strength = {
        "home": {"atk": round(sh["atk"], 4), "def": round(sh["def"], 4)},
        "away": {"atk": round(sa["atk"], 4), "def": round(sa["def"], 4)},
    }
    return {
        "schema_version": "1.0",
        "match": {
            "home": home, "away": away, "date": date_str,
            "neutral": neutral, "stage": match.get("stage", "group"),
            "venue": venue,
        },
        "base_rate": round(prior.base_rate, 4),
        "strength": strength,
        "prior_strength": {k: dict(v) for k, v in strength.items()},
        "rho": round(prior.rho, 4),
        "adjustments": {"home_atk": 1.0, "home_def": 1.0, "away_atk": 1.0, "away_def": 1.0},
        "home_advantage": {"home": round(ha_home, 4), "away": round(ha_away, 4)},
        "confidence": "baseline",
        "rationale": "Data-driven prior only (no deep research applied).",
        "research_engine": "prior-baseline",
        "prior_source": f"dixon_coles_fit@{prior.source}:{prior.asof}",
        "n_prior_matches": {
            "home": sh.get("n_matches"), "away": sa.get("n_matches"),
        },
    }
