"""Normalise national-team names across sources (fixture feed <-> martj42/prior).

Different providers spell the same country differently ("Türkiye"/"Turkey",
"Korea Republic"/"South Korea"). Cross-source joins silently drop a team when a
name doesn't match, so this curated alias map is load-bearing. ``normalize_team``
maps any known spelling to the canonical martj42 name used by the prior.
"""

from __future__ import annotations

# fixture-feed spelling -> martj42 / prior canonical spelling
ALIASES: dict[str, str] = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "USA": "United States",
    "United States of America": "United States",
    "China PR": "China",
    "Kyrgyz Republic": "Kyrgyzstan",
}


def normalize_team(name: str) -> str:
    """Return the canonical (prior/martj42) name for a team."""
    if name is None:
        return name
    n = name.strip()
    return ALIASES.get(n, n)


def resolve_strength(name: str, prior) -> dict | None:
    """Look up a team's prior strength, trying the alias-normalised name."""
    if prior is None:
        return None
    return prior.strength(name) or prior.strength(normalize_team(name))
