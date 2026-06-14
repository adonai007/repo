"""Prompt construction for the deep-research calibration step.

The LLM produces a sourced dossier and a params.json. Crucially it only *sets
parameters* (it never computes probabilities) and anchors strengths to the
statistical prior — the SOTA inputs (Elo, squad value, xG/xGA, form, lineups,
goalkeeper, injuries) justify *adjustments* to that anchor, with citations.
"""

from __future__ import annotations

import json

CHECKLIST = """\
Research these inputs (ordered by predictive power for international football) and cite sources:
 1. World Football Elo rating of each side and the gap (strongest single predictor).
 2. Squad market value + average age (Transfermarkt) — PELE-style.
 3. Recent xG / xGA (FBref/Understat); if possible split set-piece vs open-play.
 4. Form over the last ~9 matches (results AND performance, recency-weighted).
 5. Probable XI + the starters' club xG/xA per 90.
 6. Goalkeeper quality (post-shot xG faced vs conceded, save%).
 7. Injuries/absences — weight by player importance x replacement quality x timing.
 8. Rest days / travel / congestion; venue (neutral vs host advantage).
 9. Set pieces & discipline (corners, penalties, cards).
10. Style/matchup notes (e.g. counter-attack vs high line, key duels).
11. Closing market odds (average of books) — strong benchmark.
12. Stage / motivation (group vs knockout)."""


def build_prompt(baseline: dict) -> str:
    """Assemble the research+calibration prompt for one match."""
    m = baseline["match"]
    prior = baseline["prior_strength"]
    title = f"{m['home']} vs {m['away']}"
    when = f" on {m['date']}" if m.get("date") else ""
    stage = m.get("stage", "group")

    return f"""You are an elite football analyst calibrating a Dixon-Coles match model for the 2026 World Cup.

MATCH: {title}{when} (stage: {stage}, venue: {m.get('venue') or 'TBD'}, neutral: {m.get('neutral')}).

Do a focused deep-research pass on this specific match, then output a calibrated params.json.

{CHECKLIST}

STATISTICAL PRIOR (anchor — do NOT invent strengths from scratch):
  base_rate = {baseline['base_rate']}, rho = {baseline['rho']}
  {m['home']} (home): atk={prior['home']['atk']}, def={prior['home']['def']}
  {m['away']} (away): atk={prior['away']['atk']}, def={prior['away']['def']}
(atk: higher = stronger attack; def: LOWER = stronger defence. These already encode long-run strength.)

RULES:
- You set PARAMETERS ONLY. Never compute or state win/draw probabilities — the engine does that.
- Anchor `strength` to the prior. Move a value at most ~40% from the prior, and ONLY with cited evidence
  (form, injuries, lineup, matchup). Keep `prior_strength` unchanged for traceability.
- Encode injuries/absences and form as `adjustments` multipliers (0.5..1.5) on atk/def, not by overwriting strength.
- If you find closing odds, fill `market_odds` (American or decimal) with source + date.
- Every non-obvious number must have a citation. Set `confidence` to alta/media/baja.

OUTPUT FORMAT (exactly, in this order):
1. A markdown dossier: TL;DR, key findings per the checklist with inline source citations, and a short rationale.
2. Then a single fenced ```json code block containing the params.json with these keys:
   schema_version, match, base_rate, strength, prior_strength, rho, adjustments, home_advantage,
   market_odds (optional), scenarios (optional), evidence (optional), context (optional),
   confidence, citations (list of URLs), rationale.

Here is the baseline params.json to refine (keep its structure):
```json
{json.dumps(baseline, ensure_ascii=False, indent=2)}
```
"""
