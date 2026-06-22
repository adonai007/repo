Enough data. Compiling. Key facts:

- **Elo** (eloratings.net via Jan-19-2026 snapshot): France 2063 (3rd) vs Senegal 1869 (17th), gap ~194 — France clear favorite.
- **Odds** (FanDuel/CBS, match day): France -220, Draw +350, Senegal +600; O/U 2.5 (Over -140).
- **France form**: beat Brazil 2-1, Colombia 3-1 (Mar); lost Ivory Coast 2-1, beat N.Ireland 3-1 (Jun); qualifying 5-1-0, only 4 conceded; scored 2+ in 9 of last 10.
- **Senegal form**: undefeated qualifying, AFCON 2026 finalists, scored in 17 straight before a scoreless friendly vs Saudi.
- **XIs**: Fr — Maignan; Koundé, Saliba, Upamecano, T.Hernandez; Tchouaméni, Rabiot; Dembélé, Doué, Olise; Mbappé. Sen — Mendy; Diatta, Koulibaly, Niakhaté, M.Diouf; P.Gueye, L.Camara, H.Diarra; I.Sarr, Jackson, Mané.
- No major injuries either side; neutral venue (MetLife), both travel.

Writing output.

---

# Dossier — France vs Senegal (2026-06-16, World Cup Group I, MetLife/NY-NJ, neutral)

## TL;DR
France enters as a clear favorite on every predictive axis: a ~190-point Elo edge, deeper squad value, a sharper recent scoring run, and a back line that conceded only four goals across qualifying. Senegal is no minnow — AFCON 2026 finalists, undefeated in qualifying, strong spine (Koulibaly, Mendy, Mané, Jackson) — but the gap is real. Adjustments are kept modest and evidence-led: a small positive on France's attack and defence form, Senegal left at prior. Strength values stay anchored to the statistical prior.

## Key findings (per checklist)

1. **Elo (strongest single predictor).** France ≈ 2063 (3rd), Senegal ≈ 1869 (17th) as of the Jan-19-2026 snapshot; gap ≈ 194 favoring France. A gap this size historically maps to a strong but not overwhelming favorite at a neutral site. [eloratings via Wikipedia; footballratings.org]

2. **Squad value / age (PELE-style).** France carries one of the highest squad market values in the tournament (Mbappé, Dembélé — 2025 Ballon d'Or, Olise, Saliba, Upamecano), prime-age core. Senegal's value is solid-mid (Premier League / Ligue 1 spine) with veterans Mané (34) and Koulibaly. Clear France edge — already embedded in the prior. [ESPN squad; FIFA squad announcement]

3. **xG / xGA.** Direct match xG splits unavailable pre-match; proxy from results: France allowed only 4 goals in qualifying (5-1-0) and scored 2+ in 9 of their last 10 — high open-play output. Senegal scored in 17 consecutive matches before a single scoreless friendly. [CBS preview]

4. **Form (last ~9, recency-weighted).** France: W Brazil 2-1, W Colombia 3-1 (Mar), **L Ivory Coast 2-1**, W N.Ireland 3-1 (Jun) — strong but with one defensive wobble. Senegal: undefeated qualifying + AFCON final run; lone blemish a 0-0 friendly vs Saudi Arabia. Both in good form; France's attacking trend slightly stronger. [ESPN/Goal results; Al Jazeera]

5. **Probable XI.** France: Maignan; Koundé, Saliba, Upamecano, T.Hernandez; Tchouaméni, Rabiot; Dembélé, Doué, Olise; Mbappé (c). Senegal (4-3-3): Mendy; Diatta, Koulibaly, Niakhaté, M.Diouf; P.Gueye, L.Camara, H.Diarra; I.Sarr, Jackson, Mané. [lineups.com; SI]

6. **Goalkeeper.** Maignan (AC Milan) is elite shot-stopping; Mendy (Al-Ahli) reliable, ex-Chelsea. Slight France edge — folded into the small France-defence form bump. [lineups.com]

7. **Injuries / absences.** No confirmed match-day injuries to listed starters on either side; Mbappé reported fit. Camavinga and Kolo Muani were left out of the France squad (selection, not late injury) — depth, not XI, impact. No material Senegal absences reported. Net: negligible, no injury adjustment applied. [ESPN squad]

8. **Rest / travel / venue.** Group opener — both rested, no congestion. MetLife Stadium (East Rutherford, NJ) is a **neutral** site for both nations; no host advantage. `home_advantage` left at 1.0/1.0. [CBS preview]

9. **Set pieces & discipline.** Both physically strong aerially (Upamecano/Saliba vs Koulibaly/Niakhaté); no standout edge or suspension flags found. No specific adjustment.

10. **Style / matchup.** France: high-quality possession + lethal transition through Mbappé/Dembélé pace. Senegal under Thiaw: organized, physical midfield (Gueye/Camara/Diarra), willing to sit and counter via Sarr/Jackson. A compact Senegal could suppress totals, but France's transition quality is the dominant feature — supports a modest France-attack bump, not a large one.

11. **Closing market (benchmark).** France -220 / Draw +350 / Senegal +600 (FanDuel, match day); total O/U 2.5 with Over -140. Encoded in `market_odds` for the engine's de-margined blend. [CBS preview; FOX Sports]

12. **Stage / motivation.** Group-stage opener; both fully motivated (France group favorite; Senegal needs points vs the seed). No motivation distortion.

## Rationale for parameters
Strength stays at the prior — it already encodes the long-run gap, and nothing in research justifies a ±40% strength move. Form/quality signals are encoded as small `adjustments`: France attack **+5%** (scoring 2+ in 9/10, Ballon d'Or-level front line firing) and France defence **−3% on the def value = stronger** (4 conceded in qualifying, elite CB unit + Maignan). Senegal held at prior on both sides — genuinely good but with a recent attacking blank and no specific edge to justify moving off the anchor. Neutral venue → no home advantage. Confidence: **alta** (Elo, market, lineups, and form all corroborated).