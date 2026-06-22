Data sufficient. Strong consensus across Elo, market, form. Synthesizing.

Key facts gathered:
- **Norway Elo ≈ 1922** (Jan 2026, Elo rank ~11; FIFA ~31). Iraq FIFA #57–58, ESPN combined-strength rank #56 → gap ≈ 250+ Elo.
- **Market (strong benchmark):** Norway ~-450/-475/-498, draw ~+525/+600, Iraq ~+1100/+1300. Kalshi: Iraq 7% / draw 13% / Norway 82%. O/U 2.5.
- **Norway form:** 8/8 WCQ, 37 scored / 5 conceded; Haaland 16 goals; Ødegaard. Elite attack + stingy defense.
- **Iraq:** Graham Arnold, defensive (0.77 GA/game over 13), beat Bolivia 2-1 playoff; weak attack (6 in last 5, shut out twice in Arab Cup).
- **Venue neutral** (Foxborough) → no host edge either side.

Prior already encodes Norway dominance and matches market — so move strengths zero, encode form via small adjustments only.

---

# Dossier — Iraq vs Norway (2026-06-16, Group I, Boston Stadium, neutral)

## TL;DR
Norway are heavy favorites by every independent benchmark — Elo gap (~250+ pts), squad value (Haaland €180m + Ødegaard €85m vs a sub-€50m Iraq squad), recent form (perfect 8/8 WCQ, +32 GD) and the closing market (Norway ~1.21 dec, Iraq ~13.3). The statistical prior already prices this correctly, so I keep both `strength` blocks at the prior and apply only small, evidence-backed `adjustments`: a slight Norway attack/defence boost (elite form) and a slight Iraq attack trim (toothless vs quality), partly offset by a modest credit to Iraq's organized low block under Arnold. Venue is neutral → no host advantage. Confidence: **alta**.

## Key findings
1. **Elo (top predictor).** Norway ≈ **1922**, Elo rank ~11 / FIFA ~31 [eloratings, ESPN]. Iraq FIFA **#57–58**, ESPN combined-strength **#56** [ESPN, FotMob] → Elo gap **≈ 250–270**, implying a Norway win prob in the high-70s/low-80s before any adjustment — matches the market.
2. **Squad value (PELE).** Norway: Haaland €180m, Ødegaard €85m, Sørloth, Nusa, Ryerson — squad value an order of magnitude above Iraq's [Statista/Transfermarkt, FIFA]. Huge talent gap.
3. **Form (recency-weighted).** Norway: **8/8** in WCQ, **37 scored / 5 conceded**, incl. 3-0 and 4-1 over Italy [UEFA, FIFA]. Iraq: 3/5 recently but **6 scored / 5 conceded**; lost 0-2 (Algeria) and 0-1 (Jordan) in Arab Cup — struggles to score vs structured sides [FotMob, WST].
4. **Key players / attack.** Haaland 16 goals in qualifying (2× any other European); Ødegaard the creator [FIFA, Squawka]. Iraq lack an equivalent attacking focal point.
5. **Defence / manager.** Iraq under **Graham Arnold** are organized and stingy (**0.77 GA/game** over 13 comp. matches) — their one real asset is a disciplined deep block [WST, beIN].
6. **Venue / rest.** Boston Stadium (Foxborough), **neutral** for both — Norway get no host edge; both opening-ish group fixture, rest equal.
7. **Stage / motivation.** Group stage; both ending long WC absences (Iraq since 1986, Norway since 1998). Norway chasing goal difference; Iraq in damage-limitation/counter mode.
8. **Market (benchmark).** Avg of bet365 / Sports Interaction / FanDuel ≈ Norway **1.21**, draw **6.75**, Iraq **13.3**; Kalshi 7/13/82. O/U **2.5**.

## Rationale for parameters
The prior already yields ≈ Norway 1.93 / Iraq 0.61 expected goals — consistent with an ~80% Norway market. I therefore **do not move `strength`** (kept = `prior_strength` for traceability). Adjustments encode the marginal evidence only, all well inside the 0.5–1.5 band:
- `away_atk = 1.05` — Norway's record-breaking scoring form (Haaland, 37 in 8).
- `away_def = 0.95` — only 5 conceded in qualifying (lower def value = stronger).
- `home_atk = 0.95` — Iraq's blunt attack vs an elite back line.
- `home_def = 0.97` — small credit to Arnold's disciplined low block.
- `home_advantage = 1.0/1.0` — neutral venue.

Resulting lambdas ≈ **Norway 1.97 / Iraq 0.55**, total ~2.5 (on the O/U line), Norway win prob in the low-80s — aligned with Elo, market, and Kalshi.