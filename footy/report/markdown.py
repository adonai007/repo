"""Render a prediction dict (from ``footy.predict.predict``) as a Markdown report."""

from __future__ import annotations


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def match_report(result: dict, heatmap_rel: str | None = None) -> str:
    m = result["match"]
    a = result["analytic"]
    lam = result["lambda"]
    lines: list[str] = []

    title = f"{m['home']} vs {m['away']}"
    if m.get("date"):
        title += f" — {m['date']}"
    lines += [f"# {title}", ""]
    lines += [f"*Stage:* {m.get('stage', 'group')}  ·  *Engine:* Poisson bivariate + Dixon-Coles + Monte Carlo", ""]

    lines += ["## Expected goals (model)", ""]
    lines += [f"- **{m['home']}**: {lam['home']:.2f}", f"- **{m['away']}**: {lam['away']:.2f}",
              f"- **Total**: {lam['total']:.2f}  ·  rho = {result['rho']}", ""]

    lines += ["## 1X2 (match result)", "", "| Outcome | Model |"]
    lines += ["|---|---|"]
    lines += [f"| {m['home']} win | **{_pct(a['p_home'])}** |",
              f"| Draw | **{_pct(a['p_draw'])}** |",
              f"| {m['away']} win | **{_pct(a['p_away'])}** |", ""]

    if "monte_carlo" in result:
        mc = result["monte_carlo"]
        lines += [f"*Monte Carlo ({mc['n_sims']:,} sims):* "
                  f"{_pct(mc['p_home'])} / {_pct(mc['p_draw'])} / {_pct(mc['p_away'])} "
                  f"· avg goals {mc['mean_goals']:.2f}", ""]

    lines += ["## Most likely scorelines", ""]
    for s in a["top_scores"][:6]:
        lines.append(f"- {s['score']}: {_pct(s['p'])}")
    lines.append("")

    lines += ["## Derived markets", "",
              f"- Both teams to score: {_pct(a['p_btts'])}",
              f"- Over 2.5 goals: {_pct(a['p_over25'])}  ·  Under 2.5: {_pct(a['p_under25'])}",
              f"- Double chance 1X: {_pct(a['p_1x'])}  ·  X2: {_pct(a['p_x2'])}", ""]

    if "market" in result and "implied" in result["market"]:
        mk = result["market"]
        imp, bl = mk["implied"], mk["blended"]
        lines += ["## Model vs market (de-margined)", "",
                  "| Outcome | Model | Market | Blend |", "|---|---|---|---|",
                  f"| {m['home']} | {_pct(a['p_home'])} | {_pct(imp['home'])} | {_pct(bl['home'])} |",
                  f"| Draw | {_pct(a['p_draw'])} | {_pct(imp['draw'])} | {_pct(bl['draw'])} |",
                  f"| {m['away']} | {_pct(a['p_away'])} | {_pct(imp['away'])} | {_pct(bl['away'])} |",
                  "", f"*Overround: {mk['overround']:.3f}  ·  blend weight: {mk['blend_weight']}*", ""]

    if "knockout" in result:
        ko = result["knockout"]
        lines += ["## Knockout advancement (incl. extra time + penalties)", "",
                  f"- **{m['home']} advances**: {_pct(ko['p_home_advance'])}",
                  f"- **{m['away']} advances**: {_pct(ko['p_away_advance'])}", ""]

    if "ensemble" in result and result["ensemble"]:
        rng = result["ensemble"]["range"]
        lines += ["## Scenario range (ensemble)", "",
                  f"- {m['home']} win: {_pct(rng['home'][0])} – {_pct(rng['home'][1])}",
                  f"- Draw: {_pct(rng['draw'][0])} – {_pct(rng['draw'][1])}",
                  f"- {m['away']} win: {_pct(rng['away'][0])} – {_pct(rng['away'][1])}", ""]

    if heatmap_rel:
        lines += ["## Scoreline heatmap", "", f"![scoreline heatmap]({heatmap_rel})", ""]

    if result.get("citations"):
        lines += ["## Sources", ""] + [f"- {c}" for c in result["citations"]] + [""]

    lines += ["---", "*Informational model output — not betting advice.*"]
    return "\n".join(lines)
