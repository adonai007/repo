"""``footy`` command-line interface.

Commands are added incrementally as each layer lands. ``predict`` is the core
vertical slice: params.json -> probabilities + Markdown report + heatmap PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Football match-prediction framework (World Cup 2026).")


def _load_params(params_path: Path):
    from .schema import MatchParams

    data = json.loads(Path(params_path).read_text())
    return MatchParams.parse(data)


@app.command()
def predict(
    params: Path = typer.Option(..., "--params", "-p", help="Path to a params.json file."),
    out: Path = typer.Option(None, "--out", "-o", help="Directory to write report.md + heatmap.png + prediction.json."),
    blend: float = typer.Option(None, "--blend", help="Override blend weight toward the market (0..1)."),
    n_sims: int = typer.Option(200_000, "--sims", help="Monte Carlo sample size."),
    no_mc: bool = typer.Option(False, "--no-mc", help="Skip Monte Carlo validation."),
):
    """Predict a single match from a params.json file."""
    from .predict import predict as run_predict
    from .report import match_report, score_heatmap
    import numpy as np

    mp = _load_params(params)
    if blend is not None:
        mp.blend_weight = float(blend)

    result = run_predict(mp, n_sims=n_sims, run_mc=not no_mc)
    a = result["analytic"]
    m = result["match"]

    typer.echo(f"\n{m['home']} vs {m['away']}  ({m.get('stage', 'group')})")
    typer.echo(f"  lambda: {result['lambda']['home']:.2f} - {result['lambda']['away']:.2f}"
               f"   (total {result['lambda']['total']:.2f}, rho {result['rho']})")
    typer.echo(f"  1X2:  {m['home']} {a['p_home']*100:.1f}%  |  Draw {a['p_draw']*100:.1f}%  "
               f"|  {m['away']} {a['p_away']*100:.1f}%")
    top = result["analytic"]["top_scores"][0]
    typer.echo(f"  most likely scoreline: {top['score']} ({top['p']*100:.1f}%)")
    if "knockout" in result:
        ko = result["knockout"]
        typer.echo(f"  advances: {m['home']} {ko['p_home_advance']*100:.1f}%  |  "
                   f"{m['away']} {ko['p_away_advance']*100:.1f}%")

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        M = np.array(result["score_matrix"])
        score_heatmap(M, m["home"], m["away"], out / "heatmap.png")
        (out / "report.md").write_text(match_report(result, heatmap_rel="heatmap.png"))
        # don't store the full matrix in the headline prediction.json
        slim = {k: v for k, v in result.items() if k != "score_matrix"}
        (out / "prediction.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False))
        typer.echo(f"\n  wrote: {out}/report.md, heatmap.png, prediction.json")


@app.command()
def fixture(
    season: str = typer.Option("2026", "--season", help="World Cup season."),
    out: Path = typer.Option(None, "--out", help="Optional CSV output path."),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass snapshot cache."),
):
    """Fetch the World Cup match schedule (104 fixtures)."""
    from .data.fixture import load_fixture

    df = load_fixture(season=season, refresh=refresh)
    typer.echo(df.to_string(index=False))
    typer.echo(f"\n{len(df)} matches")
    if out is not None:
        df.to_csv(out, index=False)
        typer.echo(f"wrote {out}")


@app.command()
def fit(
    asof: str = typer.Option(None, "--asof", help="Cut-off date YYYY-MM-DD (default today)."),
    window: int = typer.Option(8, "--window", help="Years of history to fit on."),
    xi: float = typer.Option(0.0018, "--xi", help="Time-decay rate per day."),
):
    """Fit the data-driven attack/defence prior from international results."""
    from .ratings.prior_fit import fit_prior
    from .ratings.store import save_prior

    prior = fit_prior(asof=asof, window_years=window, xi=xi)
    path = save_prior(prior)
    typer.echo(f"fitted {len(prior.teams)} teams on {prior.n_matches_fit} matches "
               f"(as-of {prior.asof})")
    typer.echo(f"  base_rate={prior.base_rate:.3f}  home_adv={prior.home_advantage:.3f}  "
               f"rho={prior.rho:.4f}")
    typer.echo(f"  saved -> {path}")


@app.command()
def worldcup(
    date: str = typer.Option("today", "--date", help="Match date YYYY-MM-DD or 'today'."),
    research: bool = typer.Option(False, "--research/--no-research",
                                  help="Run deep research via claude -p (default: prior baseline)."),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass research/fixture cache."),
    sims: int = typer.Option(50_000, "--sims", help="Monte Carlo sims per match (0 to skip)."),
):
    """Predict every predictable match on a date; persist artifacts + ledger + reports."""
    from .worldcup.pipeline import run_day
    from .research.engine import ClaudeCliRunner
    from datetime import date as _date

    day = _date.today().isoformat() if date == "today" else date
    runner = ClaudeCliRunner() if research else None
    summary = run_day(day, runner=runner, refresh=refresh, n_sims=sims)
    typer.echo(f"date {summary['date']}: predicted {summary['predicted']} match(es) "
               f"-> {summary['day_dir']}")
    for e in summary["errors"]:
        typer.echo(f"  skipped {e['match']}: {e['error']}")


@app.command()
def research(
    home: str = typer.Option(..., "--home"),
    away: str = typer.Option(..., "--away"),
    date: str = typer.Option(None, "--date"),
    stage: str = typer.Option("group", "--stage"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Prior baseline only (no claude)."),
    out: Path = typer.Option(None, "--out", help="Directory to cache dossier + params."),
):
    """Generate a params.json (and dossier) for one match via deep research."""
    from .ratings.store import latest_prior
    from .research.engine import research_match, ClaudeCliRunner

    prior = latest_prior()
    if prior is None:
        raise typer.Exit("no prior found; run `footy fit` first")
    match = {"home": home, "away": away, "date": date, "stage": stage}
    runner = None if no_llm else ClaudeCliRunner()
    res = research_match(match, prior, runner=runner, cache_dir=out)
    typer.echo(f"engine: {res['engine']}")
    typer.echo(json.dumps(res["params"], ensure_ascii=False, indent=2))


@app.command()
def backtest(
    tournament: str = typer.Option("FIFA World Cup", "--tournament"),
    year: int = typer.Option(2022, "--year"),
    window: int = typer.Option(8, "--window"),
):
    """Backtest the model on a past tournament vs trivial baselines (RPS)."""
    from .eval.backtest import backtest_tournament

    r = backtest_tournament(tournament=tournament, year=year, window_years=window)
    m, u, b = r["model"], r["baseline_uniform"], r["baseline_baserate"]
    typer.echo(f"\n{r['tournament']}  (n={r['n']}, prior as-of {r['prior_asof']})")
    typer.echo(f"  {'':10}{'RPS':>8}{'logloss':>10}{'acc':>8}")
    typer.echo(f"  {'model':10}{m['rps']:>8.3f}{m['log_loss']:>10.3f}{m['accuracy']:>8.2f}")
    typer.echo(f"  {'uniform':10}{u['rps']:>8.3f}{u['log_loss']:>10.3f}{u['accuracy']:>8.2f}")
    typer.echo(f"  {'base-rate':10}{b['rps']:>8.3f}{b['log_loss']:>10.3f}{b['accuracy']:>8.2f}")
    typer.echo(f"  beats uniform: {r['beats_uniform']}  ·  beats base-rate: {r['beats_baserate']}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
