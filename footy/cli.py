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


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
