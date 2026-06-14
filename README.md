# ⚽ footy — automated football match-prediction framework

Predict any football match the way a quant desk would: a **deep-research step**
gathers the inputs, and a **transparent statistical engine** (bivariate Poisson with
the **Dixon-Coles** low-score correction + Monte Carlo) turns them into calibrated
probabilities. Built around the **2026 World Cup**, but the engine is general.

This generalises a hand-built model that already worked: for Brazil vs Morocco it
called the most likely scoreline (1-1) and the low-scoring profile correctly. `footy`
turns that one-off into a repeatable, automated, auditable pipeline.

> ℹ️ **Informational model output — not betting advice.** No guarantees of accuracy.

## How it works

Two layers, coupled by a JSON contract (`params.json`):

```
            deep research (LLM)                  numeric engine (deterministic)
fixture ─► dossier.md + params.json ─► Dixon-Coles matrix ─► 1X2 / scores / markets
            (strengths, rho, injuries,            + Monte Carlo + market comparison
             anchored to a statistical prior,        + knockout advancement
             every number cited)                     + heatmap + report
```

- **Research layer** (`footy/research`): runs Claude Code headless (`claude -p`) with the
  `deep-research` skill to produce, per match, a sourced dossier and a `params.json`
  validated against `footy/schema.py`. The LLM only *sets parameters* — it never
  computes probabilities — and anchors team strengths to a statistical prior.
- **Numeric layer** (`footy/model`, `footy/simulate`, `footy/odds`): the transparent,
  fully-tested core. Faithful port of the original script, generalised.

### The model

For each side the expected goals (lambda) are

```
lambda_home = base · atk_home · adj_home_atk · def_away · adj_away_def · ha_home
lambda_away = base · atk_away · adj_away_atk · def_home · adj_home_def · ha_away
```

(`atk`/`def` are strength indices where 1.0 = average elite national side; `def < 1`
means an elite defence that suppresses the opponent's lambda). The joint scoreline
distribution is independent Poisson with the Dixon-Coles `tau` correction on the four
low-score cells (0-0, 1-0, 0-1, 1-1), governed by `rho < 0` — which fixes the well-known
under-counting of draws. Markets (1X2, BTTS, over/under, top scorelines) are read off
the matrix; Monte Carlo sampling validates the analytic numbers. Knockout ties add an
extra-time + penalty-shootout model to produce an **advancement** probability.

Why Dixon-Coles: it remains the gold standard for low-scoring football, and the value of
the framework is **calibration and transparency** (reported alongside the de-margined
market), not beating the market.

## Install

```bash
pip install -e .            # core engine + CLI
pip install -e ".[data]"    # + soccerdata (fixture / historical ingestion)
pip install -e ".[dashboard]"  # + streamlit dashboard
pip install -e ".[dev]"     # + pytest
```

## Quickstart

```bash
# Predict a single match from a params.json and write report.md + heatmap.png
footy predict --params examples/bra_mar/params.json --out out/bra_mar

# Fetch the World Cup fixture
footy fixture --season 2026
```

Reproduce the validated Brazil vs Morocco result:

```
$ footy predict --params examples/bra_mar/params.json
Brazil vs Morocco  (group)
  lambda: 1.15 - 0.98   (total 2.14, rho -0.1)
  1X2:  Brazil 38.3%  |  Draw 32.2%  |  Morocco 29.6%
  most likely scoreline: 1-1 (14.7%)
```

## The `params.json` contract

Minimum required (reproduces the original engine exactly): `match`, `base_rate`,
`strength`, `rho`, `adjustments`, `home_advantage`. Everything else is optional with
sane defaults — `market_odds`/`blend_weight`, `knockout.penalty_strength`, `scenarios`
(ensemble), `evidence`/`prior_strength` (audit trail), `citations`, `rationale`. See
`examples/bra_mar/params.json` and `footy/schema.py`.

## Tests

```bash
pytest      # golden Bra-Mar reproduction + core property tests
```

## Status

Core engine, reporting, CLI `predict`, and the params contract are implemented and
tested. Data ingestion (fixture + historical prior), the research engine, the daily
World Cup pipeline, evaluation/backtest, dashboard, and the GitHub Actions cron are
being built out — see `plan` for the roadmap.
