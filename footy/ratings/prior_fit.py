"""Fit a Dixon-Coles attack/defence prior on historical international results.

This is the *data-driven baseline* the research layer then adjusts. We fit a
time-weighted Maher/Poisson model (attack_i, defence_i, home advantage, global
intercept) by maximum likelihood with analytic gradients, then estimate a global
Dixon-Coles ``rho``. Ratings map directly to the engine's strength indices:

    base = exp(mu)              # baseline goals
    atk_i = exp(attack_i)       # >1 = strong attack   (mean ~1)
    def_i = exp(-defence_i)     # <1 = strong defence  (mean ~1)
    home_advantage = exp(h)

so that ``lambda_home = base * atk_home * def_away`` reproduces the Poisson model.

Recency weighting (``xi`` per day) and a competitive>friendly tournament weight
follow Dixon-Coles (1997). An L2 penalty shrinks sparse teams (debutants) toward
average, which is the intended minnow fallback.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from ..data.fixture import SNAPSHOT_DIR

MARTJ42_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Tournament importance weights (competitive matches inform more than friendlies).
_COMPETITIVE = ("World Cup", "Euro", "Copa América", "Copa America", "Nations League",
                "African Cup", "Gold Cup", "AFC Asian Cup", "Confederations")
FRIENDLY_WEIGHT = 0.6


@dataclass
class PriorRatings:
    """Fitted ratings + global parameters, as-of a cut-off date."""

    asof: str
    base_rate: float
    home_advantage: float
    rho: float
    teams: dict[str, dict[str, float]]  # team -> {atk, def, n_matches}
    n_matches_fit: int = 0
    window_years: int = 8
    xi: float = 0.0018
    source: str = "martj42"

    def strength(self, team: str) -> dict[str, float] | None:
        return self.teams.get(team)

    def to_dict(self) -> dict:
        return {
            "asof": self.asof, "base_rate": self.base_rate,
            "home_advantage": self.home_advantage, "rho": self.rho,
            "window_years": self.window_years, "xi": self.xi,
            "source": self.source, "n_matches_fit": self.n_matches_fit,
            "teams": self.teams,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PriorRatings":
        return cls(
            asof=d["asof"], base_rate=d["base_rate"], home_advantage=d["home_advantage"],
            rho=d["rho"], teams=d["teams"], n_matches_fit=d.get("n_matches_fit", 0),
            window_years=d.get("window_years", 8), xi=d.get("xi", 0.0018),
            source=d.get("source", "martj42"),
        )


def _tournament_weight(name: str) -> float:
    if not isinstance(name, str) or name.lower() == "friendly":
        return FRIENDLY_WEIGHT
    return 1.0 if any(tok in name for tok in _COMPETITIVE) else 0.85


def load_results(asof: str | date | None = None, cache_dir: Path | str = SNAPSHOT_DIR) -> pd.DataFrame:
    """Load martj42 results (cached snapshot), keeping only played matches up to asof."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snap = cache_dir / "martj42-results.csv"
    try:
        req = urllib.request.Request(MARTJ42_URL, headers={"User-Agent": "footy/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8")
        snap.write_text(text)
        df = pd.read_csv(StringIO(text))
    except Exception:
        if not snap.exists():
            raise
        df = pd.read_csv(snap)

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if asof is not None:
        asof_ts = pd.to_datetime(asof)
        df = df[df["date"] <= asof_ts]
    return df.reset_index(drop=True)


def fit_prior(
    asof: str | date | None = None,
    window_years: int = 8,
    xi: float = 0.0018,
    reg: float = 0.15,
    cache_dir: Path | str = SNAPSHOT_DIR,
    df: pd.DataFrame | None = None,
) -> PriorRatings:
    """Fit the attack/defence prior as-of ``asof`` (default: today)."""
    asof = pd.to_datetime(asof) if asof is not None else pd.Timestamp(date.today())
    if df is None:
        df = load_results(asof=asof, cache_dir=cache_dir)
    cutoff = asof - pd.Timedelta(days=int(window_years * 365.25))
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("not enough matches to fit a prior")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    h = df["home_team"].map(tidx).to_numpy()
    a = df["away_team"].map(tidx).to_numpy()
    ys_h = df["home_score"].to_numpy(dtype=float)
    ys_a = df["away_score"].to_numpy(dtype=float)
    neutral = df["neutral"].astype(bool).to_numpy()
    days = (asof - df["date"]).dt.days.to_numpy(dtype=float)
    w = np.exp(-xi * days) * df["tournament"].map(_tournament_weight).to_numpy()

    n_played = np.zeros(T)
    np.add.at(n_played, h, 1)
    np.add.at(n_played, a, 1)

    # Parameter vector: [attack(T), defence(T), mu, home_adv]
    def unpack(theta):
        return theta[:T], theta[T:2 * T], theta[2 * T], theta[2 * T + 1]

    def nll_and_grad(theta):
        atk, dfn, mu, hadv = unpack(theta)
        log_lh = mu + atk[h] - dfn[a] + hadv * (~neutral)
        log_la = mu + atk[a] - dfn[h]
        lh, la = np.exp(log_lh), np.exp(log_la)
        nll = np.sum(w * (lh - ys_h * log_lh)) + np.sum(w * (la - ys_a * log_la))
        nll += reg * (np.sum(atk ** 2) + np.sum(dfn ** 2))

        rh = w * (lh - ys_h)  # d/dlog_lam for home obs
        ra = w * (la - ys_a)
        g_atk = np.zeros(T); g_dfn = np.zeros(T)
        np.add.at(g_atk, h, rh); np.add.at(g_atk, a, ra)
        np.add.at(g_dfn, a, -rh); np.add.at(g_dfn, h, -ra)
        g_atk += 2 * reg * atk; g_dfn += 2 * reg * dfn
        g_mu = np.sum(rh) + np.sum(ra)
        g_hadv = np.sum(rh * (~neutral))
        return nll, np.concatenate([g_atk, g_dfn, [g_mu, g_hadv]])

    theta0 = np.zeros(2 * T + 2)
    theta0[2 * T] = np.log(max(df[["home_score", "away_score"]].mean().mean(), 0.5))
    res = minimize(nll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    atk, dfn, mu, hadv = unpack(res.x)

    # Identify by centering attack & defence on ESTABLISHED teams (not the 278
    # including tiny sides), so indices follow the "~1.0 = solid national team"
    # convention and stay within the contract guard ranges. Means fold into mu.
    ref = n_played >= 30
    if ref.sum() < 10:
        ref = np.ones_like(n_played, dtype=bool)
    abar, dbar = atk[ref].mean(), dfn[ref].mean()
    atk -= abar; dfn -= dbar
    mu += abar - dbar

    rho = _fit_rho(df, tidx, atk, dfn, mu, hadv, h, a, ys_h, ys_a, neutral, w)

    team_ratings = {
        t: {"atk": float(np.exp(atk[i])), "def": float(np.exp(-dfn[i])),
            "n_matches": int(n_played[i])}
        for t, i in tidx.items()
    }
    return PriorRatings(
        asof=str(pd.Timestamp(asof).date()), base_rate=float(np.exp(mu)),
        home_advantage=float(np.exp(hadv)), rho=float(rho), teams=team_ratings,
        n_matches_fit=len(df), window_years=window_years, xi=xi,
    )


def _fit_rho(df, tidx, atk, dfn, mu, hadv, h, a, ys_h, ys_a, neutral, w) -> float:
    """Estimate a single global Dixon-Coles rho via profile likelihood."""
    lh = np.exp(mu + atk[h] - dfn[a] + hadv * (~neutral))
    la = np.exp(mu + atk[a] - dfn[h])
    yh = ys_h.astype(int); ya = ys_a.astype(int)
    # only the four low-score cells are affected
    m00 = (yh == 0) & (ya == 0)
    m01 = (yh == 0) & (ya == 1)
    m10 = (yh == 1) & (ya == 0)
    m11 = (yh == 1) & (ya == 1)

    def neg_ll(rho):
        tau = np.ones_like(lh)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-6, None)
        return -np.sum(w * np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.0), method="bounded")
    return float(res.x)
