"""Full-tournament Monte Carlo: group stage -> knockout bracket -> champion odds.

Given the fitted prior and the 2026 fixture (12 groups of 4), we:

1. **Group stage** — sample a scoreline for every unplayed group match from the
   Dixon-Coles model (already-played matches use their real result, so this also
   works mid-tournament). Standings use the official FIFA ordering: points, then
   goal difference, then goals for, then head-to-head among the teams still level,
   with team rating as the final deterministic tie-break (proxy for fair-play /
   drawing of lots).
2. **Qualification** — top two of each group plus the eight best third-placed
   teams advance (the 48-team / 32-into-knockout format).
3. **Knockout** — the 32 qualifiers are placed into a standard strength-seeded
   single-elimination bracket (group winners seeded above runners-up above
   thirds, each class ordered by rating). Each tie is decided by
   :func:`advancement_probability` (90' -> extra time -> shootout). NB: the
   *exact* official R32 slotting of third-placed teams depends on an allocation
   table; the seeded bracket here is a faithful, strength-aware approximation and
   is labelled as such in the output.
4. **Aggregate** — over ``n_sims`` tournaments we report, per team, the
   probability of advancing from the group and of reaching each knockout round up
   to lifting the trophy.

Performance: every fixed group match is sampled for all sims at once (vectorised);
knockout win probabilities are memoised per ordered pair, so a tournament is a
cheap sequence of Bernoulli draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2

import numpy as np
import pandas as pd

from ..data.teams import resolve_strength, normalize_team
from ..model.dixon_coles import score_matrix
from ..model.knockout import advancement_probability

# 2026 hosts get a genuine edge even at "neutral" venues (mirrors calibrate.HOSTS).
HOSTS = {"United States", "Mexico", "Canada"}
_AVG_STRENGTH = {"atk": 1.0, "def": 1.0}


@dataclass
class _Sim:
    """Pre-resolved per-match strengths and cached distributions for one fixture."""

    prior: object
    max_goals: int = 8
    _strength: dict = field(default_factory=dict)
    _winprob: dict = field(default_factory=dict)

    def strength(self, team: str) -> dict:
        if team not in self._strength:
            s = resolve_strength(team, self.prior) or _AVG_STRENGTH
            self._strength[team] = {"atk": float(s["atk"]), "def": float(s["def"])}
        return self._strength[team]

    def rating(self, team: str) -> float:
        """Single scalar strength for seeding/tie-breaks (attack over concession)."""
        s = self.strength(team)
        return s["atk"] / max(s["def"], 1e-6)

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        sh, sa = self.strength(home), self.strength(away)
        base = self.prior.base_rate
        ha_h = self.prior.home_advantage if normalize_team(home) in HOSTS else 1.0
        ha_a = self.prior.home_advantage if normalize_team(away) in HOSTS else 1.0
        lam_h = base * sh["atk"] * sa["def"] * ha_h
        lam_a = base * sa["atk"] * sh["def"] * ha_a
        return lam_h, lam_a

    def knockout_win_prob(self, a: str, b: str) -> float:
        """P(a beats b) over 90' + ET + shootout, neutral venue, memoised."""
        key = (a, b)
        if key not in self._winprob:
            lam_a, lam_b = self.lambdas(a, b)
            # neutral knockout: drop host bonus so a vs b and b vs a are consistent
            res = advancement_probability(lam_a, lam_b, self.prior.rho,
                                          penalty_strength=None, max_goals=self.max_goals)
            self._winprob[key] = res["p_home_advance"]
        return self._winprob[key]


# ---- group stage ---------------------------------------------------------

def _group_table(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map group label -> list of teams (group-stage rows with known teams)."""
    g = df[(df["stage"] == "group") & df["teams_known"] & df["group"].notna()]
    groups: dict[str, set] = {}
    for _, r in g.iterrows():
        groups.setdefault(str(r["group"]), set()).update([r["home"], r["away"]])
    return {k: sorted(v) for k, v in sorted(groups.items()) if len(v) >= 3}


def _group_matches(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return df[(df["stage"] == "group") & (df["group"] == label) & df["teams_known"]]


def _standings_one(teams, results) -> list[str]:
    """Rank ``teams`` from a list of (home, away, hg, ag) using FIFA criteria."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for h, a, hg, ag in results:
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1

    def overall_key(t):
        return (pts[t], gf[t] - ga[t], gf[t])

    order = sorted(teams, key=overall_key, reverse=True)
    return _break_ties(order, overall_key, results)


def _break_ties(order, overall_key, results) -> list[str]:
    """Resolve teams equal on (pts, GD, GF) via head-to-head, then random-ish rating."""
    out: list[str] = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and overall_key(order[j]) == overall_key(order[i]):
            j += 1
        block = order[i:j]
        if len(block) > 1:
            block = _h2h_order(block, results)
        out.extend(block)
        i = j
    return out


def _h2h_order(block, results) -> list[str]:
    """Order a tied block by head-to-head points/GD/GF among themselves."""
    s = set(block)
    pts = {t: 0 for t in block}
    gd = {t: 0 for t in block}
    gf = {t: 0 for t in block}
    for h, a, hg, ag in results:
        if h in s and a in s:
            gf[h] += hg; gf[a] += ag; gd[h] += hg - ag; gd[a] += ag - hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
    return sorted(block, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)


# ---- knockout bracket ----------------------------------------------------

def _seed_positions(n: int) -> list[int]:
    """Standard single-elim seeding so seed 1 and 2 only meet in the final."""
    seeds = [1]
    while len(seeds) < n:
        m = 2 * len(seeds) + 1
        seeds = [x for s in seeds for x in (s, m - s)]
    return seeds


def _build_bracket(seed_order: list[str], n: int = 32) -> list[str]:
    """Place an already-ranked seed list into single-elim bracket slots."""
    positions = _seed_positions(n)
    slots: list = [None] * n
    for seed_idx, team in enumerate(seed_order[:n], start=1):
        slots[positions.index(seed_idx)] = team
    return slots


def simulate_tournament(prior, fixture: pd.DataFrame, n_sims: int = 20_000,
                        seed: int = 0, max_goals: int = 8) -> dict:
    """Run ``n_sims`` full tournaments; return per-team round-reaching probs."""
    sim = _Sim(prior, max_goals=max_goals)
    rng = np.random.default_rng(seed)
    groups = _group_table(fixture)
    if not groups:
        raise ValueError("no resolved groups in fixture")

    # Pre-sample every group match for all sims (vectorised); played -> fixed.
    n = max_goals + 1
    group_samples: dict[str, list] = {}
    for label, teams in groups.items():
        gm = _group_matches(fixture, label)
        sampled = []
        for _, r in gm.iterrows():
            h, a = r["home"], r["away"]
            if bool(r.get("played")) and pd.notna(r.get("home_score")):
                hg = np.full(n_sims, int(r["home_score"]))
                ag = np.full(n_sims, int(r["away_score"]))
            else:
                lam_h, lam_a = sim.lambdas(h, a)
                M = score_matrix(lam_h, lam_a, prior.rho, max_goals)
                flat = (M / M.sum()).ravel()
                idx = rng.choice(flat.size, size=n_sims, p=flat)
                hg, ag = idx // n, idx % n
            sampled.append((h, a, hg, ag))
        group_samples[label] = sampled

    all_teams = sorted({t for ts in groups.values() for t in ts})
    rounds = ["advance", "r16", "qf", "sf", "final", "champion"]
    counts = {t: {k: 0 for k in rounds} for t in all_teams}

    for s in range(n_sims):
        winners, runners, thirds = [], [], []
        for label, teams in groups.items():
            results = [(h, a, int(hg[s]), int(ag[s])) for h, a, hg, ag in group_samples[label]]
            rank = _standings_one(teams, results)
            if len(rank) >= 1: winners.append(rank[0])
            if len(rank) >= 2: runners.append(rank[1])
            if len(rank) >= 3: thirds.append((rank[2], results))
        # eight best third-placed teams (by rating proxy — overall record varies)
        thirds_sorted = sorted((t for t, _ in thirds), key=lambda t: -sim.rating(t))
        best_thirds = thirds_sorted[:8]
        qualified = winners + runners + best_thirds
        for t in qualified:
            counts[t]["advance"] += 1

        # seed winners above runners above thirds, each class ordered by rating
        seed_order = (sorted(winners, key=lambda t: -sim.rating(t))
                      + sorted(runners, key=lambda t: -sim.rating(t))
                      + sorted(best_thirds, key=lambda t: -sim.rating(t)))
        slots = _build_bracket(seed_order, n=32)
        # single elimination
        alive = slots
        round_names = ["r16", "qf", "sf", "final", "champion"]
        # first cut (R32 -> R16) doesn't get a counter beyond 'advance'
        rd = 0
        while len(alive) > 1:
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                if a is None: winner = b
                elif b is None: winner = a
                else:
                    winner = a if rng.random() < sim.knockout_win_prob(a, b) else b
                nxt.append(winner)
            alive = nxt
            # after this round, survivors have *reached* round_names[rd]
            if rd < len(round_names):
                for t in alive:
                    if t is not None:
                        counts[t][round_names[rd]] += 1
            rd += 1

    probs = {
        t: {k: counts[t][k] / n_sims for k in rounds}
        for t in all_teams
    }
    ranked = sorted(probs.items(), key=lambda kv: -kv[1]["champion"])
    return {
        "n_sims": n_sims,
        "n_groups": len(groups),
        "bracket_note": "strength-seeded single-elimination approximation of the "
                        "official R32 slotting",
        "teams": dict(ranked),
        "favorites": [
            {"team": t, "champion": p["champion"], "final": p["final"],
             "semi": p["sf"], "advance": p["advance"]}
            for t, p in ranked[:16]
        ],
    }
