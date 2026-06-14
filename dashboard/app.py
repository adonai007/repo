"""Interactive World Cup dashboard.

Run with: ``streamlit run dashboard/app.py``

Three things: (1) a scoreboard of the model's accuracy on graded matches, (2)
the predictions for a chosen date with heatmaps and model-vs-market, (3) a live
"what-if" panel — move the strength/rho/adjustment sliders and watch the score
matrix and 1X2 recompute in real time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from footy.schema import MatchParams
from footy.predict import predict
from footy.eval.metrics import summarize, reliability
from footy.worldcup.results_store import load_ledger

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"

st.set_page_config(page_title="footy — World Cup 2026", layout="wide")
st.title("⚽ footy — World Cup 2026 predictions")
st.caption("Dixon-Coles + Monte Carlo, calibrated by automated deep research. "
           "Informational model output — not betting advice.")


@st.cache_data
def _ledger() -> pd.DataFrame:
    return load_ledger(PRED)


def scoreboard(led: pd.DataFrame) -> None:
    graded = led[led["graded"] == True] if not led.empty else led  # noqa: E712
    st.subheader("📊 Scoreboard")
    if led.empty or not len(graded):
        st.info("No graded matches yet — the scoreboard fills in as results come in.")
        return
    fc = [((r.p_home, r.p_draw, r.p_away), r.actual_result) for r in graded.itertuples()]
    s = summarize(fc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graded", s["n"])
    c2.metric("Mean RPS", f"{s['rps']:.3f}")
    c3.metric("Log-loss", f"{s['log_loss']:.3f}")
    c4.metric("Hit rate", f"{s['accuracy']*100:.0f}%")

    rel = reliability(fc, n_bins=8)
    if rel:
        rdf = pd.DataFrame(rel).set_index("predicted")[["empirical"]]
        st.caption("Calibration (predicted vs empirical) — closer to diagonal is better")
        st.line_chart(rdf)


def predictions_view(led: pd.DataFrame) -> None:
    st.subheader("🗓️ Predictions by date")
    if led.empty:
        st.info("No predictions yet. Run `footy worldcup --date today`.")
        return
    dates = sorted(led["date"].dropna().unique())
    day = st.selectbox("Date", dates, index=len(dates) - 1)
    day_dir = PRED / str(day)
    sub = led[led["date"] == day]
    for r in sub.itertuples():
        with st.expander(f"{r.home} vs {r.away}  —  "
                         f"{r.p_home*100:.0f}% / {r.p_draw*100:.0f}% / {r.p_away*100:.0f}%  ({r.top_score})"):
            cols = st.columns([1, 1])
            mdir = next(day_dir.glob(f"*{_slugfrag(r.home)}*"), None) if day_dir.exists() else None
            if mdir and (mdir / "heatmap.png").exists():
                cols[0].image(str(mdir / "heatmap.png"))
            if mdir and (mdir / "prediction.json").exists():
                pred = json.loads((mdir / "prediction.json").read_text())
                cols[1].json(pred.get("market", {}) or {"market": "n/a"})
                if (mdir / "dossier.md").exists():
                    with cols[1].expander("Dossier"):
                        st.markdown((mdir / "dossier.md").read_text())


def _slugfrag(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name)[:6]


def whatif() -> None:
    st.subheader("🎛️ Live what-if")
    st.caption("Move the sliders to see the score matrix and 1X2 update instantly.")
    c = st.columns(4)
    home = c[0].text_input("Home", "Brazil")
    away = c[0].text_input("Away", "Morocco")
    base = c[0].slider("Base rate", 0.8, 2.0, 1.30, 0.01)
    rho = c[0].slider("rho (Dixon-Coles)", -0.20, 0.0, -0.10, 0.01)
    atk_h = c[1].slider("Home attack", 0.3, 3.0, 1.45, 0.01)
    def_h = c[1].slider("Home defence", 0.1, 3.0, 0.86, 0.01)
    atk_a = c[2].slider("Away attack", 0.3, 3.0, 0.88, 0.01)
    def_a = c[2].slider("Away defence", 0.1, 3.0, 0.63, 0.01)

    params = {
        "match": {"home": home, "away": away, "stage": "group", "neutral": True},
        "base_rate": base, "rho": rho,
        "strength": {"home": {"atk": atk_h, "def": def_h},
                     "away": {"atk": atk_a, "def": def_a}},
        "adjustments": {}, "home_advantage": {},
    }
    out = predict(MatchParams.parse(params), run_mc=False)
    a = out["analytic"]
    c[3].metric(f"{home} win", f"{a['p_home']*100:.1f}%")
    c[3].metric("Draw", f"{a['p_draw']*100:.1f}%")
    c[3].metric(f"{away} win", f"{a['p_away']*100:.1f}%")
    c[3].metric("Most likely", a["top_scores"][0]["score"])

    M = np.array(out["score_matrix"])[:6, :6] * 100
    hm = pd.DataFrame(M, index=[f"{home} {i}" for i in range(6)],
                      columns=[f"{away} {j}" for j in range(6)])
    st.caption(f"Score matrix (%) — λ {out['lambda']['home']:.2f} / {out['lambda']['away']:.2f}")
    st.dataframe(hm.style.background_gradient(cmap="OrRd").format("{:.1f}"))


led = _ledger()
scoreboard(led)
st.divider()
predictions_view(led)
st.divider()
whatif()
