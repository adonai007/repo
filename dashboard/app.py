"""Interactive World Cup dashboard.

Run with: ``streamlit run dashboard/app.py``

Sections: (1) a freshness bar (UTC clock + ledger mtime + refresh), (2) an
"upcoming" panel listing the next predictable kickoffs in Bolivia + UTC time,
(3) a scoreboard of the model's accuracy on graded matches, (4) the predictions
for a chosen date with kickoff lead-time badges, heatmaps, model-vs-market,
per-match traceability (audited params, citations, rationale, report) and a
predicted-vs-actual block for graded matches, and (5) a live "what-if" panel —
move the strength/rho/adjustment sliders and watch the score matrix and 1X2
recompute in real time.

Encoding note: the package has a Windows cp1252 bug, so every file read here is
explicit ``encoding="utf-8"`` and guarded with ``Path.exists()``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
LEDGER_PATH = PRED / "ledger.parquet"
BACKTEST_PATH = PRED / "backtest_report.json"
PIPELINE_RUN_PATH = ROOT / "pipeline_runs" / "latest.json"
RATINGS = ROOT / "ratings"

# Bolivia has no DST and sits at UTC-4 year-round.
BOLIVIA_TZ = timezone(timedelta(hours=-4))
# A prediction made this far ahead of kickoff counts as "comfortably ahead".
LEAD_OK_MIN = 60

st.set_page_config(page_title="footy — World Cup 2026", layout="wide")
st.title("⚽ footy — World Cup 2026 predictions")
st.caption("Dixon-Coles + Monte Carlo, calibrated by automated deep research. "
           "Informational model output — not betting advice.")


# --- cached loaders -----------------------------------------------------------
# ttl=30 so new predictions written by the scheduler show up quickly (the old
# 5-min ttl made the ledger stale). The 🔄 button below force-clears the cache.
@st.cache_data(ttl=30)
def _ledger() -> pd.DataFrame:
    return load_ledger(PRED)


@st.cache_data(ttl=60)
def _fixture() -> pd.DataFrame:
    """World-Cup fixture with tz-aware UTC kickoffs. Empty frame on failure."""
    try:
        from footy.data.fixture import load_fixture
        return load_fixture()
    except Exception:  # offline + no snapshot -> degrade, never crash the page
        return pd.DataFrame(
            columns=["match_number", "datetime_utc", "date", "home", "away",
                     "played", "teams_known", "stage", "group"]
        )


# --- small utilities ----------------------------------------------------------
def _read_json(path: Path) -> dict:
    """Read a JSON file as utf-8; return {} if missing or unreadable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    """Read a text file as utf-8; return '' if missing or unreadable."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


@st.cache_data(ttl=30)
def _latest_run() -> dict:
    return _read_json(PIPELINE_RUN_PATH)


@st.cache_data(ttl=60)
def _latest_prior_payload() -> dict:
    snaps = sorted(RATINGS.glob("*.json"))
    return _read_json(snaps[-1]) if snaps else {}


@st.cache_data(ttl=60)
def _prediction_strengths() -> dict:
    strengths: dict[str, dict[str, float]] = {}
    for path in sorted(PRED.glob("*/*/params.json")):
        params = _read_json(path)
        match = params.get("match") or {}
        strength = params.get("strength") or {}
        for side in ("home", "away"):
            name = match.get(side)
            side_strength = strength.get(side) or {}
            atk = side_strength.get("atk")
            deff = side_strength.get("def")
            if name and isinstance(atk, (int, float)) and isinstance(deff, (int, float)):
                strengths[str(name)] = {"atk": float(atk), "def": float(deff)}
    return strengths


def _slugfrag(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name))[:6]


def _match_dir(day_dir: Path, home: str) -> Path | None:
    """Locate the per-match artifact folder via the existing slug-glob approach."""
    if not day_dir.exists():
        return None
    return next(day_dir.glob(f"*{_slugfrag(home)}*"), None)


def _parse_dt(value) -> datetime | None:
    """Parse an ISO timestamp (any tz, or naive=UTC) into a tz-aware UTC datetime."""
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _pipeline_run_caption(run: dict) -> str:
    if not run:
        return "Pipeline: _sin registro_"
    status = str(run.get("status") or "?").upper()
    finished = _parse_dt(run.get("finished_at"))
    finished_label = finished.strftime("%Y-%m-%d %H:%M UTC") if finished else "sin hora"
    predictions = run.get("predictions") or {}
    fixture = run.get("fixture") or {}
    selected = fixture.get("selected_for_prediction", "?")
    predicted = predictions.get("predicted", "?")
    errors = len(predictions.get("errors") or [])
    return (
        f"Pipeline: **{status}** | {finished_label} | "
        f"predicho {predicted}/{selected} | errores {errors}"
    )


def _fmt_hm(total_minutes: float) -> str:
    total = int(abs(total_minutes))
    return f"{total // 60}h{total % 60:02d}m"


def _kickoff_lookup(fix: pd.DataFrame) -> dict[int, datetime]:
    """match_number -> tz-aware UTC kickoff datetime."""
    out: dict[int, datetime] = {}
    if fix.empty or "datetime_utc" not in fix.columns:
        return out
    for r in fix.itertuples():
        mn = getattr(r, "match_number", None)
        dt = _parse_dt(getattr(r, "datetime_utc", None))
        if mn is not None and not pd.isna(mn) and dt is not None:
            out[int(mn)] = dt
    return out


# --- freshness bar ------------------------------------------------------------
def freshness_bar() -> None:
    now_utc = datetime.now(timezone.utc)
    latest_run = _latest_run()
    c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
    c1.caption(f"🕒 Ahora (UTC): **{now_utc:%Y-%m-%d %H:%M:%S}**")
    if LEDGER_PATH.exists():
        mtime = datetime.fromtimestamp(LEDGER_PATH.stat().st_mtime, tz=timezone.utc)
        age_min = (now_utc - mtime).total_seconds() / 60.0
        c2.caption(f"📒 ledger.parquet: **{mtime:%Y-%m-%d %H:%M:%S} UTC** "
                   f"(hace {age_min:.0f} min)")
    else:
        c2.caption("📒 ledger.parquet: _no encontrado_")
    c3.caption(_pipeline_run_caption(latest_run))
    if c4.button("🔄 Actualizar"):
        st.cache_data.clear()
        st.rerun()


# --- upcoming panel -----------------------------------------------------------
def upcoming_panel(fix: pd.DataFrame, n: int = 8) -> None:
    st.subheader("⏭️ Próximos partidos")
    if fix.empty or "datetime_utc" not in fix.columns:
        st.info("Fixture no disponible (sin red ni snapshot).")
        return

    now_utc = datetime.now(timezone.utc)
    df = fix.copy()
    df = df[df.get("teams_known", False) == True]            # noqa: E712
    if "played" in df.columns:
        df = df[df["played"] == False]                        # noqa: E712
    df = df[df["datetime_utc"].notna()]
    # Keep only kickoffs that are still in the future, soonest first.
    df = df[pd.to_datetime(df["datetime_utc"], utc=True) >= now_utc]
    df = df.sort_values("datetime_utc").head(n)

    if df.empty:
        st.info("No hay próximos partidos predecibles.")
        return

    rows = []
    for r in df.itertuples():
        ko = _parse_dt(getattr(r, "datetime_utc", None))
        if ko is None:
            continue
        ko_bo = ko.astimezone(BOLIVIA_TZ)
        mins = (ko - now_utc).total_seconds() / 60.0
        day = getattr(r, "date", None)
        day_str = str(day) if day is not None else ko.date().isoformat()
        mdir = _match_dir(PRED / day_str, getattr(r, "home", ""))
        has_pred = bool(mdir and (mdir / "prediction.json").exists())
        rows.append({
            "Nº": int(getattr(r, "match_number")),
            "Partido": f"{getattr(r, 'home', '')} vs {getattr(r, 'away', '')}",
            "Bolivia (UTC-4)": ko_bo.strftime("%a %d %b %H:%M"),
            "UTC": ko.strftime("%a %d %b %H:%M"),
            "Faltan": _fmt_hm(mins),
            "Predicción": "✅ predicho" if has_pred else "⏳ pendiente",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# --- scoreboard ---------------------------------------------------------------
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


# --- backtest / evaluation ----------------------------------------------------
def _fmt_pct(value) -> str:
    return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else "—"


def _fmt_rps(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def backtest_view() -> None:
    st.subheader("📉 Backtest / Evaluación")

    report = _read_json(BACKTEST_PATH)
    if not report:
        st.info("Aún no se generó el backtest — corré ops/run_backtest.py")
        return

    metrics = report.get("metrics") or {}
    model = metrics.get("model") or {}
    uniform = metrics.get("uniform") or {}
    baserate = metrics.get("baserate") or {}
    n_graded = report.get("n_graded")

    # --- HONEST framing: in-sample (live) vs out-of-sample (2022) -------------
    live_col, hist_col = st.columns(2)

    # In-sample (live) ---------------------------------------------------------
    with live_col:
        n_lbl = n_graded if n_graded is not None else model.get("n", "—")
        st.markdown(f"**🔴 En vivo (in-sample, n={n_lbl})**")
        m_rps = model.get("rps")
        u_rps = uniform.get("rps")
        b_rps = baserate.get("rps")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy modelo", _fmt_pct(model.get("accuracy")))
        m2.metric("RPS modelo", _fmt_rps(m_rps))
        m3.metric("RPS uniforme", _fmt_rps(u_rps))
        m4.metric("RPS base-rate", _fmt_rps(b_rps))

        # Compute whether the model beats the baselines (lower RPS is better).
        beats_u = report.get("metrics", {}).get("model", {}).get("beats_uniform_rps")
        beats_b = report.get("metrics", {}).get("model", {}).get("beats_baserate_rps")
        if beats_u is None and isinstance(m_rps, (int, float)) and isinstance(u_rps, (int, float)):
            beats_u = m_rps < u_rps
        if beats_b is None and isinstance(m_rps, (int, float)) and isinstance(b_rps, (int, float)):
            beats_b = m_rps < b_rps
        if beats_u and beats_b:
            verdict = "el modelo **supera** a ambos baselines"
        elif beats_u or beats_b:
            verdict = "el modelo supera a **uno** de los baselines"
        elif beats_u is None and beats_b is None:
            verdict = "no se pudo comparar contra los baselines"
        else:
            verdict = "el modelo **no supera** a los baselines (RPS más alto)"
        st.caption(
            f"⚠️ Muestra muy chica (in-sample) — error bars amplios, tratar como "
            f"direccional, no concluyente. Aquí {verdict}."
        )

    # Out-of-sample (2022) — the credibility check ----------------------------
    with hist_col:
        hist = report.get("historical_2022") or {}
        if not hist:
            st.markdown("**🟢 Histórico 2022 (out-of-sample)**")
            st.info("Backtest histórico 2022 no disponible.")
        else:
            h_n = hist.get("n", "—")
            h_model = hist.get("model") or {}
            h_uniform = hist.get("baseline_uniform") or {}
            h_baserate = hist.get("baseline_baserate") or {}
            st.markdown(f"**🟢 Histórico 2022 (out-of-sample, n={h_n})**")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Accuracy modelo", _fmt_pct(h_model.get("accuracy")))
            h2.metric("RPS modelo", _fmt_rps(h_model.get("rps")))
            h3.metric("RPS uniforme", _fmt_rps(h_uniform.get("rps")))
            h4.metric("RPS base-rate", _fmt_rps(h_baserate.get("rps")))
            if hist.get("beats_uniform") and hist.get("beats_baserate"):
                st.success("✅ supera a ambos baselines (out-of-sample, n grande)")
            else:
                st.warning("⚠️ no supera a ambos baselines en el histórico")
            st.caption("Esta es la prueba de credibilidad: muestra grande, "
                       "sin sobreajuste a los partidos en curso.")

    # --- Diagnostics ----------------------------------------------------------
    diag = report.get("diagnostics") or {}
    st.markdown("**🔬 Diagnósticos**")
    draw_bias = diag.get("draw_bias") or {}
    fav = diag.get("favorite_hit_rate") or {}
    d1, d2, d3 = st.columns(3)
    pred_draw = draw_bias.get("mean_predicted_draw_prob")
    emp_draw = draw_bias.get("empirical_draw_frequency")
    gap = draw_bias.get("gap_pred_minus_emp")
    d1.metric("Draw predicho (medio)", _fmt_pct(pred_draw))
    d2.metric("Draw empírico", _fmt_pct(emp_draw),
              delta=f"{gap:+.3f} gap" if isinstance(gap, (int, float)) else None)
    d3.metric("Favorito hit rate", _fmt_pct(fav.get("hit_rate")),
              help=f"{fav.get('hits', '?')} de {fav.get('n_favorites', '?')} "
                   f"(umbral {fav.get('threshold', '?')})")

    misses = diag.get("biggest_misses") or []
    if misses:
        st.caption("Peores fallos (mayor RPS)")
        miss_rows = []
        for m in misses:
            miss_rows.append({
                "Partido": f"{m.get('home', '?')} vs {m.get('away', '?')}",
                "Pick": m.get("pick", "—"),
                "Real": m.get("actual_result", "—"),
                "Marcador": m.get("actual_score", "—"),
                "RPS": _fmt_rps(m.get("rps")),
            })
        st.dataframe(pd.DataFrame(miss_rows), hide_index=True,
                     use_container_width=True)

    # --- Expected vs observed (Σp) -------------------------------------------
    evo = report.get("expected_vs_observed") or {}
    if evo:
        st.markdown("**🔢 Suma de probabilidades (esperado vs observado)**")
        evo_labels = {"home": "Local", "draw": "Empate", "away": "Visita"}
        evo_cols = st.columns(3)
        for col, key in zip(evo_cols, ("home", "draw", "away")):
            block = evo.get(key) or {}
            exp = block.get("expected")
            obs = block.get("observed")
            gap = block.get("gap")
            exp_s = f"{exp:.2f}" if isinstance(exp, (int, float)) else "—"
            obs_s = f"{obs}" if isinstance(obs, (int, float)) else "—"
            gap_s = f"{gap:+.2f}" if isinstance(gap, (int, float)) else None
            col.metric(
                f"{evo_labels[key]} (esperado Σp)",
                f"{exp_s} → {obs_s}",
                delta=gap_s,
                help="Esperado Σp (suma de probabilidades predichas) → Observado "
                     "(conteo real). Gap = observado - esperado.",
            )
        st.caption(
            "Σp = partidos esperados de cada tipo; un gap grande revela sesgo "
            "sistemático de calibración (ej. empates sub-predichos)."
        )

    # --- Per-match table ------------------------------------------------------
    per_match = report.get("per_match") or []
    if per_match:
        st.caption("Detalle por partido (in-sample)")
        pm_rows = []
        for p in per_match:
            hit = p.get("hit")
            pm_rows.append({
                "Partido": f"{p.get('home', '?')} vs {p.get('away', '?')}",
                "Fecha": p.get("date", "—"),
                "P(home)": _fmt_pct(p.get("p_home")),
                "P(draw)": _fmt_pct(p.get("p_draw")),
                "P(away)": _fmt_pct(p.get("p_away")),
                "Pick": p.get("pick", "—"),
                "Real": p.get("actual_score", "—"),
                "Acierto": "✓" if hit else ("✗" if hit is False else "—"),
                "RPS": _fmt_rps(p.get("rps")),
            })
        st.dataframe(pd.DataFrame(pm_rows), hide_index=True,
                     use_container_width=True)

    # --- Calibration chart ----------------------------------------------------
    calibration = report.get("calibration") or []
    if calibration:
        cal_rows = []
        for c in calibration:
            pred = c.get("predicted")
            emp = c.get("empirical")
            if isinstance(pred, (int, float)) and isinstance(emp, (int, float)):
                cal_rows.append({"predicted": pred, "empirical": emp})
        if cal_rows:
            cdf = pd.DataFrame(cal_rows).set_index("predicted")[["empirical"]]
            st.caption("Calibración (predicho vs empírico) — más cerca de la "
                       "diagonal es mejor")
            st.line_chart(cdf)

    # --- Improvements ---------------------------------------------------------
    improvements = report.get("improvements") or []
    if improvements:
        st.markdown("**💡 Qué podemos mejorar**")
        st.markdown("\n".join(f"- {str(item)}" for item in improvements))


# --- predictions by date ------------------------------------------------------
def _lead_badge(col, lead_min: float | None) -> None:
    if lead_min is None:
        col.caption("🕓 Lead-time: _generated_at desconocido_")
        return
    txt = _fmt_hm(lead_min)
    if lead_min >= LEAD_OK_MIN:
        col.success(f"✅ Predicho {txt} antes del inicio")
    elif lead_min >= 0:
        col.warning(f"⚠️ {txt} antes")
    else:
        col.error(f"⛔ Predicho {txt} DESPUÉS del inicio")


def _traceability(container, pred: dict, params: dict, mdir: Path) -> None:
    """Audited params/prediction fields, citations, rationale and report.md."""
    # (i) audited fields — research_engine/confidence/generated_at come from
    # params.json; prediction.json carries code_sha, rho, lambda.
    engine = params.get("research_engine") or pred.get("research_engine")
    confidence = params.get("confidence") or pred.get("confidence")
    generated_at = params.get("generated_at") or pred.get("generated_at")
    code_sha = pred.get("code_sha") or params.get("code_sha")
    base_rate = params.get("base_rate")
    rho = pred.get("rho", params.get("rho"))

    container.markdown("**🔎 Auditoría (params / prediction)**")
    audit = {
        "research_engine": engine,
        "confidence": confidence,
        "generated_at": generated_at,
        "code_sha": code_sha,
        "base_rate": base_rate,
        "rho": rho,
    }
    audit_rows = [{"campo": k, "valor": "" if v is None else str(v)}
                  for k, v in audit.items()]
    container.dataframe(pd.DataFrame(audit_rows), hide_index=True,
                        use_container_width=True)

    # prior_strength vs final strength — shows the anti-hallucination clipping.
    strength = params.get("strength") or {}
    prior = params.get("prior_strength") or {}
    cmp_rows = []
    for side in ("home", "away"):
        for comp, label in (("atk", "atk"), ("def", "def")):
            fin = (strength.get(side) or {}).get(comp)
            pri = (prior.get(side) or {}).get(comp)
            delta = ""
            if isinstance(fin, (int, float)) and isinstance(pri, (int, float)) and pri:
                delta = f"{(fin / pri - 1.0) * 100:+.1f}%"
            cmp_rows.append({
                "lado": side, "componente": label,
                "prior": "" if pri is None else f"{pri:.4f}" if isinstance(pri, (int, float)) else str(pri),
                "final": "" if fin is None else f"{fin:.4f}" if isinstance(fin, (int, float)) else str(fin),
                "Δ vs prior": delta,
            })
    if cmp_rows:
        container.caption("prior_strength → final (clipping anti-alucinación ±40%)")
        container.dataframe(pd.DataFrame(cmp_rows), hide_index=True,
                            use_container_width=True)

    # (ii) citations as a bullet list (clickable if URLs).
    citations = pred.get("citations") or params.get("citations") or []
    if citations:
        container.markdown("**📚 Citas**")
        lines = []
        for c in citations:
            c = str(c)
            if c.startswith("http://") or c.startswith("https://"):
                lines.append(f"- [{c}]({c})")
            else:
                lines.append(f"- {c}")
        container.markdown("\n".join(lines))

    # (iii) rationale text.
    rationale = params.get("rationale") or pred.get("rationale")
    if rationale:
        container.markdown("**🧠 Rationale**")
        container.markdown(str(rationale))

    # (iv) full report.md inside an expander.
    report = _read_text(mdir / "report.md")
    if report:
        with container.expander("📄 report.md"):
            st.markdown(report)


def _actual_block(container, row) -> None:
    """Predicted-vs-actual for a graded ledger row."""
    if not bool(getattr(row, "graded", False)):
        return
    ah, aa = getattr(row, "actual_home", None), getattr(row, "actual_away", None)
    res = getattr(row, "actual_result", None)
    rps = getattr(row, "rps", None)
    probs = {"home": getattr(row, "p_home", None),
             "draw": getattr(row, "p_draw", None),
             "away": getattr(row, "p_away", None)}
    valid = {k: v for k, v in probs.items() if isinstance(v, (int, float))}
    pick = max(valid, key=valid.get) if valid else None
    hit = (pick == res) if (pick and res) else None

    container.markdown("**🎯 Predicho vs Real**")
    cols = container.columns(4)
    score = f"{int(ah)}-{int(aa)}" if ah is not None and aa is not None and not pd.isna(ah) else "—"
    cols[0].metric("Marcador real", score)
    cols[1].metric("Resultado", "" if res is None else str(res).upper())
    cols[2].metric("RPS", f"{rps:.3f}" if isinstance(rps, (int, float)) and not pd.isna(rps) else "—")
    if hit is None:
        cols[3].metric("Acierto", "—")
    else:
        cols[3].metric("Acierto (argmax)", "✓" if hit else "✗",
                       help=f"Pick del modelo: {pick} · Real: {res}")


def _match_key(match_number, day, home: str, away: str) -> str:
    if match_number is None or pd.isna(match_number):
        num = ""
    else:
        num = str(int(match_number))
    return "|".join([num, str(day), str(home), str(away)])


def _whatif_options(led: pd.DataFrame) -> list[dict]:
    if led.empty:
        return []
    sort_cols = [c for c in ("graded", "date", "match_number") if c in led.columns]
    df = led.sort_values(sort_cols) if sort_cols else led
    options = []
    for r in df.itertuples():
        key = _match_key(
            getattr(r, "match_number", None),
            getattr(r, "date", ""),
            getattr(r, "home", ""),
            getattr(r, "away", ""),
        )
        options.append(
            {
                "key": key,
                "label": (
                    f"{getattr(r, 'date', '')} - #{getattr(r, 'match_number', '')} "
                    f"{getattr(r, 'home', '')} vs {getattr(r, 'away', '')}"
                ),
                "date": getattr(r, "date", ""),
                "home": getattr(r, "home", ""),
                "away": getattr(r, "away", ""),
                "graded": bool(getattr(r, "graded", False)),
            }
        )
    return options


def _default_whatif_key(options: list[dict]) -> str | None:
    for option in options:
        if not option["graded"]:
            return option["key"]
    return options[-1]["key"] if options else None


def _whatif_params_for_option(option: dict) -> dict:
    day_dir = PRED / str(option["date"])
    mdir = _match_dir(day_dir, option["home"])
    if not mdir:
        return {}
    return _read_json(mdir / "params.json")


def _safe_float(value, fallback: float) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def _bounded_float(value, fallback: float, lo: float, hi: float) -> float:
    return min(max(_safe_float(value, fallback), lo), hi)


def _country_options(led: pd.DataFrame, selected_params: dict) -> list[str]:
    names: set[str] = set(_prediction_strengths().keys())
    if not led.empty:
        for col in ("home", "away"):
            if col in led.columns:
                names.update(str(v) for v in led[col].dropna().unique() if str(v).strip())
    match = selected_params.get("match") or {}
    names.update(str(match.get(side)) for side in ("home", "away") if match.get(side))
    if not names:
        prior = _latest_prior_payload()
        names.update((prior.get("teams") or {}).keys())
    return sorted(names)


def _option_index(options: list[str], preferred: str | None, fallback: str | None = None) -> int:
    if not options:
        return 0
    for candidate in (preferred, fallback):
        if candidate in options:
            return options.index(candidate)
    return 0


def _strength_for_country(country: str, selected_params: dict, selected_side: str) -> dict:
    match = selected_params.get("match") or {}
    strength = selected_params.get("strength") or {}
    for side in (selected_side, "home", "away"):
        if country == match.get(side):
            side_strength = strength.get(side) or {}
            if "atk" in side_strength and "def" in side_strength:
                return side_strength
    prediction_strength = _prediction_strengths().get(country)
    if prediction_strength:
        return prediction_strength
    prior_strength = (_latest_prior_payload().get("teams") or {}).get(country)
    if prior_strength:
        return prior_strength
    return {}


def predictions_view(led: pd.DataFrame, fix: pd.DataFrame) -> None:
    st.subheader("🗓️ Predictions by date")
    if led.empty:
        st.info("No predictions yet. Run `footy worldcup --date today`.")
        return

    kickoffs = _kickoff_lookup(fix)
    dates = sorted(led["date"].dropna().unique())
    day = st.selectbox("Date", dates, index=len(dates) - 1)
    day_dir = PRED / str(day)
    sub = led[led["date"] == day]
    for r in sub.itertuples():
        with st.expander(f"{r.home} vs {r.away}  —  "
                         f"{r.p_home*100:.0f}% / {r.p_draw*100:.0f}% / {r.p_away*100:.0f}%  ({r.top_score})"):
            mdir = _match_dir(day_dir, r.home)
            pred = _read_json(mdir / "prediction.json") if mdir else {}
            params = _read_json(mdir / "params.json") if mdir else {}
            match_key = _match_key(
                getattr(r, "match_number", None),
                getattr(r, "date", ""),
                getattr(r, "home", ""),
                getattr(r, "away", ""),
            )
            if st.button("Usar este partido en what-if", key=f"use_whatif_{match_key}"):
                st.session_state["whatif_selected_match"] = match_key

            # Kickoff lead-time badge (kickoff_utc - generated_at).
            mn = getattr(r, "match_number", None)
            ko = kickoffs.get(int(mn)) if (mn is not None and not pd.isna(mn)) else None
            gen = _parse_dt(params.get("generated_at") or pred.get("generated_at"))
            lead_min = None
            if ko is not None and gen is not None:
                lead_min = (ko - gen).total_seconds() / 60.0
            _lead_badge(st, lead_min)

            # --- existing features preserved: heatmap + market + dossier ---
            cols = st.columns([1, 1])
            if mdir and (mdir / "heatmap.png").exists():
                cols[0].image(str(mdir / "heatmap.png"))
            if pred:
                cols[1].json(pred.get("market", {}) or {"market": "n/a"})
            if mdir and (mdir / "dossier.md").exists():
                with cols[1].expander("Dossier"):
                    st.markdown(_read_text(mdir / "dossier.md"))

            # --- predicted-vs-actual for graded matches ---
            _actual_block(st, r)

            # --- per-match traceability (audit/citations/rationale/report) ---
            if pred or params:
                st.divider()
                _traceability(st, pred, params, mdir if mdir else day_dir)


# --- live what-if -------------------------------------------------------------
def whatif(led: pd.DataFrame) -> None:
    st.subheader("🎛️ Live what-if")
    st.caption("Choose countries, then move sliders to update 1X2 instantly.")
    options = _whatif_options(led)
    labels = {option["key"]: option["label"] for option in options}
    option_by_key = {option["key"]: option for option in options}
    default_key = _default_whatif_key(options)
    if default_key and st.session_state.get("whatif_selected_match") not in option_by_key:
        st.session_state["whatif_selected_match"] = default_key

    selected_key = None
    selected_params = {}
    if options:
        selected_key = st.selectbox(
            "Partido base",
            [option["key"] for option in options],
            format_func=lambda key: labels.get(key, key),
            key="whatif_selected_match",
        )
        selected_params = _whatif_params_for_option(option_by_key[selected_key])

    match = selected_params.get("match") or {}
    countries = _country_options(led, selected_params)
    if not countries:
        st.info("No countries available for what-if.")
        return
    base_suffix = selected_key or "manual"

    c = st.columns(4)
    home = c[0].selectbox(
        "Home",
        countries,
        index=_option_index(countries, str(match.get("home") or ""), "Brazil"),
        key=f"wi_home_country_{base_suffix}",
    )
    away = c[0].selectbox(
        "Away",
        countries,
        index=_option_index(countries, str(match.get("away") or ""), "Morocco"),
        key=f"wi_away_country_{base_suffix}",
    )
    if home == away:
        c[0].warning("Home and Away are the same country.")

    home_strength = _strength_for_country(home, selected_params, "home")
    away_strength = _strength_for_country(away, selected_params, "away")
    prior = _latest_prior_payload()
    widget_suffix = f"{base_suffix}_{_slugfrag(home)}_{_slugfrag(away)}"
    base = c[0].slider(
        "Base rate",
        0.8,
        2.0,
        _bounded_float(
            selected_params.get("base_rate"),
            _safe_float(prior.get("base_rate"), 1.30),
            0.8,
            2.0,
        ),
        0.01,
        key=f"wi_base_{widget_suffix}",
    )
    rho = c[0].slider(
        "rho (Dixon-Coles)",
        -0.20,
        0.20,
        _bounded_float(
            selected_params.get("rho"),
            _safe_float(prior.get("rho"), -0.10),
            -0.20,
            0.20,
        ),
        0.01,
        key=f"wi_rho_{widget_suffix}",
    )
    atk_h = c[1].slider(
        "Home attack", 0.1, 6.0, _bounded_float(home_strength.get("atk"), 1.45, 0.1, 6.0), 0.01,
        key=f"wi_atk_h_{widget_suffix}",
    )
    def_h = c[1].slider(
        "Home defence", 0.1, 6.0, _bounded_float(home_strength.get("def"), 0.86, 0.1, 6.0), 0.01,
        key=f"wi_def_h_{widget_suffix}",
    )
    atk_a = c[2].slider(
        "Away attack", 0.1, 6.0, _bounded_float(away_strength.get("atk"), 0.88, 0.1, 6.0), 0.01,
        key=f"wi_atk_a_{widget_suffix}",
    )
    def_a = c[2].slider(
        "Away defence", 0.1, 6.0, _bounded_float(away_strength.get("def"), 0.63, 0.1, 6.0), 0.01,
        key=f"wi_def_a_{widget_suffix}",
    )

    params = {
        "match": {"home": home, "away": away, "stage": match.get("stage", "group"), "neutral": True},
        "base_rate": base,
        "rho": rho,
        "strength": {"home": {"atk": atk_h, "def": def_h},
                     "away": {"atk": atk_a, "def": def_a}},
        "adjustments": {},
        "home_advantage": {},
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


# --- page assembly ------------------------------------------------------------
st.sidebar.caption("Data refreshes automatically every 30 s (ledger ttl=30).")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

led = _ledger()
fix = _fixture()

freshness_bar()
st.divider()
upcoming_panel(fix)
st.divider()
scoreboard(led)
st.divider()
backtest_view()
st.divider()
predictions_view(led, fix)
st.divider()
whatif(led)
