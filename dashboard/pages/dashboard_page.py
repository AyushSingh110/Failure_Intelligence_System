import time
import streamlit as st

from utils.api import fetch_inferences, fetch_trend
from utils.data import build_inference_dataframe, compute_kpi_summary
from components.charts import (
    entropy_agreement_timeseries,
    latency_histogram,
    model_comparison_bar,
    model_risk_timeline,
)
from components.widgets import (
    render_section_label,
    render_kpi_card_html,
    render_inference_row,
    render_empty_state,
    render_callout,
)


@st.cache_data(ttl=10)
def _cached_inferences() -> list[dict]:
    return fetch_inferences()


@st.cache_data(ttl=10)
def _cached_trend() -> dict:
    return fetch_trend()


def _archetype_badge(archetype: str) -> str:
    colours = {
        "HALLUCINATION_RISK":    ("#f85149", "🔴"),
        "OVERCONFIDENT_FAILURE": ("#f85149", "🔴"),
        "MODEL_BLIND_SPOT":      ("#f85149", "🟠"),
        "UNSTABLE_OUTPUT":       ("#e3b341", "🟡"),
        "LOW_CONFIDENCE":        ("#e3b341", "🟡"),
        "STABLE":                ("#3fb950", "🟢"),
    }
    colour, icon = colours.get(archetype, ("#8b949e", "⚪"))
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:600;'
        f'background:{colour}22;color:{colour};border:1px solid {colour}44;">'
        f'{icon} {archetype}</span>'
    )


def render(auto_refresh: bool, refresh_interval: int) -> None:

    records = _cached_inferences()
    trend   = _cached_trend()
    df      = build_inference_dataframe(records)
    kpi     = compute_kpi_summary(df, records)

    total    = kpi["total"]
    r_count  = kpi["high_risk_count"]
    r_pct    = kpi["risk_pct"]
    avg_e    = kpi["avg_entropy"]
    avg_a    = kpi["avg_agreement"]
    n_models = df["model"].nunique() if not df.empty and "model" in df.columns else 0

    # ── Status banner ──────────────────────────────────────────────────
    is_degrading = trend.get("is_degrading", False)
    if is_degrading:
        st.markdown(
            "<div style='background:#f8514922;border:1px solid #f8514944;"
            "border-radius:8px;padding:12px 16px;margin-bottom:16px;"
            "border-left:4px solid #f85149;'>"
            "<b style='color:#f85149;'>⚠ DEGRADATION DETECTED</b> "
            "<span style='color:#c9d1d9;font-size:13px;'>"
            f"Degradation velocity: {trend.get('degradation_velocity', 0):.3f} — "
            "Model performance is declining. Check the trend charts below."
            "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#3fb95022;border:1px solid #3fb95044;"
            "border-radius:8px;padding:10px 16px;margin-bottom:16px;"
            "border-left:4px solid #3fb950;'>"
            "<b style='color:#3fb950;'>✓ SYSTEM HEALTHY</b> "
            "<span style='color:#8b949e;font-size:13px;'>"
            "All models stable. Monitoring active."
            "</span></div>",
            unsafe_allow_html=True,
        )

    # ── KPI cards ──────────────────────────────────────────────────────
    st.markdown(render_section_label("System Overview"), unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(render_kpi_card_html(
            label="Total Inferences",
            value=str(total),
            delta=f"{n_models} model(s) tracked",
            variant="info",
        ), unsafe_allow_html=True)

    with c2:
        above = avg_e is not None and avg_e > 0.75
        st.markdown(render_kpi_card_html(
            label="Avg Entropy",
            value=f"{avg_e:.3f}" if avg_e is not None else "—",
            delta="↑ above threshold" if above else "↓ within range",
            delta_dir="up" if above else "down",
            variant="risk" if above else "ok",
        ), unsafe_allow_html=True)

    with c3:
        low_a = avg_a is not None and avg_a < 0.50
        st.markdown(render_kpi_card_html(
            label="Avg Agreement",
            value=f"{avg_a:.3f}" if avg_a is not None else "—",
            delta="↓ low agreement" if low_a else "↑ stable",
            delta_dir="up" if low_a else "down",
            variant="risk" if low_a else "ok",
        ), unsafe_allow_html=True)

    with c4:
        st.markdown(render_kpi_card_html(
            label="High-Risk Rate",
            value=f"{r_pct:.1f}%" if total else "—",
            delta=f"{r_count} flagged records",
            delta_dir="up" if r_count > 0 else "",
            variant="risk" if r_count > 0 else "ok",
        ), unsafe_allow_html=True)

    with c5:
        dv = trend.get("degradation_velocity", 0)
        st.markdown(render_kpi_card_html(
            label="Degradation Velocity",
            value=f"{dv:.4f}",
            delta="↑ degrading" if is_degrading else "→ stable",
            delta_dir="up" if is_degrading else "",
            variant="risk" if is_degrading else "ok",
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    if not df.empty and "model" in df.columns and df["model"].nunique() > 1:
        st.markdown(render_section_label("Model Comparison"), unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(model_comparison_bar(df), use_container_width=True,
                            config={"displayModeBar": False})
        with cc2:
            st.plotly_chart(model_risk_timeline(df), use_container_width=True,
                            config={"displayModeBar": False})

    # ── Signal time series ─────────────────────────────────────────────
    st.markdown(render_section_label("Signal Time Series"), unsafe_allow_html=True)
    has_entropy = (not df.empty and "entropy" in df.columns
                   and df["entropy"].dropna().shape[0] > 0)
    if has_entropy:
        st.plotly_chart(entropy_agreement_timeseries(df), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.caption("No entropy data available yet.")

    # ── Live failure feed + latency ────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(render_section_label("Live Failure Feed"), unsafe_allow_html=True)

        # Filter to only high-risk records
        high_risk = [r for r in records if (r.get("metrics") or {}).get("entropy", 0) > 0.5]
        display   = list(reversed(records[-15:])) if not high_risk else list(reversed(high_risk[-15:]))

        if display:
            for r in display:
                metrics   = r.get("metrics") or {}
                e_val     = metrics.get("entropy") or metrics.get("entropy_score")
                model_lbl = f"{r.get('model_name','unknown')} v{r.get('model_version','?')}"
                st.markdown(
                    render_inference_row(
                        request_id=r.get("request_id", ""),
                        model_name=model_lbl,
                        timestamp=str(r.get("timestamp", "")),
                        entropy_val=e_val,
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                render_empty_state("No inferences yet. Use the SDK or Postman to send data."),
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown(render_section_label("Latency Distribution"), unsafe_allow_html=True)
        has_lat = (not df.empty and "latency_ms" in df.columns
                   and df["latency_ms"].dropna().shape[0] > 0)
        if has_lat:
            st.plotly_chart(latency_histogram(df), use_container_width=True,
                            config={"displayModeBar": False})
            avg_lat = kpi.get("avg_latency")
            if avg_lat:
                st.caption(f"Mean latency: {avg_lat:.1f} ms")
        else:
            st.caption("No latency data.")

        # EMA trend summary
        st.markdown(render_section_label("EMA Trend"), unsafe_allow_html=True)
        if trend:
            t1, t2 = st.columns(2)
            t1.metric("EMA Entropy",   f"{trend.get('ema_entropy', 0):.3f}")
            t2.metric("EMA Agreement", f"{trend.get('ema_agreement', 0):.3f}")
            t1.metric("Risk Rate",     f"{trend.get('ema_high_risk_rate', 0):.1%}")
            t2.metric("Signals",       str(trend.get("signals_recorded", 0)))
        else:
            st.caption("No trend data yet.")

    # ── Auto-refresh ───────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(refresh_interval)
        st.cache_data.clear()
        st.rerun()