import streamlit as st
from utils.api import fetch_inferences, fetch_trend
from utils.data import build_inference_dataframe
from components.widgets import render_section_label, render_kpi_card_html


_ARCHETYPE_RISK_LEVEL = {
    "HALLUCINATION_RISK":    ("🔴", "#f85149", "CRITICAL"),
    "OVERCONFIDENT_FAILURE": ("🔴", "#f85149", "CRITICAL"),
    "MODEL_BLIND_SPOT":      ("🟠", "#f85149", "HIGH"),
    "UNSTABLE_OUTPUT":       ("🟡", "#e3b341", "MEDIUM"),
    "LOW_CONFIDENCE":        ("🟡", "#e3b341", "MEDIUM"),
    "STABLE":                ("🟢", "#3fb950", "OK"),
}


def _risk_card(r: dict) -> str:
    metrics   = r.get("metrics") or {}
    entropy   = metrics.get("entropy") or 0
    agreement = metrics.get("agreement_score") or 1
    model     = r.get("model_name", "unknown")
    prompt    = (r.get("input_text") or "")[:80]
    output    = (r.get("output_text") or "")[:100]
    ts        = str(r.get("timestamp", ""))[:19]

    risk_colour = "#f85149" if entropy > 0.75 else "#e3b341"

    return f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                padding:14px 16px;margin-bottom:10px;
                border-left:4px solid {risk_colour};">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;
                     color:{risk_colour};font-weight:700;">{model}</span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:10px;
                     color:#6e7681;">{ts}</span>
      </div>
      <div style="font-size:12px;color:#8b949e;margin-bottom:6px;">
        <b style="color:#c9d1d9;">Prompt:</b> {prompt}
      </div>
      <div style="font-size:12px;color:#8b949e;margin-bottom:8px;">
        <b style="color:#c9d1d9;">Output:</b> {output}
      </div>
      <div style="display:flex;gap:12px;">
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:{risk_colour};">
          entropy={entropy:.3f}
        </span>
        <span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8b949e;">
          agreement={agreement:.3f}
        </span>
      </div>
    </div>
    """


def render() -> None:
    st.markdown(render_section_label("Alerts & Degradation Monitor"), unsafe_allow_html=True)

    trend   = fetch_trend()
    records = fetch_inferences()
    df      = build_inference_dataframe(records)

    # ── Degradation status ─────────────────────────────────────────────
    is_degrading = trend.get("is_degrading", False)
    dv           = trend.get("degradation_velocity", 0)
    risk_rate    = trend.get("ema_high_risk_rate", 0)

    if is_degrading:
        st.markdown(
            f"<div style='background:#f8514922;border:1px solid #f8514966;"
            f"border-radius:10px;padding:16px 20px;margin-bottom:20px;"
            f"border-left:5px solid #f85149;'>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;"
            f"color:#f85149;font-weight:700;letter-spacing:1px;margin-bottom:6px;'>"
            f"⚠ ACTIVE DEGRADATION ALERT</div>"
            f"<div style='font-size:13px;color:#c9d1d9;'>"
            f"Degradation velocity <b>{dv:.4f}</b> exceeds threshold. "
            f"High-risk rate: <b>{risk_rate:.1%}</b>. "
            f"Immediate investigation recommended."
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#3fb95022;border:1px solid #3fb95044;"
            "border-radius:10px;padding:14px 20px;margin-bottom:20px;"
            "border-left:5px solid #3fb950;'>"
            "<span style='color:#3fb950;font-weight:700;'>✓ NO ACTIVE ALERTS</span>"
            "<span style='color:#8b949e;font-size:13px;'> — All models operating within normal parameters.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── EMA metrics ────────────────────────────────────────────────────
    st.markdown(render_section_label("EMA Health Metrics"), unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(render_kpi_card_html(
            label="EMA Entropy",
            value=f"{trend.get('ema_entropy', 0):.3f}",
            delta="↑ high" if trend.get('ema_entropy', 0) > 0.75 else "↓ normal",
            variant="risk" if trend.get('ema_entropy', 0) > 0.75 else "ok",
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card_html(
            label="EMA Agreement",
            value=f"{trend.get('ema_agreement', 0):.3f}",
            delta="↓ low" if trend.get('ema_agreement', 0) < 0.5 else "↑ stable",
            variant="risk" if trend.get('ema_agreement', 0) < 0.5 else "ok",
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card_html(
            label="High-Risk Rate",
            value=f"{risk_rate:.1%}",
            delta="↑ elevated" if risk_rate > 0.3 else "↓ normal",
            variant="risk" if risk_rate > 0.3 else "ok",
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card_html(
            label="Degradation Velocity",
            value=f"{dv:.4f}",
            delta="↑ degrading" if is_degrading else "→ stable",
            variant="risk" if is_degrading else "ok",
        ), unsafe_allow_html=True)
    with k5:
        st.markdown(render_kpi_card_html(
            label="Signals Recorded",
            value=str(trend.get("signals_recorded", 0)),
            delta=f"decay α={trend.get('decay_alpha', 0):.2f}",
            variant="info",
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

    # ── High risk inference feed ───────────────────────────────────────
    st.markdown(render_section_label("High-Risk Inference Feed"), unsafe_allow_html=True)

    col_filter, _ = st.columns([1, 2])
    with col_filter:
        min_entropy = st.slider(
            "Min entropy threshold",
            min_value=0.0, max_value=1.0,
            value=0.5, step=0.05,
            label_visibility="collapsed",
        )

    high_risk_records = [
        r for r in records
        if ((r.get("metrics") or {}).get("entropy") or 0) >= min_entropy
    ]
    high_risk_records = list(reversed(high_risk_records[-30:]))

    if not high_risk_records:
        st.markdown(
            "<div style='text-align:center;padding:40px;color:#6e7681;'>"
            f"No inferences with entropy ≥ {min_entropy:.2f}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Showing {len(high_risk_records)} high-risk inferences "
                   f"(entropy ≥ {min_entropy:.2f})")
        for r in high_risk_records:
            st.markdown(_risk_card(r), unsafe_allow_html=True)

    # ── All records summary table ──────────────────────────────────────
    with st.expander(f"📋 All Records Summary ({len(records)} total)", expanded=False):
        if not df.empty:
            display_cols = [c for c in ["model", "timestamp", "entropy", "agreement_score",
                                         "latency_ms"] if c in df.columns]
            st.dataframe(
                df[display_cols].sort_values("timestamp", ascending=False).head(50),
                use_container_width=True,
            )
        else:
            st.caption("No data yet.")