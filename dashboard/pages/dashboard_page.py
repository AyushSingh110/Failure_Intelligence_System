import time
import streamlit as st

from utils.api import fetch_inferences
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
)


@st.cache_data(ttl=10)
def _cached_inferences() -> list[dict]:
    return fetch_inferences()


def render(auto_refresh: bool, refresh_interval: int) -> None:
    records = _cached_inferences()
    df      = build_inference_dataframe(records)
    kpi     = compute_kpi_summary(df, records)

    avg_e   = kpi["avg_entropy"]
    avg_a   = kpi["avg_agreement"]
    total   = kpi["total"]
    r_count = kpi["high_risk_count"]
    r_pct   = kpi["risk_pct"]

    # Count unique models for the KPI
    n_models = df["model"].nunique() if not df.empty and "model" in df.columns else 0

    # ── KPI cards ──
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(render_kpi_card_html(
            label="Total Inferences",
            value=str(total),
            delta=f"{n_models} model{'s' if n_models != 1 else ''} tracked",
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

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    has_data    = not df.empty
    has_entropy = has_data and "entropy" in df.columns and df["entropy"].dropna().shape[0] > 0
    has_models  = has_data and "model"   in df.columns and df["model"].nunique() > 0

    if not has_data:
        st.info("No data yet — run `python inject_test_data.py` to populate the vault.")
        return

    #  Model comparison charts (shown when multiple models exist) 
    if has_models and df["model"].nunique() > 1:
        st.markdown(render_section_label("Model Comparison"), unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.plotly_chart(
                model_comparison_bar(df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with chart_col2:
            st.plotly_chart(
                model_risk_timeline(df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    elif has_models:
        # Single model — show a note
        only_model = df["model"].iloc[0]
        st.info(
            f"Only one model detected: **{only_model}**. "
            "Run `python inject_test_data.py` to inject data from multiple models "
            "and enable the Model Comparison charts.",
            icon="ℹ️",
        )

    # Blended signal time series 
    st.markdown(render_section_label("Signal Time Series (All Models)"), unsafe_allow_html=True)

    if has_entropy:
        st.plotly_chart(
            entropy_agreement_timeseries(df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("No entropy data yet.")

    # Lower row: recent inferences + latency 
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown(render_section_label("Recent Inferences"), unsafe_allow_html=True)
        if records:
            for r in reversed(records[-8:]):
                metrics = r.get("metrics") or {}
                e_val   = metrics.get("entropy") or metrics.get("entropy_score")
                # Show model name in the inference row
                model_label = f"{r.get('model_name', 'unknown')} v{r.get('model_version', '?')}"
                st.markdown(
                    render_inference_row(
                        request_id=r.get("request_id", ""),
                        model_name=model_label,
                        timestamp=str(r.get("timestamp", "")),
                        entropy_val=e_val,
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(render_empty_state("No records yet."), unsafe_allow_html=True)

    with col_b:
        st.markdown(render_section_label("Latency Distribution"), unsafe_allow_html=True)
        has_lat = (
            has_data
            and "latency_ms" in df.columns
            and df["latency_ms"].dropna().shape[0] > 0
        )
        if has_lat:
            st.plotly_chart(
                latency_histogram(df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            avg_lat = kpi.get("avg_latency")
            if avg_lat:
                st.caption(f"Avg: {avg_lat:.1f} ms")
        else:
            st.caption("No latency data.")

    # Auto-refresh 
    if auto_refresh:
        time.sleep(refresh_interval)
        st.cache_data.clear()
        st.rerun()