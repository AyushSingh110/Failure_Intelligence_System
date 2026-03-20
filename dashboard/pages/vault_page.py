import streamlit as st
import pandas as pd

from utils.api import fetch_inferences
from utils.data import build_inference_dataframe
from components.widgets import (
    render_section_label,
    render_kpi_card_html,
    render_field_label,
    render_callout,
    render_empty_state,
)


@st.cache_data(ttl=10)
def _cached_inferences() -> list[dict]:
    return fetch_inferences()


def _model_summary_table(df: pd.DataFrame) -> None:
    """Renders a per-model statistics table."""
    if df.empty or "model" not in df.columns:
        return

    models = df["model"].dropna().unique()
    rows = []
    for m in sorted(models):
        mdf = df[df["model"] == m]
        avg_e   = mdf["entropy"].mean()   if "entropy"   in mdf.columns else None
        avg_a   = mdf["agreement"].mean() if "agreement" in mdf.columns else None
        avg_lat = mdf["latency_ms"].mean() if "latency_ms" in mdf.columns else None
        n_risk  = int((mdf["entropy"] > 0.75).sum()) if "entropy" in mdf.columns else 0
        rows.append({
            "Model":         m,
            "Records":       len(mdf),
            "Avg Entropy":   round(avg_e, 3)   if avg_e   is not None else "—",
            "Avg Agreement": round(avg_a, 3)   if avg_a   is not None else "—",
            "Avg Latency":   f"{avg_lat:.0f} ms" if avg_lat is not None else "—",
            "High-Risk":     n_risk,
            "Risk %":        f"{100*n_risk//len(mdf)}%",
        })

    summary_df = pd.DataFrame(rows)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg Entropy":   st.column_config.ProgressColumn("Avg Entropy",   min_value=0, max_value=1, format="%.3f"),
            "Avg Agreement": st.column_config.ProgressColumn("Avg Agreement", min_value=0, max_value=1, format="%.3f"),
            "Records":       st.column_config.NumberColumn("Records"),
            "High-Risk":     st.column_config.NumberColumn("⚠ High-Risk"),
        }
    )


def render() -> None:
    st.markdown(render_section_label("Inference Vault"), unsafe_allow_html=True)
    st.markdown(
        render_callout(
            "Browse, filter, and inspect every inference record stored in the system. "
            "Use the filters below to drill into specific models or request IDs.",
            "info",
        ),
        unsafe_allow_html=True,
    )

    records = _cached_inferences()

    if not records:
        st.markdown(render_empty_state("No records in the vault yet"), unsafe_allow_html=True)
        st.markdown(
            render_callout("Run <code>python inject_data.py</code> to seed the vault with sample records.", "warning"),
            unsafe_allow_html=True,
        )
        return

    df_all = build_inference_dataframe(records)

    # Model stats summary 
    st.markdown(render_section_label("Model Summary"), unsafe_allow_html=True)

    models_found = sorted(df_all["model"].dropna().unique()) if not df_all.empty else []

    if models_found:
        cols = st.columns(min(len(models_found), 4))
        for i, model in enumerate(models_found):
            mdf    = df_all[df_all["model"] == model]
            avg_e  = mdf["entropy"].mean() if "entropy" in mdf.columns else None
            n_risk = int((mdf["entropy"] > 0.75).sum()) if "entropy" in mdf.columns else 0
            pct    = f"{100*n_risk//len(mdf)}%" if len(mdf) > 0 else "—"
            is_bad = avg_e is not None and avg_e > 0.75

            with cols[i % 4]:
                st.markdown(render_kpi_card_html(
                    label=model,
                    value=f"{len(mdf)} records",
                    delta=f"⚠ {pct} high-risk" if n_risk > 0 else "✓ stable",
                    delta_dir="up" if n_risk > 0 else "down",
                    variant="risk" if is_bad else "ok",
                ), unsafe_allow_html=True)

        st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)
        _model_summary_table(df_all)

    st.markdown("<div style='margin:24px 0'></div>", unsafe_allow_html=True)

    # Filter bar
    st.markdown(render_section_label("Filter & Browse"), unsafe_allow_html=True)

    filter_col, model_col, count_col = st.columns([3, 2, 1])

    with filter_col:
        st.markdown(render_field_label("Search", "request ID, model, or input text"), unsafe_allow_html=True)
        search = st.text_input(
            "Search",
            placeholder="Filter by request ID, model name, or input text…",
            label_visibility="collapsed",
        )

    with model_col:
        st.markdown(render_field_label("Model", "filter by model"), unsafe_allow_html=True)
        model_options = ["All models"] + models_found
        selected_model = st.selectbox(
            "Model filter",
            model_options,
            label_visibility="collapsed",
        )

    with count_col:
        st.markdown(
            f"<div style='padding-top:26px;font-family:JetBrains Mono,monospace;"
            f"font-size:12px;color:#6e7681;text-align:right;'>"
            f"{len(records)} total</div>",
            unsafe_allow_html=True,
        )

    # Apply filters
    filtered = records

    if selected_model != "All models":
        filtered = [r for r in filtered if r.get("model_name", "") == selected_model]

    if search:
        q = search.strip().lower()
        filtered = [
            r for r in filtered
            if q in r.get("request_id", "").lower()
            or q in r.get("model_name", "").lower()
            or q in r.get("input_text", "").lower()
        ]

    if not filtered:
        st.markdown(render_empty_state("No records match your filters"), unsafe_allow_html=True)
        return

    # Records table 
    st.markdown(render_section_label("Records"), unsafe_allow_html=True)

    df = build_inference_dataframe(filtered)
    if not df.empty:
        display_cols = [
            c for c in
            ["timestamp", "request_id", "model", "version",
             "entropy", "agreement", "fsd", "latency_ms", "temperature", "is_correct"]
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols].tail(200),
            use_container_width=True,
            height=340,
            hide_index=True,
            column_config={
                "entropy":    st.column_config.ProgressColumn("Entropy",   min_value=0, max_value=1, format="%.4f"),
                "agreement":  st.column_config.ProgressColumn("Agreement", min_value=0, max_value=1, format="%.4f"),
                "fsd":        st.column_config.ProgressColumn("FSD",       min_value=0, max_value=1, format="%.4f"),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
                "model":      st.column_config.TextColumn("Model"),
                "version":    st.column_config.TextColumn("Version"),
            },
        )

    # ── Record detail ─────────────────────────────────────────────────────────
    st.markdown(render_section_label("Record Detail", margin_top=True), unsafe_allow_html=True)

    record_ids = [r.get("request_id", "") for r in filtered[-100:]]
    if not record_ids:
        return

    st.markdown(render_field_label("Select Record", "choose a request ID to inspect"), unsafe_allow_html=True)
    selected_id = st.selectbox(
        "Select record",
        record_ids,
        label_visibility="collapsed",
    )
    selected = next((r for r in filtered if r.get("request_id") == selected_id), None)

    if selected:
        dc_a, dc_b, dc_c = st.columns(3)

        with dc_a:
            st.markdown(render_section_label("Model"), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:13px;color:#c9d1d9;line-height:1.8;'>"
                f"<b>Name:</b> <code style='color:#79c0ff;'>{selected.get('model_name', '—')}</code><br>"
                f"<b>Version:</b> <code style='color:#79c0ff;'>{selected.get('model_version', '—')}</code><br>"
                f"<b>Temperature:</b> <code style='color:#79c0ff;'>{selected.get('temperature', '—')}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with dc_b:
            st.markdown(render_section_label("Request"), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:13px;color:#c9d1d9;line-height:1.8;'>"
                f"<b>Timestamp:</b> <code style='color:#79c0ff;'>{str(selected.get('timestamp', '—'))[:19]}</code><br>"
                f"<b>Latency:</b> <code style='color:#79c0ff;'>{selected.get('latency_ms', '—')} ms</code><br>"
                f"<b>Correct:</b> <code style='color:#79c0ff;'>{selected.get('is_correct', '—')}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with dc_c:
            st.markdown(render_section_label("Metrics"), unsafe_allow_html=True)
            metrics = selected.get("metrics") or {}
            if metrics:
                lines = "".join(
                    f"<b>{k}:</b> <code style='color:#79c0ff;'>{v}</code><br>"
                    for k, v in metrics.items() if v is not None
                )
                st.markdown(
                    f"<div style='font-family:Inter,sans-serif;font-size:13px;color:#c9d1d9;line-height:1.8;'>"
                    f"{lines}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No metrics stored.")

        st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

        # Input / Output
        io_col_a, io_col_b = st.columns(2)
        with io_col_a:
            st.markdown(render_section_label("Input"), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:13px;color:#8b949e;"
                f"background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px;'>"
                f"{selected.get('input_text', '—')}</div>",
                unsafe_allow_html=True,
            )
        with io_col_b:
            st.markdown(render_section_label("Output"), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:13px;color:#8b949e;"
                f"background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px;'>"
                f"{selected.get('output_text', '—')}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Full JSON"):
            st.json(selected)