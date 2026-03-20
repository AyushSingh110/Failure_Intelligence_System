import streamlit as st

from utils.api import analyze_outputs, fetch_inferences, fetch_questions_with_outputs
from components.charts import answer_distribution_bar, signal_radar
from components.widgets import render_section_label, render_status_pill, render_empty_state


_ARCHETYPE_COLOURS = {
    "HALLUCINATION_RISK":    "#f85149",
    "OVERCONFIDENT_FAILURE": "#f85149",
    "MODEL_BLIND_SPOT":      "#f85149",
    "UNSTABLE_OUTPUT":       "#e3b341",
    "LOW_CONFIDENCE":        "#e3b341",
    "STABLE":                "#3fb950",
}


def render() -> None:
    st.markdown(render_section_label("Failure Signal Extraction"), unsafe_allow_html=True)

    # ── Mode selector ──────────────────────────────────────────────────
    mode = st.radio(
        "mode",
        ["🔴  Live Feed", "📦  From Vault", "✏️  Manual"],
        horizontal=True,
        label_visibility="collapsed",
    )

    model_outputs: list[str] = []
    selected_prompt: str     = ""

    # ── Mode: Live Feed ────────────────────────────────────────────────
    if mode == "🔴  Live Feed":
        st.markdown(
            "<div style='font-size:12px;color:#8b949e;margin-bottom:8px;'>"
            "Recent inferences from MongoDB — click any row to analyze it."
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Loading live inferences…"):
            records = fetch_inferences()

        if not records:
            st.warning("No inferences in MongoDB yet. Use the SDK or run `python inject_test_data.py`.")
            return

        # Show last 20 records as clickable rows
        recent = list(reversed(records[-20:]))
        cols_header = st.columns([3, 2, 1, 1])
        cols_header[0].caption("**Prompt**")
        cols_header[1].caption("**Model**")
        cols_header[2].caption("**Entropy**")
        cols_header[3].caption("**Action**")

        st.markdown("<hr style='border-color:#21262d;margin:4px 0 8px 0;'>",
                    unsafe_allow_html=True)

        selected_record = None
        for i, r in enumerate(recent):
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            prompt_text = r.get("input_text", "")[:60]
            model_name  = r.get("model_name", "unknown")
            entropy     = (r.get("metrics") or {}).get("entropy", 0) or 0
            c1.caption(f"`{prompt_text}`")
            c2.caption(model_name)
            c3.caption(f"{entropy:.3f}")
            if c4.button("Analyze", key=f"live_{i}", use_container_width=True):
                selected_record = r

        if selected_record:
            model_outputs   = [selected_record.get("output_text", "")]
            selected_prompt = selected_record.get("input_text", "")
            st.info(f"Selected: **{selected_record.get('model_name')}** — "
                    f"`{selected_prompt[:80]}`")

    # ── Mode: Vault ────────────────────────────────────────────────────
    elif mode == "📦  From Vault":
        with st.spinner("Loading vault questions…"):
            grouped = fetch_questions_with_outputs()

        if not grouped:
            st.warning("No vault records found.")
            return

        selected_q = st.selectbox(
            "Select question",
            options=sorted(grouped.keys()),
            label_visibility="collapsed",
        )

        records    = grouped.get(selected_q, [])
        model_names = [r["model_name"] for r in records]
        st.caption(f"{len(records)} output(s): " + ", ".join(f"**{m}**" for m in model_names))

        for r in records:
            st.markdown(
                f"<div style='font-size:12px;padding:4px 0;border-bottom:"
                f"1px solid #21262d;'>"
                f"<b style='color:#58a6ff;'>{r['model_name']}</b>"
                f"&nbsp;→&nbsp;<span style='color:#c9d1d9;'>{r['output_text'][:120]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        model_outputs   = [r["output_text"] for r in records]
        selected_prompt = selected_q

    # ── Mode: Manual ───────────────────────────────────────────────────
    else:
        raw = st.text_area(
            "Model outputs — one per line",
            placeholder="Paris\nParis\nLondon\nParis",
            height=140,
            label_visibility="collapsed",
        )
        model_outputs = [o.strip() for o in raw.splitlines() if o.strip()]

    # ── Run button ─────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    run = col_btn.button("▶  Run Analysis", use_container_width=True, type="primary")
    if selected_prompt:
        col_info.caption(f"Prompt: `{selected_prompt[:100]}`")

    if not run:
        return

    if not model_outputs:
        st.error("No outputs to analyze.")
        return

    # ── Call API ───────────────────────────────────────────────────────
    with st.spinner("Extracting failure signals…"):
        result = analyze_outputs(model_outputs)

    if result is None:
        st.error("Could not reach the API. Ensure the backend is running.")
        return

    fsv       = result.get("failure_signal_vector", {})
    archetype = result.get("archetype", "—")
    risk      = fsv.get("high_failure_risk", False)

    st.markdown("<hr style='border-color:#21262d;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown(render_section_label("Results"), unsafe_allow_html=True)

    # Status pills
    pill_label   = "⚠ HIGH RISK" if risk else "✓ STABLE"
    pill_variant = "risk" if risk else "ok"
    arch_colour  = _ARCHETYPE_COLOURS.get(archetype, "#8b949e")

    st.markdown(
        f"<div style='margin-bottom:16px;display:flex;gap:8px;align-items:center;'>"
        f"{render_status_pill(pill_label, pill_variant)}"
        f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
        f"font-family:IBM Plex Mono,monospace;font-size:12px;font-weight:700;"
        f"background:{arch_colour}22;color:{arch_colour};border:1px solid {arch_colour}44;'>"
        f"{archetype}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Agreement",  f"{fsv.get('agreement_score', 0):.4f}")
    m2.metric("Entropy",    f"{fsv.get('entropy_score', 0):.4f}")
    m3.metric("FSD",        f"{fsv.get('fsd_score', 0):.4f}")
    m4.metric("Similarity", f"{fsv.get('ensemble_similarity', 0):.4f}")

    # Charts
    ca, cb = st.columns([3, 2])
    with ca:
        ac = fsv.get("answer_counts", {})
        if ac:
            st.plotly_chart(answer_distribution_bar(ac), use_container_width=True,
                            config={"displayModeBar": False})
    with cb:
        st.plotly_chart(signal_radar(fsv), use_container_width=True,
                        config={"displayModeBar": False})

    with st.expander("Raw Signal Vector JSON"):
        st.json(fsv)