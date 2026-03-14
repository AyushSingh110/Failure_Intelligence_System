"""
pages/analyze_page.py

Failure Signal Extraction — analysis of multi-model outputs.

Two modes:
  Auto   — select a stored question from the vault dropdown; model outputs
            are fetched automatically from stored inference records.
  Manual — paste outputs directly, one per line.
"""

import streamlit as st

from utils.api import analyze_outputs, fetch_questions_with_outputs
from components.charts import answer_distribution_bar, signal_radar
from components.widgets import render_section_label, render_status_pill, render_empty_state


def render() -> None:
    st.markdown(render_section_label("Failure Signal Extraction"), unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(render_section_label("Inputs"), unsafe_allow_html=True)

        # ── Mode toggle ────────────────────────────────────────────────
        mode = st.radio(
            "Input mode",
            ["Auto — load from vault", "Manual — paste outputs"],
            horizontal=True,
            label_visibility="collapsed",
        )

        model_outputs: list[str] = []

        if mode == "Auto — load from vault":
            # ── Load grouped questions from backend ────────────────────
            with st.spinner("Loading stored questions…"):
                grouped = fetch_questions_with_outputs()

            if not grouped:
                st.warning(
                    "No stored inference records found. "
                    "Run `python inject_test_data.py` first to populate the vault."
                )
                return

            questions = sorted(grouped.keys())
            selected_q = st.selectbox(
                "Select a question from the vault",
                options=questions,
                label_visibility="collapsed",
            )

            records = grouped.get(selected_q, [])

            # Show which models were found
            model_names = [r["model_name"] for r in records]
            st.caption(
                f"{len(records)} model output(s) found: "
                + ", ".join(f"**{m}**" for m in model_names)
            )

            # Show a preview table of what will be analyzed
            if records:
                st.markdown(
                    "<div style='font-size:12px;color:var(--text-color);opacity:.7;"
                    "margin:8px 0 4px'>Outputs that will be compared:</div>",
                    unsafe_allow_html=True,
                )
                for r in records:
                    st.markdown(
                        f"<div style='font-size:12px;padding:4px 0;border-bottom:"
                        f"1px solid rgba(128,128,128,0.15)'>"
                        f"<b>{r['model_name']}</b> &nbsp;→&nbsp; {r['output_text'][:120]}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            model_outputs = [r["output_text"] for r in records]

        else:
            # ── Manual mode ────────────────────────────────────────────
            model_outputs_raw = st.text_area(
                "Model outputs — one per line",
                placeholder=(
                    "Paris\n"
                    "Paris\n"
                    "London\n"
                    "Paris\n"
                    "Berlin"
                ),
                height=160,
                label_visibility="collapsed",
                help=(
                    "Paste one output per line. "
                    "Line 1 = primary model, Line 2 = reference model, "
                    "Lines 3+ = additional ensemble members."
                ),
            )
            model_outputs = [
                o.strip()
                for o in model_outputs_raw.splitlines()
                if o.strip()
            ]

        run = st.button("▶  Run Signal Extraction", use_container_width=True, type="primary")

    with col_result:
        st.markdown(render_section_label("Results"), unsafe_allow_html=True)

        if not run:
            st.markdown(
                render_empty_state(
                    "Select a question or paste outputs<br>"
                    "on the left and run extraction<br>"
                    "to see the Failure Signal Vector here."
                ),
                unsafe_allow_html=True,
            )
            return

        if not model_outputs:
            st.error("No model outputs to analyze. Select a question or paste outputs.")
            return

        with st.spinner("Extracting failure signals…"):
            result = analyze_outputs(model_outputs)

        if result is None:
            st.error(
                "Could not reach the API. "
                "Ensure `uvicorn app.main:app` is running and try again."
            )
            return

        fsv      = result.get("failure_signal_vector", {})
        archetype = result.get("archetype", "—")
        risk      = fsv.get("high_failure_risk", False)

        # Status row
        pill_variant = "risk" if risk else "ok"
        pill_label   = "⚠ HIGH RISK" if risk else "✓ STABLE"
        st.markdown(
            f"""
            <div style="margin-bottom:16px;display:flex;gap:8px;align-items:center;">
              {render_status_pill(pill_label, pill_variant)}
              {render_status_pill(archetype, "novel")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric strip
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agreement",  f"{fsv.get('agreement_score', 0):.4f}")
        m2.metric("Entropy",    f"{fsv.get('entropy_score', 0):.4f}")
        m3.metric("FSD",        f"{fsv.get('fsd_score', 0):.4f}")
        m4.metric("Similarity", f"{fsv.get('ensemble_similarity', 0):.4f}")

        # Charts
        chart_col_a, chart_col_b = st.columns([3, 2])

        with chart_col_a:
            answer_counts = fsv.get("answer_counts", {})
            if answer_counts:
                st.plotly_chart(
                    answer_distribution_bar(answer_counts),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

        with chart_col_b:
            st.plotly_chart(
                signal_radar(fsv),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with st.expander("Raw Signal Vector JSON"):
            st.json(fsv)