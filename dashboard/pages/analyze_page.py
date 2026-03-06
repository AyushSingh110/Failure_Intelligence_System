import streamlit as st

from utils.api import analyze_outputs
from components.charts import answer_distribution_bar, signal_radar
from components.widgets import render_section_label, render_status_pill, render_empty_state


def render() -> None:
    st.markdown(render_section_label("Failure Signal Extraction"), unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(render_section_label("Inputs"), unsafe_allow_html=True)

        model_outputs_raw = st.text_area(
            "Model Outputs — one per line",
            placeholder="Paris\nParis\nLondon\nParis",
            height=130,
            help="Paste multiple sampled outputs from your model, one per line.",
        )
        primary_output = st.text_input(
            "Primary Model Output",
            placeholder="The capital of France is Paris.",
            help="Output from your primary / production model.",
        )
        secondary_output = st.text_input(
            "Secondary Model Output",
            placeholder="France's capital city is Lyon.",
            help="Output from a secondary / shadow model for ensemble comparison.",
        )

        run = st.button("▶  Run Signal Extraction", use_container_width=True, type="primary")

    with col_result:
        st.markdown(render_section_label("Results"), unsafe_allow_html=True)

        if not run:
            st.markdown(
                render_empty_state(
                    "Enter model outputs on the left<br>and run extraction to see<br>"
                    "the Failure Signal Vector here."
                ),
                unsafe_allow_html=True,
            )
            return

        model_outputs = [o.strip() for o in model_outputs_raw.splitlines() if o.strip()]
        if not model_outputs:
            st.error("Enter at least one model output.")
            return

        with st.spinner("Extracting failure signals..."):
            result = analyze_outputs(model_outputs, primary_output, secondary_output)

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
