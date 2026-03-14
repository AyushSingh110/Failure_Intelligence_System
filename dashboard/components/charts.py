import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Shared layout applied to every chart
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, -apple-system, sans-serif", color="#6e7681", size=11),
    margin=dict(l=44, r=16, t=36, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#21262d",
        borderwidth=1,
        font=dict(size=11, color="#8b949e"),
    ),
)

_AXIS_STYLE = dict(
    gridcolor="rgba(33,38,45,0.6)",
    zeroline=False,
    showline=False,
    tickfont=dict(size=10, color="#484f58"),
)


def _apply_base(fig: go.Figure, height: int, title: str = "") -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, height=height)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=12, color="#6e7681"), x=0))
    return fig


def entropy_agreement_timeseries(df: pd.DataFrame) -> go.Figure:
    """Dual-panel time series: Entropy (top) + Agreement (bottom)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Entropy Score", "Agreement Score"),
    )

    if not df.empty:
        entropy_vals = df["entropy"].dropna()
        if not entropy_vals.empty:
            fig.add_trace(go.Scatter(
                x=df.loc[entropy_vals.index, "timestamp"],
                y=entropy_vals,
                mode="lines+markers",
                name="Entropy",
                line=dict(color="#f85149", width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.06)",
            ), row=1, col=1)
            # Threshold reference line
            fig.add_hline(
                y=0.75, row=1, col=1,
                line=dict(color="#e3b341", width=1, dash="dot"),
                annotation_text="threshold",
                annotation_font=dict(size=9, color="#e3b341"),
            )

        agree_vals = df["agreement"].dropna()
        if not agree_vals.empty:
            fig.add_trace(go.Scatter(
                x=df.loc[agree_vals.index, "timestamp"],
                y=agree_vals,
                mode="lines+markers",
                name="Agreement",
                line=dict(color="#3fb950", width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(63,185,80,0.06)",
            ), row=2, col=1)
            fig.add_hline(
                y=0.5, row=2, col=1,
                line=dict(color="#e3b341", width=1, dash="dot"),
                annotation_text="threshold",
                annotation_font=dict(size=9, color="#e3b341"),
            )

    fig.update_layout(**_BASE_LAYOUT, height=400)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(range=[0, 1], **_AXIS_STYLE)
    fig.update_annotations(font=dict(color="#6e7681", size=10, family="Inter, sans-serif"))
    return fig


def latency_histogram(df: pd.DataFrame) -> go.Figure:
    """Latency distribution histogram."""
    fig = go.Figure()
    if not df.empty and "latency_ms" in df.columns:
        vals = df["latency_ms"].dropna()
        if not vals.empty:
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=20,
                marker_color="#58a6ff",
                marker_line_color="#21262d",
                marker_line_width=1,
                opacity=0.85,
                name="Latency",
            ))
    return _apply_base(fig, height=200)


def answer_distribution_bar(answer_counts: dict[str, int]) -> go.Figure:
    """Bar chart of answer frequency from a consistency result."""
    keys = list(answer_counts.keys())
    vals = list(answer_counts.values())
    colors = ["#f85149" if i == 0 else "#58a6ff" for i in range(len(keys))]
    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=colors,
        marker_line_color="#21262d",
        marker_line_width=1,
    ))
    return _apply_base(fig, height=200, title="Answer Distribution")


def signal_radar(fsv: dict) -> go.Figure:
    """Radar chart summarising all dimensions of a FailureSignalVector."""
    categories = ["Agreement", "FSD", "Ensemble\nSimilarity", "Stability", "Confidence"]
    values = [
        fsv.get("agreement_score", 0),
        fsv.get("fsd_score", 0),
        fsv.get("ensemble_similarity", 0),
        1.0 - fsv.get("entropy_score", 0),   # invert: high entropy = low stability
        1.0 if not fsv.get("high_failure_risk") else 0.0,
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(88,166,255,0.10)",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=5, color="#58a6ff"),
        name="Signal",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        height=280,
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(range=[0, 1], gridcolor="rgba(33,38,45,0.6)", tickfont=dict(size=9, color="#484f58")),
            angularaxis=dict(gridcolor="rgba(33,38,45,0.6)", tickfont=dict(size=10, color="#c9d1d9")),
        ),
    )
    return fig


def model_comparison_bar(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: one group per model, showing Avg Entropy vs Avg Agreement.
    Immediately shows which models are underperforming at a glance.
    """
    fig = go.Figure()
    if df.empty or "model" not in df.columns:
        return _apply_base(fig, 240)

    models   = sorted(df["model"].dropna().unique())
    entropies = []
    agreements = []

    for m in models:
        mdf = df[df["model"] == m]
        entropies.append(round(mdf["entropy"].mean(), 3)   if "entropy"   in mdf.columns else 0)
        agreements.append(round(mdf["agreement"].mean(), 3) if "agreement" in mdf.columns else 0)

    fig.add_trace(go.Bar(
        name="Avg Entropy",
        x=models, y=entropies,
        marker_color="#f85149",
        marker_line_color="#21262d",
        marker_line_width=1,
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Avg Agreement",
        x=models, y=agreements,
        marker_color="#3fb950",
        marker_line_color="#21262d",
        marker_line_width=1,
        opacity=0.85,
    ))

    # Threshold lines
    fig.add_hline(y=0.75, line=dict(color="#e3b341", width=1, dash="dot"),
                  annotation_text="entropy threshold", annotation_font=dict(size=9, color="#e3b341"))
    fig.add_hline(y=0.50, line=dict(color="#58a6ff", width=1, dash="dot"),
                  annotation_text="agreement threshold", annotation_font=dict(size=9, color="#58a6ff"))

    fig.update_layout(barmode="group")
    return _apply_base(fig, height=260, title="Model Performance Comparison")


def model_risk_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Multi-line time series: one line per model, showing entropy over time.
    Makes degradation events visible per-model rather than as a blended average.
    """
    fig = go.Figure()
    if df.empty or "model" not in df.columns:
        return _apply_base(fig, 300)

    MODEL_COLORS = {
        "gpt-4":           "#58a6ff",
        "gpt-3.5-turbo":   "#3fb950",
        "claude-3-sonnet": "#bc8cff",
        "gemini-pro":      "#f0883e",
    }
    DEFAULT_COLORS = ["#58a6ff", "#3fb950", "#bc8cff", "#f0883e", "#e3b341", "#f85149"]

    models = sorted(df["model"].dropna().unique())
    for i, model in enumerate(models):
        mdf   = df[df["model"] == model].dropna(subset=["entropy"])
        color = MODEL_COLORS.get(model, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        if mdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=mdf["timestamp"], y=mdf["entropy"],
            mode="lines+markers",
            name=model,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            opacity=0.9,
        ))

    fig.add_hline(y=0.75, line=dict(color="#e3b341", width=1, dash="dot"),
                  annotation_text="risk threshold", annotation_font=dict(size=9, color="#e3b341"))

    fig.update_yaxes(range=[0, 1])
    return _apply_base(fig, height=300, title="Entropy by Model Over Time")