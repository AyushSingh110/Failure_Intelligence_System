"""
styles/theme.py — Failure Intelligence Engine Dashboard

All CSS in one place. inject_styles() must be called once per page load,
immediately after st.set_page_config().

Key rules:
- Use data-testid selectors (not class names) for Streamlit internals
  because Streamlit's generated class names change between versions.
- Never use [class*="css"] — too broad and breaks on Streamlit 1.30+.
- Toggle, radio, and slider need explicit colour overrides because the
  sidebar wildcard `* { color }` previously collapsed all their colours
  to a flat grey, making them look invisible or broken.
"""

MAIN_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  /* ── Base reset ───────────────────────────────────────────────── */
  html, body {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #0a0c10 !important;
    color: #c9d1d9 !important;
  }

  /* Main content area */
  .main .block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1600px !important;
    background-color: #0a0c10 !important;
  }

  /* App background */
  .stApp {
    background-color: #0a0c10 !important;
  }

  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
  }

  /* ── Sidebar ──────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
  }

  /* Sidebar text — ONLY direct text, not widget internals */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span:not([data-baseweb]),
  [data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
  }

  /* Radio buttons in sidebar */
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #c9d1d9 !important;
    font-size: 13px !important;
  }

  /* Radio selected indicator */
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div {
    color: #58a6ff !important;
    font-weight: 600 !important;
  }

  /* ── Toggle fix ───────────────────────────────────────────────── */
  /* The toggle track — OFF state */
  [data-testid="stToggle"] > label > div[data-testid="stWidgetLabel"] {
    color: #c9d1d9 !important;
  }

  /* Toggle track colour when OFF */
  [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] {
    color: #c9d1d9 !important;
  }

  /* Toggle thumb and track — target BaseWeb toggle component */
  [role="switch"] {
    background-color: #30363d !important;
  }
  [role="switch"][aria-checked="true"] {
    background-color: #238636 !important;
  }
  [role="switch"] div {
    background-color: #ffffff !important;
  }

  /* Toggle label text */
  [data-testid="stToggle"] p,
  [data-testid="stToggle"] span,
  [data-testid="stToggle"] label {
    color: #c9d1d9 !important;
    font-size: 13px !important;
  }

  /* ── Slider ───────────────────────────────────────────────────── */
  [data-testid="stSlider"] p,
  [data-testid="stSlider"] span,
  [data-testid="stSlider"] label {
    color: #c9d1d9 !important;
  }
  /* Slider filled track */
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #58a6ff !important;
  }
  div[data-baseweb="slider"] > div > div {
    background: linear-gradient(to right, #58a6ff 0%, #58a6ff var(--progress, 50%), #30363d var(--progress, 50%)) !important;
  }

  /* ── Buttons ──────────────────────────────────────────────────── */
  .stButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
  }
  /* Primary button (Run Signal Extraction) */
  .stButton > button[kind="primary"] {
    background: #1f6feb !important;
    color: #ffffff !important;
    border-color: #1f6feb !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #388bfd !important;
    border-color: #388bfd !important;
  }

  /* ── Text inputs / textareas ──────────────────────────────────── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
    outline: none !important;
  }
  .stTextInput label,
  .stTextArea label {
    color: #8b949e !important;
    font-size: 12px !important;
  }

  /* ── Selectbox ────────────────────────────────────────────────── */
  .stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
  }
  .stSelectbox label { color: #8b949e !important; }

  /* ── Metric boxes ─────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #e6edf3 !important;
  }
  [data-testid="stMetricLabel"] { color: #6e7681 !important; }

  /* ── Dataframe ────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
  }
  [data-testid="stDataFrame"] * { color: #c9d1d9 !important; }

  /* ── Expander ─────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] summary {
    color: #8b949e !important;
    font-size: 12px !important;
  }

  /* ── Info / warning / error alerts ───────────────────────────── */
  [data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
  }

  /* ── Caption / small text ─────────────────────────────────────── */
  [data-testid="stCaptionContainer"] p {
    color: #6e7681 !important;
    font-size: 11px !important;
  }

  /* ── Divider ──────────────────────────────────────────────────── */
  hr { border-color: #21262d !important; }

  /* ── Spinner ──────────────────────────────────────────────────── */
  [data-testid="stSpinner"] { color: #58a6ff !important; }

  /* ── Hide default Streamlit chrome ───────────────────────────── */
  header[data-testid="stHeader"]  { display: none !important; }
  #MainMenu                        { visibility: hidden !important; }
  footer                           { visibility: hidden !important; }
  .stDeployButton                  { display: none !important; }
  [data-testid="stToolbar"]        { display: none !important; }
</style>
"""


def inject_styles() -> None:
    """Call once per page load, immediately after st.set_page_config()."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)