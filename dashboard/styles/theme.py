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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  /* ── Base reset ───────────────────────────────────────────────── */
  html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #090b10 !important;
    color: #c9d1d9 !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
  }

  /* Main content area */
  .main .block-container {
    padding: 1.2rem 2.5rem 2.5rem 2.5rem !important;
    max-width: 1440px !important;
    background-color: #090b10 !important;
  }

  /* App background */
  .stApp {
    background-color: #090b10 !important;
  }

  h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
  }

  /* ── Custom scrollbar ─────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: #21262d;
    border-radius: 3px;
  }
  ::-webkit-scrollbar-thumb:hover { background: #30363d; }

  /* ── Sidebar ──────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #080b12 100%) !important;
    border-right: 1px solid #161b22 !important;
    overflow: visible !important;
  }
  /* Keep fixed width only while expanded so collapsed state can show its toggle correctly. */
  [data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 260px !important;
    max-width: 260px !important;
  }
  [data-testid="stSidebar"] > div:first-child {
    padding: 1.2rem 1.2rem 1rem 1.2rem !important;
  }

  /* Sidebar text — ONLY direct text, not widget internals */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span:not([data-baseweb]),
  [data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
  }

  /* Radio nav items — pill-style buttons */
  [data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #8b949e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #e6edf3 !important;
    background: rgba(88,166,255,0.06) !important;
  }
  /* Selected radio item */
  [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div,
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    color: #58a6ff !important;
    background: rgba(88,166,255,0.08) !important;
    font-weight: 600 !important;
    border-left: 2px solid #58a6ff !important;
  }

  /* Sidebar dividers */
  [data-testid="stSidebar"] hr {
    border-color: #1b1f27 !important;
    margin: 12px 0 !important;
  }

  /* ── Toggle fix ───────────────────────────────────────────────── */
  [data-testid="stToggle"] > label > div[data-testid="stWidgetLabel"] {
    color: #c9d1d9 !important;
  }
  [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] {
    color: #c9d1d9 !important;
  }
  [role="switch"] {
    background-color: #30363d !important;
    transition: background-color 0.2s ease !important;
  }
  [role="switch"][aria-checked="true"] {
    background-color: #238636 !important;
  }
  [role="switch"] div {
    background-color: #ffffff !important;
  }
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
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #58a6ff !important;
  }
  div[data-baseweb="slider"] > div > div {
    background: linear-gradient(to right, #58a6ff 0%, #58a6ff var(--progress, 50%), #30363d var(--progress, 50%)) !important;
  }

  /* ── Buttons ──────────────────────────────────────────────────── */
  .stButton > button {
    background: rgba(33,38,45,0.8) !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    backdrop-filter: blur(8px) !important;
  }
  .stButton > button:hover {
    background: rgba(48,54,61,0.9) !important;
    border-color: #58a6ff !important;
    color: #58a6ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }
  /* Primary button */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    font-weight: 600 !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 16px rgba(31,111,235,0.35) !important;
  }

  /* ── Text inputs / textareas ──────────────────────────────────── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
    outline: none !important;
  }
  .stTextInput label,
  .stTextArea label {
    color: #8b949e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
  }

  /* ── Number input ─────────────────────────────────────────────── */
  .stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
  }
  .stNumberInput label {
    color: #8b949e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }

  /* ── Selectbox ────────────────────────────────────────────────── */
  .stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease !important;
  }
  .stSelectbox > div > div:hover {
    border-color: #30363d !important;
  }
  .stSelectbox label {
    color: #8b949e !important;
    font-weight: 500 !important;
  }

  /* ── Metric boxes ─────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s ease !important;
  }
  [data-testid="stMetric"]:hover {
    border-color: #30363d !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #e6edf3 !important;
  }
  [data-testid="stMetricLabel"] { color: #6e7681 !important; }

  /* ── Dataframe ────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    overflow: hidden !important;
  }
  [data-testid="stDataFrame"] * { color: #c9d1d9 !important; }

  /* ── Expander ─────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
  }
  [data-testid="stExpander"]:hover {
    border-color: #30363d !important;
  }
  [data-testid="stExpander"] summary {
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
  }

  /* ── Info / warning / error alerts ───────────────────────────── */
  [data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
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

  /* ── Tabs ─────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px !important;
    background-color: transparent !important;
    border-bottom: 1px solid #21262d !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: #6e7681 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.15s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #c9d1d9 !important; }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    background-color: rgba(88,166,255,0.06) !important;
  }

  /* ── JSON viewer ──────────────────────────────────────────────── */
  pre {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
  }

  /* ── Hide default Streamlit chrome (keep header intact for sidebar toggle) */
  #MainMenu                        { visibility: hidden !important; }
  footer                           { visibility: hidden !important; }
  .stDeployButton                  { display: none !important; }
    /* Do not hide the toolbar container globally: newer Streamlit versions can host
      the sidebar toggle affordance in this header area. */
    [data-testid="stToolbar"]        { visibility: visible !important; }
  [data-testid="stDecoration"]     { display: none !important; }

  /* Header: transparent but fully functional — sidebar toggle lives here */
  header[data-testid="stHeader"] {
    background: #090b10 !important;
    box-shadow: none !important;
    border: none !important;
    z-index: 999990 !important;
  }

  /* ── Sidebar collapse / expand button ─────────────────────────── */
  [data-testid="stSidebar"] button[kind="header"],
  [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {
    color: #c9d1d9 !important;
    visibility: visible !important;
  }
  [data-testid="collapsedControl"] button,
  [data-testid="stSidebarCollapsedControl"] button {
    color: #c9d1d9 !important;
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
  }
  [data-testid="collapsedControl"] button:hover,
  [data-testid="stSidebarCollapsedControl"] button:hover {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #58a6ff !important;
  }

  [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: inline-flex !important;
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 999991 !important;
  }

  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    opacity: 1 !important;
    visibility: visible !important;
    position: fixed !important;
    top: 0.65rem !important;
    left: 0.75rem !important;
    z-index: 999992 !important;
  }

  /* ── Smooth transitions for columns ───────────────────────────── */
  [data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
  }

  /* ── Plotly chart containers ──────────────────────────────────── */
  [data-testid="stPlotlyChart"] {
    border-radius: 10px !important;
    overflow: hidden !important;
  }
</style>
"""


def inject_styles() -> None:
    """Call once per page load, immediately after st.set_page_config()."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)