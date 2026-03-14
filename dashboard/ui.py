"""
ui.py — Failure Intelligence Engine: Dashboard Entry Point

Run with:
    streamlit run dashboard/ui.py

Folder structure:
    dashboard/
    ├── ui.py                   ← this file (entry point only)
    ├── styles/
    │   └── theme.py            ← all CSS
    ├── components/
    │   ├── charts.py           ← Plotly figure builders
    │   ├── widgets.py          ← HTML widget helpers
    │   └── sidebar.py          ← sidebar + PAGE_* constants
    ├── pages/
    │   ├── dashboard_page.py   ← 📊 Dashboard
    │   ├── analyze_page.py     ← 🔬 Analyze    (Phase 1)
    │   ├── diagnose_page.py    ← ⚖  Diagnose   (Phase 3 DiagnosticJury)
    │   └── vault_page.py       ← 📦 Vault
    └── utils/
        ├── api.py              ← all HTTP calls (including /diagnose)
        └── data.py             ← dataframe builders

Navigation labels and routing constants live in components/sidebar.py.
Import PAGE_* from there — never hardcode strings in the router.
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_DASHBOARD_DIR)
for _p in [_PROJECT_ROOT, _DASHBOARD_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Streamlit MUST be the very first import and set_page_config MUST be ───────
# ── the very first st.* call — nothing else before this block ─────────────────
try:
    import streamlit as st
except ImportError:
    raise SystemExit("Run: pip install streamlit requests pandas plotly")

st.set_page_config(
    page_title="FIE · Failure Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── All other imports AFTER set_page_config ────────────────────────────────────
from styles.theme import inject_styles
from components.sidebar import (
    render_sidebar,
    PAGE_DASHBOARD,
    PAGE_ANALYZE,
    PAGE_DIAGNOSE,
    PAGE_VAULT,
)
from components.widgets import render_page_header
from pages import dashboard_page, analyze_page, vault_page
from pages import diagnose_page

# ── Inject CSS ─────────────────────────────────────────────────────────────────
inject_styles()

# ── Sidebar ────────────────────────────────────────────────────────────────────
nav = render_sidebar(default_refresh=10)

# ── Shared page header ─────────────────────────────────────────────────────────
st.markdown(render_page_header(), unsafe_allow_html=True)

# ── Connection banner ──────────────────────────────────────────────────────────
if not nav["connected"]:
    st.error(
        "**API server is unreachable.** "
        "Start the backend: `uvicorn app.main:app --reload` then refresh.",
        icon="🔴",
    )

# ── Page router ────────────────────────────────────────────────────────────────
page = nav["page"]

if page == PAGE_DASHBOARD:
    dashboard_page.render(
        auto_refresh=nav["auto_refresh"],
        refresh_interval=nav["refresh_interval"],
    )
elif page == PAGE_ANALYZE:
    analyze_page.render()
elif page == PAGE_DIAGNOSE:
    diagnose_page.render()
elif page == PAGE_VAULT:
    vault_page.render()