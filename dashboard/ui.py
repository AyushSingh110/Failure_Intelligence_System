import sys
import os

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_DASHBOARD_DIR)
for _p in [_PROJECT_ROOT, _DASHBOARD_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

from styles.theme import inject_styles
from components.sidebar import (
    render_sidebar,
    PAGE_DASHBOARD,
    PAGE_ANALYZE,
    PAGE_DIAGNOSE,
    PAGE_VAULT,
)
from components.widgets import render_page_header
from pages import dashboard_page, analyze_page, vault_page, diagnose_page
from pages import alerts_page

inject_styles()

nav = render_sidebar(default_refresh=10)

st.markdown(render_page_header(), unsafe_allow_html=True)

if not nav["connected"]:
    st.error(
        "**API server unreachable.** "
        "Start the backend: `uvicorn app.main:app --reload` then refresh.",
        icon="🔴",
    )

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
elif page == "🚨  Alerts":
    alerts_page.render()