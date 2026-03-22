"""
ui.py - FIE Dashboard Entry Point
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(_DASHBOARD_DIR)
for _p in [_PROJECT_ROOT, _DASHBOARD_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Run: pip install streamlit")

st.set_page_config(
    page_title            = "FIE - Failure Intelligence",
    page_icon             = "diamond",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

from styles.theme import inject_styles
from components.sidebar import (
    render_sidebar,
    PAGE_DASHBOARD,
    PAGE_ANALYZE,
    PAGE_DIAGNOSE,
    PAGE_VAULT,
    PAGE_ALERTS,
    PAGE_SETTINGS,
)
from components.widgets import render_page_header
from pages import dashboard_page, analyze_page, vault_page, diagnose_page
from pages import alerts_page, login_page, settings_page

inject_styles()


def _is_logged_in() -> bool:
    if not st.session_state.get("logged_in"):
        return False
    token = st.session_state.get("token", "")
    return bool(token)


# Show login if not authenticated
if not _is_logged_in():
    logged_in = login_page.render()
    if not logged_in:
        st.stop()

# User is logged in - show dashboard
nav = render_sidebar(default_refresh=10)

st.markdown(render_page_header(), unsafe_allow_html=True)

if not nav["connected"]:
    st.error(
        "**API server unreachable.** "
        "Start: `uvicorn app.main:app --reload` then refresh.",
        icon="🔴",
    )

# Set tenant filter
tenant_id = st.session_state.get("tenant_id", "")
is_admin  = st.session_state.get("is_admin", False)

if is_admin:
    os.environ["FIE_TENANT_ID"] = ""
else:
    os.environ["FIE_TENANT_ID"] = tenant_id

# Page router - using constants from sidebar.py (no hardcoded strings)
page = nav["page"]

if page == PAGE_DASHBOARD:
    dashboard_page.render(
        auto_refresh     = nav["auto_refresh"],
        refresh_interval = nav["refresh_interval"],
    )
elif page == PAGE_ANALYZE:
    analyze_page.render()
elif page == PAGE_DIAGNOSE:
    diagnose_page.render()
elif page == PAGE_VAULT:
    vault_page.render()
elif page == PAGE_ALERTS:
    alerts_page.render()
elif page == PAGE_SETTINGS:
    settings_page.render()