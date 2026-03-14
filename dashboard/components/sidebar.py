"""
components/sidebar.py

Renders the sidebar and returns navigation state.

PAGE_* constants are the single source of truth for navigation labels.
They are imported by ui.py for routing — no copy-paste drift possible.
"""

import streamlit as st
from components.widgets import render_connection_badge
from utils.api import check_connection, API_BASE

# ── Single source of truth for page labels ────────────────────────────────
PAGE_DASHBOARD = "📊 Dashboard"
PAGE_ANALYZE   = "🔬 Analyze"
PAGE_DIAGNOSE  = "⚖  Diagnose"       # Phase 3 — DiagnosticJury
PAGE_VAULT     = "📦 Vault"

_ALL_PAGES = [PAGE_DASHBOARD, PAGE_ANALYZE, PAGE_DIAGNOSE, PAGE_VAULT]

# Nav item metadata for the custom navigation
_NAV_META = {
    PAGE_DASHBOARD: {"icon": "📊", "desc": "Overview & KPIs"},
    PAGE_ANALYZE:   {"icon": "🔬", "desc": "Phase 1 signals"},
    PAGE_DIAGNOSE:  {"icon": "⚖",  "desc": "Jury diagnosis"},
    PAGE_VAULT:     {"icon": "📦", "desc": "Record browser"},
}


def _render_brand() -> None:
    """Renders the brand/logo block at the top of the sidebar."""
    st.markdown("""
    <div style="padding:6px 0 20px 0;">
      <div style="display:flex;align-items:center;gap:11px;">
        <div style="
          width:36px;height:36px;border-radius:10px;
          background:linear-gradient(135deg,#1f6feb 0%,#8b5cf6 100%);
          display:flex;align-items:center;justify-content:center;
          font-size:16px;line-height:1;
          box-shadow:0 0 18px rgba(31,111,235,0.25);
        ">⬡</div>
        <div>
          <div style="font-family:'Inter',sans-serif;font-size:16px;font-weight:700;
                      color:#e6edf3;letter-spacing:-0.4px;line-height:1.2;">FIE</div>
          <div style="font-family:'Inter',sans-serif;font-size:9px;color:#6e7681;
                      letter-spacing:1.8px;text-transform:uppercase;font-weight:500;">
            Failure Intelligence
          </div>
        </div>
      </div>
      <div style="margin-top:10px;display:inline-flex;align-items:center;gap:6px;
                  padding:3px 10px;border-radius:6px;
                  background:rgba(63,185,80,0.08);border:1px solid rgba(63,185,80,0.12);">
        <span style="width:5px;height:5px;border-radius:50%;
                     background:#3fb950;box-shadow:0 0 5px #3fb950;display:inline-block;"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                     color:#3fb950;letter-spacing:0.8px;font-weight:600;">v3.0 ACTIVE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_nav_section() -> str:
    """Renders navigation using st.radio but with a styled section header."""
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:9px;font-weight:600;
                text-transform:uppercase;letter-spacing:2px;color:#484f58;
                margin:4px 0 10px 2px;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        _ALL_PAGES,
        label_visibility="collapsed",
    )
    return page


def _render_phase_status(page: str) -> None:
    """Shows which phase is active based on current page."""
    phases = [
        ("Phase 1", "Signal Extraction", "#3fb950", page == PAGE_ANALYZE),
        ("Phase 2", "Archetype Discovery", "#3fb950", False),
        ("Phase 3", "DiagnosticJury", "#58a6ff", page == PAGE_DIAGNOSE),
    ]

    items = []
    for tag, name, color, active in phases:
        if active:
            dot = f"background:{color};box-shadow:0 0 6px {color};"
        else:
            dot = "background:#30363d;"
        tc = "#e6edf3" if active else "#484f58"
        tg = color if active else "#484f58"
        fw = "600" if active else "400"

        items.append(
            '<div style="display:flex;align-items:center;gap:8px;padding:5px 0;">'
            f'<span style="width:6px;height:6px;border-radius:50%;{dot}'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:9px;'
            f'color:{tg};font-weight:600;min-width:42px;">{tag}</span>'
            f'<span style="font-family:Inter,sans-serif;font-size:11px;'
            f'color:{tc};font-weight:{fw};">{name}</span>'
            '</div>'
        )

    body = "".join(items)
    html = (
        '<div style="font-family:Inter,sans-serif;font-size:9px;font-weight:600;'
        'text-transform:uppercase;letter-spacing:2px;color:#484f58;'
        'margin:0 0 8px 2px;">Pipeline</div>'
        '<div style="background:rgba(13,17,23,0.5);border:1px solid #1b1f27;'
        f'border-radius:8px;padding:8px 12px;">{body}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_refresh_controls(default_refresh: int) -> tuple[bool, int]:
    """Renders refresh controls in a compact layout."""
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:9px;font-weight:600;
                text-transform:uppercase;letter-spacing:2px;color:#484f58;
                margin:0 0 8px 2px;">Controls</div>
    """, unsafe_allow_html=True)

    auto_refresh     = st.toggle("Auto-refresh", value=False)
    refresh_interval = st.slider("Interval (sec)", 5, 60, default_refresh)

    if st.button("⟳  Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return auto_refresh, refresh_interval


def _render_connection(connected: bool) -> None:
    """Renders connection status with API endpoint."""
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:9px;font-weight:600;
                text-transform:uppercase;letter-spacing:2px;color:#484f58;
                margin:0 0 8px 2px;">Connection</div>
    """, unsafe_allow_html=True)

    st.markdown(render_connection_badge(connected), unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:10px;"
        f"color:#484f58;margin-top:6px;word-break:break-all;'>{API_BASE}</div>",
        unsafe_allow_html=True,
    )

    if not connected:
        st.warning(
            "Cannot reach the API.\n\n"
            "Start: `uvicorn app.main:app --reload`",
            icon="⚠️",
        )


def render_sidebar(default_refresh: int = 10) -> dict:
    """
    Renders the full sidebar.
    Returns: { page, auto_refresh, refresh_interval, connected }
    """
    with st.sidebar:
        _render_brand()

        st.markdown("<div style='border-top:1px solid #1b1f27;margin:4px 0 16px 0;'></div>",
                    unsafe_allow_html=True)

        page = _render_nav_section()

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

        _render_phase_status(page)

        st.markdown("<div style='border-top:1px solid #1b1f27;margin:16px 0;'></div>",
                    unsafe_allow_html=True)

        auto_refresh, refresh_interval = _render_refresh_controls(default_refresh)

        st.markdown("<div style='border-top:1px solid #1b1f27;margin:16px 0;'></div>",
                    unsafe_allow_html=True)

        connected = check_connection()
        _render_connection(connected)

    return {
        "page":             page,
        "auto_refresh":     auto_refresh,
        "refresh_interval": refresh_interval,
        "connected":        connected,
    }