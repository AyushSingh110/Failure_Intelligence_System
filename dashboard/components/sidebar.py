import streamlit as st
from components.widgets import render_connection_badge, render_section_label
from utils.api import check_connection, API_BASE

# ── Single source of truth for page labels ────────────────────────────────
PAGE_DASHBOARD = "📊 Dashboard"
PAGE_ANALYZE   = "🔬 Analyze"
PAGE_DIAGNOSE  = "⚖  Diagnose"       # Phase 3 — DiagnosticJury
PAGE_VAULT     = "📦 Vault"
PAGE_ALERTS    = "🚨  Alerts"

_ALL_PAGES = [PAGE_DASHBOARD, PAGE_ANALYZE, PAGE_DIAGNOSE, PAGE_VAULT, PAGE_ALERTS]


def render_sidebar(default_refresh: int = 10) -> dict:
    """
    Renders the full sidebar.
    Returns: { page, auto_refresh, refresh_interval, connected }
    """
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:8px 0 20px 0;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;
                      font-weight:600;color:#58a6ff;letter-spacing:-0.5px;">⬡ FIE</div>
          <div style="font-size:10px;color:#6e7681;letter-spacing:2px;
                      text-transform:uppercase;margin-top:2px;">
            Failure Intelligence Engine
          </div>
          <div style="font-size:9px;color:#3fb950;letter-spacing:1px;
                      text-transform:uppercase;margin-top:3px;">
            Phase 3 Active
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────
        page = st.radio(
            "Navigation",
            _ALL_PAGES,
            label_visibility="collapsed",
        )

        # ── Phase badge under Diagnose ────────────────────────────────
        if page == PAGE_DIAGNOSE:
            st.markdown(
                "<div style='font-size:10px;color:#bc8cff;letter-spacing:1px;"
                "text-transform:uppercase;padding:4px 0 8px 0;'>"
                "⚖ DiagnosticJury · 2 agents active</div>",
                unsafe_allow_html=True,
            )

        # ── Refresh controls ──────────────────────────────────────────
        st.markdown("---")
        st.markdown(render_section_label("Refresh"), unsafe_allow_html=True)
        auto_refresh     = st.toggle("Auto-refresh", value=False)
        refresh_interval = st.slider("Interval (sec)", 5, 60, default_refresh)

        if st.button("⟳  Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # ── Connection status ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(render_section_label("Connection"), unsafe_allow_html=True)
        st.caption(f"API: `{API_BASE}`")

        connected = check_connection()
        st.markdown(render_connection_badge(connected), unsafe_allow_html=True)

        if not connected:
            st.warning(
                "Cannot reach the API.\n\n"
                "Start: `uvicorn app.main:app --reload`",
                icon="⚠️",
            )

        # ── Phase status legend ───────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;color:#6e7681;line-height:2.0;"
            "font-family:IBM Plex Mono,monospace;'>"
            "<span style='color:#3fb950;'>●</span> Phase 1 — Signal Extraction<br>"
            "<span style='color:#3fb950;'>●</span> Phase 2 — Archetype Discovery<br>"
            "<span style='color:#3fb950;'>●</span> Phase 3 — DiagnosticJury<br>"
            "<span style='color:#6e7681;'>○</span> Agent 3 — DomainCritic (pending)"
            "</div>",
            unsafe_allow_html=True,
        )

    return {
        "page":             page,
        "auto_refresh":     auto_refresh,
        "refresh_interval": refresh_interval,
        "connected":        connected,
    }