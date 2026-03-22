"""
dashboard/pages/settings_page.py
Professional settings page showing API key, usage, and SDK guide.
"""

from __future__ import annotations
import os
import requests
import streamlit as st

FIE_API_BASE = os.getenv("FIE_API_URL", "http://localhost:8000/api/v1")


def render() -> None:

    name      = st.session_state.get("name", "User")
    email     = st.session_state.get("email", "")
    api_key   = st.session_state.get("api_key", "")
    plan      = st.session_state.get("plan", "free")
    is_admin  = st.session_state.get("is_admin", False)
    tenant_id = st.session_state.get("tenant_id", "")

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:32px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:#8b949e;letter-spacing:2px;text-transform:uppercase;
                        margin-bottom:8px;">Account Settings</div>
            <div style="font-size:26px;font-weight:700;color:#e6edf3;">
                Your FIE Account
            </div>
            <div style="font-size:13px;color:#8b949e;margin-top:4px;">
                Manage your API key, view usage, and connect your LLM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two column layout ──────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT COLUMN ────────────────────────────────────────────────────────
    with col_left:

        # Profile card
        plan_colour = "#f85149" if is_admin else "#58a6ff"
        plan_label  = "ADMIN" if is_admin else plan.upper()

        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:12px;padding:24px;margin-bottom:20px;">
                <div style="display:flex;align-items:center;gap:16px;
                            margin-bottom:16px;">
                    <div style="width:52px;height:52px;border-radius:50%;
                                background:linear-gradient(135deg,#58a6ff,#3fb950);
                                display:flex;align-items:center;justify-content:center;
                                font-size:22px;font-weight:700;color:#0d1117;">
                        {name[0].upper()}
                    </div>
                    <div>
                        <div style="font-size:17px;font-weight:700;
                                    color:#e6edf3;">{name}</div>
                        <div style="font-size:12px;color:#8b949e;
                                    margin-top:2px;">{email}</div>
                    </div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span style="padding:3px 12px;border-radius:20px;
                                 font-family:'IBM Plex Mono',monospace;
                                 font-size:11px;font-weight:700;
                                 background:{plan_colour}22;color:{plan_colour};
                                 border:1px solid {plan_colour}44;">
                        {plan_label}
                    </span>
                    <span style="padding:3px 12px;border-radius:20px;
                                 font-family:'IBM Plex Mono',monospace;
                                 font-size:11px;color:#6e7681;
                                 background:#21262d;border:1px solid #30363d;">
                        {tenant_id}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Key card
        st.markdown(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:#58a6ff;letter-spacing:1px;text-transform:uppercase;
                        margin-bottom:12px;">Your API Key</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background:#0d1117;border:1px solid #30363d;
                        border-radius:8px;padding:16px;margin-bottom:8px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;
                            color:#3fb950;letter-spacing:1px;word-break:break-all;">
                    {api_key}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Copy button using st.code (built-in copy)
        st.code(api_key, language=None)
        st.caption("Click the copy icon above to copy your API key.")

        st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

        # Regenerate key
        with st.expander("Regenerate API Key"):
            st.warning(
                "Your old key will stop working immediately. "
                "Update your @monitor decorator with the new key.",
            )
            if st.button("Confirm — Generate New Key", type="primary", key="regen"):
                token = st.session_state.get("token", "")
                try:
                    resp = requests.post(
                        f"{FIE_API_BASE}/auth/regenerate-key",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        new_key = resp.json()["api_key"]
                        st.session_state["api_key"] = new_key
                        st.success(f"New key generated!")
                        st.rerun()
                    else:
                        st.error("Failed to regenerate key.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        # Logout
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        if st.button("Logout", key="logout", use_container_width=True):
            for key in ["token", "email", "name", "api_key",
                        "tenant_id", "is_admin", "plan", "logged_in"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── RIGHT COLUMN ───────────────────────────────────────────────────────
    with col_right:

        # SDK Integration Guide
        st.markdown(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:#58a6ff;letter-spacing:1px;text-transform:uppercase;
                        margin-bottom:12px;">Quick Start — Connect Your LLM</div>
            """,
            unsafe_allow_html=True,
        )

        # Step 1
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:14px 16px;margin-bottom:12px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                            color:#3fb950;margin-bottom:8px;">STEP 1 — Install SDK</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("pip install fie-sdk", language="bash")

        # Step 2
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:14px 16px;margin-bottom:12px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                            color:#3fb950;margin-bottom:8px;">
                    STEP 2 — Add decorator to your LLM function
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.code(
            f"""from fie import monitor

@monitor(
    api_key = "{api_key}",
    fie_url = "http://localhost:8000",
    mode    = "correct",
)
def call_your_llm(prompt: str) -> str:
    # your existing LLM code here
    return your_llm(prompt)

# Use exactly as before
answer = call_your_llm("Who invented telephone?")
# FIE monitors and fixes automatically""",
            language="python",
        )

        # Step 3
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:14px 16px;margin-bottom:12px;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                            color:#3fb950;margin-bottom:8px;">
                    STEP 3 — Watch your dashboard update live
                </div>
                <div style="font-size:12px;color:#8b949e;line-height:1.7;">
                    Every LLM call is monitored automatically.<br>
                    Failures are detected, diagnosed, and fixed.<br>
                    Come back to this dashboard to see your model performance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Two modes explanation
        st.markdown(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:#58a6ff;letter-spacing:1px;text-transform:uppercase;
                        margin:20px 0 12px;">Two Modes Available</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:#161b22;border:1px solid #30363d;
                            border-radius:8px;padding:14px;
                            border-top:3px solid #3fb950;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:11px;color:#3fb950;
                                margin-bottom:8px;">mode="monitor"</div>
                    <div style="font-size:12px;color:#8b949e;line-height:1.6;">
                        Fast async monitoring.<br>
                        User gets answer instantly.<br>
                        FIE checks in background.<br>
                        <b style="color:#c9d1d9;">Best for: speed</b>
                    </div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;
                            border-radius:8px;padding:14px;
                            border-top:3px solid #58a6ff;">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:11px;color:#58a6ff;
                                margin-bottom:8px;">mode="correct"</div>
                    <div style="font-size:12px;color:#8b949e;line-height:1.6;">
                        Real-time correction.<br>
                        User always gets fixed answer.<br>
                        Slight delay (~2-3 seconds).<br>
                        <b style="color:#c9d1d9;">Best for: accuracy</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )