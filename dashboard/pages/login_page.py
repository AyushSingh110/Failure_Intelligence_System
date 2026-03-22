"""
dashboard/pages/login_page.py
Google OAuth login page.
"""

from __future__ import annotations

import os
import urllib.parse
from dotenv import load_dotenv

# Load .env BEFORE reading env vars
load_dotenv()

import requests
import streamlit as st

# ── Google OAuth Config ────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")
FIE_API_BASE         = os.getenv("FIE_API_URL", "http://localhost:8000/api/v1")

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"


def _build_google_auth_url() -> str:
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "online",
        "prompt":        "select_account",
    }
    return GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)


def _exchange_code_for_user(code: str) -> dict | None:
    try:
        token_resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
            timeout=10,
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        user_resp = requests.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10,
        )
        user_resp.raise_for_status()
        return user_resp.json()

    except Exception as exc:
        st.error(f"Google authentication failed: {exc}")
        return None


def _login_with_fie(email: str, name: str, picture: str) -> dict | None:
    try:
        resp = requests.post(
            f"{FIE_API_BASE}/auth/google",
            json={"email": email, "name": name, "picture": picture},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"FIE login failed: {exc}")
        return None


def render() -> bool:
    """
    Returns True if logged in, False if showing login form.
    """

    # ── Handle Google OAuth callback ──────────────────────────────────────
    query_params = st.query_params
    auth_code    = query_params.get("code", None)

    if auth_code:
        with st.spinner("Signing you in..."):
            google_user = _exchange_code_for_user(auth_code)

            if google_user:
                email   = google_user.get("email", "")
                name    = google_user.get("name", "")
                picture = google_user.get("picture", "")

                fie_user = _login_with_fie(email, name, picture)

                if fie_user:
                    # ── Store ALL user data in session_state ───────────
                    st.session_state["token"]     = fie_user["token"]
                    st.session_state["email"]     = fie_user["email"]
                    st.session_state["name"]      = fie_user["name"]
                    st.session_state["api_key"]   = fie_user["api_key"]
                    st.session_state["tenant_id"] = fie_user["tenant_id"]
                    st.session_state["is_admin"]  = fie_user["is_admin"]
                    st.session_state["plan"]      = fie_user["plan"]
                    st.session_state["logged_in"] = True

                    # ── Clear code from URL ────────────────────────────
                    st.query_params.clear()

                    # ── Force full page reload ─────────────────────────
                    st.success(f"✅ Welcome, {name}! Redirecting...")
                    st.rerun()

    # ── If already logged in via session ──────────────────────────────────
    if st.session_state.get("logged_in"):
        return True

    # ── Show login page ────────────────────────────────────────────────────
    # Center everything
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        # Logo + Title
        st.markdown(
            """
            <div style="text-align:center;">
                <div style="font-size:52px;margin-bottom:8px;">⬡</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                            font-weight:700;color:#e6edf3;margin-bottom:8px;">
                    Failure Intelligence Engine
                </div>
                <div style="font-size:13px;color:#8b949e;line-height:1.7;
                            margin-bottom:28px;">
                    Real-time LLM monitoring, failure detection,<br>
                    and automatic correction platform.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Features box
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:10px;padding:16px 20px;margin-bottom:24px;">
                <div style="font-size:12px;color:#8b949e;margin:5px 0;">
                    <span style="color:#3fb950;margin-right:8px;">✓</span>
                    Monitor every LLM call in real time
                </div>
                <div style="font-size:12px;color:#8b949e;margin:5px 0;">
                    <span style="color:#3fb950;margin-right:8px;">✓</span>
                    Auto-fix wrong answers before users see them
                </div>
                <div style="font-size:12px;color:#8b949e;margin:5px 0;">
                    <span style="color:#3fb950;margin-right:8px;">✓</span>
                    Detect prompt injections and jailbreaks
                </div>
                <div style="font-size:12px;color:#8b949e;margin:5px 0;">
                    <span style="color:#3fb950;margin-right:8px;">✓</span>
                    Your private dashboard, your data only
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Google login button
        auth_url = _build_google_auth_url()
        st.markdown(
            f"""
            <div style="text-align:center;">
                <a href="{auth_url}" target="_self" style="
                    display:inline-flex;align-items:center;gap:12px;
                    padding:12px 28px;background:#ffffff;color:#1f1f1f;
                    border-radius:8px;font-size:15px;font-weight:500;
                    text-decoration:none;border:1px solid #dadce0;">
                    <img src="https://www.google.com/favicon.ico"
                         width="20" height="20">
                    Continue with Google
                </a>
                <div style="margin-top:14px;font-size:11px;color:#6e7681;">
                    Your data is private and isolated from other users.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return False