"""
app/auth_routes.py - Auth endpoints
"""
from __future__ import annotations
import logging
import os
import urllib.parse
from typing import Optional

import requests as http_requests
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5173")
GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL      = "https://www.googleapis.com/oauth2/v2/userinfo"


# ── Schemas ────────────────────────────────────────────────────────────────

class GoogleCallbackRequest(BaseModel):
    code:         str
    redirect_uri: str = "http://localhost:5173"

class GoogleLoginRequest(BaseModel):
    email:   str
    name:    str
    picture: str = ""

class LoginResponse(BaseModel):
    token:       str
    email:       str
    name:        str
    api_key:     str
    tenant_id:   str
    plan:        str
    is_admin:    bool
    calls_used:  int
    calls_limit: int

class UserInfo(BaseModel):
    email:       str
    name:        str
    api_key:     str
    tenant_id:   str
    plan:        str
    is_admin:    bool
    calls_used:  int
    calls_limit: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/google-callback", response_model=LoginResponse)
def google_callback(body: GoogleCallbackRequest) -> LoginResponse:
    """
    React sends Google auth code here.
    We exchange it for user info, then create/fetch user.
    """
    from app.auth import get_or_create_user, create_session_token
    redirect_uri = GOOGLE_REDIRECT_URI or body.redirect_uri

    if body.redirect_uri and body.redirect_uri != redirect_uri:
        logger.warning(
            "Google callback redirect mismatch: frontend=%s backend=%s",
            body.redirect_uri,
            redirect_uri,
        )

    # Step 1 — Exchange code for access token
    try:
        token_resp = http_requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          body.code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
            timeout=10,
        )
        if not token_resp.ok:
            logger.error(
                "Google token exchange failed. redirect_uri=%s client_id_present=%s client_secret_present=%s response=%s",
                redirect_uri,
                bool(GOOGLE_CLIENT_ID),
                bool(GOOGLE_CLIENT_SECRET),
                token_resp.text,
            )
            raise HTTPException(status_code=400, detail=f"Token exchange failed: {token_resp.text}")
        tokens = token_resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Token exchange failed: %s", exc)
        response_text = ""
        if getattr(exc, "response", None) is not None:
            try:
                response_text = exc.response.text
            except Exception:
                response_text = ""
        detail = f"Token exchange failed: {exc}"
        if response_text:
            detail = f"{detail} | Google response: {response_text}"
        raise HTTPException(status_code=400, detail=detail)

    # Step 2 — Get user info from Google
    try:
        user_resp = http_requests.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10,
        )
        user_resp.raise_for_status()
        google_user = user_resp.json()
    except Exception as exc:
        logger.error("Failed to get Google user info: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to get user info: {exc}")

    # Step 3 — Create/fetch user in MongoDB
    try:
        user  = get_or_create_user(
            email   = google_user.get("email", ""),
            name    = google_user.get("name", ""),
            picture = google_user.get("picture", ""),
        )
        token = create_session_token(user)

        return LoginResponse(
            token       = token,
            email       = user["email"],
            name        = user["name"],
            api_key     = user["api_key"],
            tenant_id   = user["tenant_id"],
            plan        = user.get("plan", "free"),
            is_admin    = user.get("is_admin", False),
            calls_used  = user.get("calls_used", 0),
            calls_limit = user.get("calls_limit", 1000),
        )
    except Exception as exc:
        logger.error("User creation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}")


@router.post("/google", response_model=LoginResponse)
def google_login(body: GoogleLoginRequest) -> LoginResponse:
    """Direct login with email/name (used by Streamlit)."""
    from app.auth import get_or_create_user, create_session_token
    try:
        user  = get_or_create_user(email=body.email, name=body.name, picture=body.picture)
        token = create_session_token(user)
        return LoginResponse(
            token=token, email=user["email"], name=user["name"],
            api_key=user["api_key"], tenant_id=user["tenant_id"],
            plan=user.get("plan","free"), is_admin=user.get("is_admin",False),
            calls_used=user.get("calls_used",0), calls_limit=user.get("calls_limit",1000),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}")


@router.get("/me", response_model=UserInfo)
def get_me(
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
) -> UserInfo:
    from app.auth import verify_session_token, get_user_by_email, get_user_by_api_key
    user = None
    if authorization and authorization.startswith("Bearer "):
        payload = verify_session_token(authorization.split(" ")[1])
        if payload:
            user = get_user_by_email(payload["email"])
    if not user and x_api_key:
        user = get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    return UserInfo(
        email=user["email"], name=user["name"],
        api_key=user["api_key"], tenant_id=user["tenant_id"],
        plan=user.get("plan","free"), is_admin=user.get("is_admin",False),
        calls_used=user.get("calls_used",0), calls_limit=user.get("calls_limit",1000),
    )


@router.get("/users")
def get_users(authorization: Optional[str] = Header(None)) -> list[dict]:
    from app.auth import verify_session_token, get_all_users
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    payload = verify_session_token(authorization.split(" ")[1])
    if not payload or not payload.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return get_all_users()


@router.post("/regenerate-key")
def regenerate_key_endpoint(authorization: Optional[str] = Header(None)) -> dict:
    from app.auth import verify_session_token, regenerate_api_key
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    payload = verify_session_token(authorization.split(" ")[1])
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid session")
    new_key = regenerate_api_key(payload["email"])
    return {"api_key": new_key, "message": "New API key generated"}
