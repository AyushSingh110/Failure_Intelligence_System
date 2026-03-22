"""
app/auth_routes.py

FastAPI endpoints for authentication:
  POST /auth/google         → Google OAuth login/signup
  GET  /auth/me             → get current user info
  GET  /auth/users          → admin only — all users
  POST /auth/regenerate-key → generate new API key
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ────────────────────────────────────────────────────────────────

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

@router.post("/google", response_model=LoginResponse)
def google_login(body: GoogleLoginRequest) -> LoginResponse:
    """
    Called after Google OAuth completes in Streamlit.
    Creates user if new, returns JWT session token + API key.
    """
    # Import here to avoid circular import
    from app.auth import get_or_create_user, create_session_token

    try:
        user  = get_or_create_user(
            email   = body.email,
            name    = body.name,
            picture = body.picture,
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
        logger.error("Google login failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}")


@router.get("/me", response_model=UserInfo)
def get_me(
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
) -> UserInfo:
    """
    Returns current user info.
    Accepts:
      - Authorization: Bearer <jwt_token>  (dashboard users)
      - X-API-Key: fie-xxx                 (SDK users)
    """
    from app.auth import verify_session_token, get_user_by_email, get_user_by_api_key

    user = None

    # Try JWT token first (dashboard login)
    if authorization and authorization.startswith("Bearer "):
        token   = authorization.split(" ")[1]
        payload = verify_session_token(token)
        if payload:
            user = get_user_by_email(payload["email"])

    # Try API key (SDK usage)
    if not user and x_api_key:
        user = get_user_by_api_key(x_api_key)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    return UserInfo(
        email       = user["email"],
        name        = user["name"],
        api_key     = user["api_key"],
        tenant_id   = user["tenant_id"],
        plan        = user.get("plan", "free"),
        is_admin    = user.get("is_admin", False),
        calls_used  = user.get("calls_used", 0),
        calls_limit = user.get("calls_limit", 1000),
    )


@router.get("/users")
def get_users(
    authorization: Optional[str] = Header(None),
) -> list[dict]:
    """Admin only — returns all registered users."""
    from app.auth import verify_session_token, get_all_users

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")

    token   = authorization.split(" ")[1]
    payload = verify_session_token(token)

    if not payload or not payload.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")

    return get_all_users()


@router.post("/regenerate-key")
def regenerate_key_endpoint(
    authorization: Optional[str] = Header(None),
) -> dict:
    """Generates new API key — old key immediately invalid."""
    from app.auth import verify_session_token, regenerate_api_key

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")

    token   = authorization.split(" ")[1]
    payload = verify_session_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid session")

    new_key = regenerate_api_key(payload["email"])
    return {"api_key": new_key, "message": "New API key generated successfully"}