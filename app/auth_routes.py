from __future__ import annotations
import logging
import os

import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from app.auth_guard import Principal, ensure_principal, public, require_platform_admin, require_tenant
from app.limiter import rate_limit
from app.security_events import emit

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"


def _google_client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "")

def _google_client_secret() -> str:
    return os.getenv("GOOGLE_CLIENT_SECRET", "")

def _google_redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5173")


# Schemas
class GoogleCallbackRequest(BaseModel):
    code:         str
    redirect_uri: str = "http://localhost:5173"

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


# Endpoints
# Rate limits protect the login surface from brute-force / credential-stuffing.
# Limits are per client IP (slowapi get_remote_address) and no-op when slowapi
# is not installed, so local development is unaffected.
@router.post("/google-callback", response_model=LoginResponse, dependencies=[Depends(public)])
@rate_limit("10/minute")
def google_callback(request: Request, body: GoogleCallbackRequest) -> LoginResponse:
    """
    React sends Google auth code here.
    We exchange it for user info, then create/fetch user.

    Error bodies are generic on purpose: the provider's response and our own
    exception text go to the log, not to the caller.
    """
    from app.auth import get_or_create_user, create_session_token
    client_id     = _google_client_id()
    client_secret = _google_client_secret()
    redirect_uri  = body.redirect_uri or _google_redirect_uri()
    logger.info(
        "google_callback: client_id_present=%s client_secret_present=%s",
        bool(client_id), bool(client_secret),
    )

    #Exchange code for access token
    try:
        token_resp = http_requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          body.code,
                "client_id":     client_id,
                "client_secret": client_secret,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
            timeout=10,
        )
        if not token_resp.ok:
            logger.error(
                "Google token exchange failed. client_id_present=%s client_secret_present=%s",
                bool(client_id), bool(client_secret),
            )
            raise HTTPException(status_code=400, detail="Token exchange failed")
        tokens = token_resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Token exchange failed: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="Token exchange failed")

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
        logger.error("Failed to get Google user info: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail="Failed to get user info")

    # An account is tied to an e-mail address Google has verified. A profile with
    # no e-mail (the caller can ask Google for a scope without one) or with an
    # unverified one does not get an account.
    email = google_user.get("email") if isinstance(google_user, dict) else None
    if not isinstance(email, str) or not email.strip() or google_user.get("verified_email") is not True:
        emit("authn.invalid", severity="warning", outcome="denied", reason="unverified_email", request=request)
        raise HTTPException(status_code=403, detail="Login requires a verified e-mail address")

    # Step 3 — Create/fetch user in MongoDB
    try:
        user  = get_or_create_user(
            email   = email,
            name    = google_user.get("name", "") or "",
            picture = google_user.get("picture", "") or "",
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
        logger.error("User creation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me", response_model=UserInfo)
@rate_limit("60/minute")
def get_me(
    request:   Request,
    principal: Principal = Depends(require_tenant),
) -> UserInfo:
    """The caller's own account, read fresh from the user store."""
    from app.auth import env_key_user, get_user_by_email

    principal = ensure_principal(principal)
    if principal.credential_kind == "env_key":
        user = env_key_user(os.getenv("FIE_API_KEY", ""))   # the operator's key has no stored account
    else:
        user = get_user_by_email(principal.subject)
    if not user or user.get("tenant_id") != principal.tenant_id:
        raise HTTPException(status_code=401, detail="Invalid session")
    return UserInfo(
        email=user["email"], name=user.get("name", ""),
        api_key=user["api_key"], tenant_id=user["tenant_id"],
        plan=user.get("plan","free"), is_admin=user.get("is_admin",False),
        calls_used=user.get("calls_used",0), calls_limit=user.get("calls_limit",1000),
    )


@router.get("/users")
@rate_limit("30/minute")
def get_users(
    request:   Request,
    principal: Principal = Depends(require_platform_admin),
) -> list[dict]:
    """Every registered user, without API keys. Platform admin only."""
    from app.auth import get_all_users
    return get_all_users()


@router.post("/regenerate-key")
@rate_limit("5/minute")
def regenerate_key_endpoint(
    request:   Request,
    principal: Principal = Depends(require_tenant),
) -> dict:
    """Replace the caller's API key. The old key stops working immediately."""
    from app.auth import regenerate_api_key

    principal = ensure_principal(principal)
    if principal.credential_kind == "env_key":
        raise HTTPException(status_code=400, detail="The environment key is rotated in the deployment, not here")
    new_key = regenerate_api_key(principal.subject)
    if not new_key:
        raise HTTPException(status_code=503, detail="Key rotation is unavailable; the existing key is unchanged")
    emit("auth.key_rotated", outcome="changed", reason="key_rotation", principal=principal, request=request)
    return {"api_key": new_key, "message": "New API key generated"}
