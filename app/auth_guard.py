from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from app.auth import get_user_by_api_key, get_user_by_email, verify_session_token


def resolve_user(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> Optional[dict]:
    """Resolve the current user from a bearer token or API key."""
    if not isinstance(authorization, str):
        authorization = None
    if not isinstance(x_api_key, str):
        x_api_key = None

    user = None

    if authorization and authorization.startswith("Bearer "):
        payload = verify_session_token(authorization.split(" ", 1)[1])
        if payload:
            user = get_user_by_email(payload["email"])

    if not user and x_api_key:
        user = get_user_by_api_key(x_api_key)

    return user


def require_user(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> dict:
    """Require an authenticated user for protected routes."""
    user = resolve_user(authorization, x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_admin(
    authorization: Optional[str],
    x_api_key: Optional[str],
) -> dict:
    """Require an authenticated admin for protected routes."""
    user = require_user(authorization, x_api_key)
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def can_access_tenant_record(user: dict, tenant_id: str) -> bool:
    """Admins can see all records; regular users only see their own tenant data."""
    return bool(user.get("is_admin")) or user.get("tenant_id") == tenant_id
