"""
Authentication and route policy — the only place that reads a credential.

    request → authenticate() → Principal → policy dependency → handler

Every route declares exactly one policy as a FastAPI dependency:

    public                   no credential needed (declared, so it is a decision)
    require_tenant           a valid credential; the handler receives a Principal
    require_platform_admin   a valid credential whose admin right is confirmed
                             against the user store during this request

The tenant comes from the verified credential and from nothing else. A request
body, query parameter, header, session id or conversation id never selects it.
A route with no policy fails tests/security/test_route_matrix.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

from app import auth
from app.security_events import emit

_RESERVED_TENANTS = frozenset({"", "anonymous"})
_UNAUTHENTICATED = "Authentication required"


@dataclass(frozen=True)
class Principal:
    """Who is calling. Built only by authenticate(); never from request data."""
    tenant_id:       str
    subject:         str            # e-mail; for audit, never for scoping
    role:            str            # "tenant" | "platform_admin"
    credential_kind: str            # "api_key" | "session" | "env_key"
    admin_verified:  bool = False   # admin right read from the store (or env key) in this request
    admin_hint:      bool = False   # the token claims admin; a reason to check, never a grant

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or self.tenant_id.strip() in _RESERVED_TENANTS:
            raise ValueError("a principal needs a real tenant")
        if self.role not in ("tenant", "platform_admin"):
            raise ValueError("unknown role")

    @property
    def is_platform_admin(self) -> bool:
        return self.role == "platform_admin" and self.admin_verified


def _principal_from_user(user: dict, kind: str) -> Optional[Principal]:
    tenant_id, email = user.get("tenant_id"), user.get("email")
    if not isinstance(tenant_id, str) or not isinstance(email, str):
        return None
    is_admin = user.get("is_admin") is True
    try:
        return Principal(
            tenant_id       = tenant_id,
            subject         = email,
            role            = "platform_admin" if is_admin else "tenant",
            credential_kind = kind,
            admin_verified  = is_admin,
        )
    except ValueError:
        return None


def authenticate(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[Principal]:
    """
    Turn a credential into a Principal, or None.

    A session token proves identity and tenant for its lifetime. Its `is_admin`
    claim is not believed: it only tells require_platform_admin to look the user
    up. An API key is resolved against the user store, so its admin flag is
    current. If the store cannot be read, nobody is authenticated by key.
    """
    if not isinstance(authorization, str):
        authorization = None
    if not isinstance(x_api_key, str):
        x_api_key = None

    if authorization and authorization.startswith("Bearer "):
        payload = auth.verify_session_token(authorization.split(" ", 1)[1].strip())
        if isinstance(payload, dict):
            tenant_id, email = payload.get("tenant_id"), payload.get("email")
            if isinstance(tenant_id, str) and isinstance(email, str) and email:
                try:
                    return Principal(
                        tenant_id       = tenant_id,
                        subject         = email,
                        role            = "tenant",
                        credential_kind = "session",
                        admin_hint      = payload.get("is_admin") is True,
                    )
                except ValueError:
                    return None
            return None

    if x_api_key:
        env_user = auth.env_key_user(x_api_key)
        if env_user is not None:
            return _principal_from_user(env_user, "env_key")
        try:
            user = auth.find_user_by_api_key(x_api_key)
        except auth.UserStoreUnavailable:
            return None
        if user is not None:
            return _principal_from_user(user, "api_key")

    return None


def _authenticated(request: Request, authorization: Optional[str], x_api_key: Optional[str]) -> Principal:
    principal = authenticate(authorization, x_api_key)
    if principal is None:
        supplied = bool(authorization) or bool(x_api_key)
        emit(
            "authn.invalid" if supplied else "authn.missing",
            severity="warning" if supplied else "info",
            outcome="denied",
            reason="invalid_credential" if supplied else "missing_credential",
            request=request,
        )
        raise HTTPException(status_code=401, detail=_UNAUTHENTICATED)
    request.state.principal = principal
    return principal


def verify_platform_admin(principal: Principal) -> Principal:
    """
    Confirm the admin right now. Raises 403 when the caller is not an admin and
    503 when that cannot be established — never "trust the token instead".
    """
    if principal.is_platform_admin:
        return principal
    if principal.credential_kind != "session" or not principal.admin_hint:
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        user = auth.find_user_by_email(principal.subject)
    except auth.UserStoreUnavailable:
        raise HTTPException(status_code=503, detail="Authorization service unavailable")
    if not user or user.get("is_admin") is not True or user.get("tenant_id") != principal.tenant_id:
        raise HTTPException(status_code=403, detail="Admin access required")
    return Principal(
        tenant_id       = principal.tenant_id,
        subject         = principal.subject,
        role            = "platform_admin",
        credential_kind = principal.credential_kind,
        admin_verified  = True,
        admin_hint      = True,
    )


# ── Policies ──────────────────────────────────────────────────────────────────

def public() -> None:
    """Declared policy for a route that needs no credential."""
    return None


def require_tenant(
    request:       Request,
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
) -> Principal:
    return _authenticated(request, authorization, x_api_key)


def require_platform_admin(
    request:       Request,
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
) -> Principal:
    principal = _authenticated(request, authorization, x_api_key)
    try:
        verified = verify_platform_admin(principal)
    except HTTPException as exc:
        emit(
            "authz.admin_denied", severity="warning", outcome="denied",
            reason="not_admin" if exc.status_code == 403 else "admin_check_unavailable",
            principal=principal, request=request,
        )
        raise
    request.state.principal = verified
    return verified


def platform_admin_or_none(
    request:       Request,
    authorization: Optional[str] = Header(None),
    x_api_key:     Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[Principal]:
    """For a public route that shows more to a platform admin. Never raises, grants nothing by default."""
    principal = authenticate(authorization, x_api_key)
    if principal is None:
        return None
    try:
        return verify_platform_admin(principal)
    except HTTPException:
        return None


# ── Helpers for handlers ──────────────────────────────────────────────────────

def ensure_principal(principal: object) -> Principal:
    """
    A handler called without dependency resolution (a direct Python call) receives
    the dependency marker instead of a Principal. It must refuse, not run as nobody.
    """
    if not isinstance(principal, Principal):
        raise HTTPException(status_code=401, detail=_UNAUTHENTICATED)
    return principal


def is_current_admin(principal: Principal) -> bool:
    """True only when the admin right is confirmed now. Used for optional detail, never for access."""
    try:
        return verify_platform_admin(principal).is_platform_admin
    except HTTPException:
        return False


def authorize_cross_tenant_read(request: Request, principal: Principal, resource: str) -> Principal:
    """
    The single gate for reading across tenants on a data route (`all_tenants=true`).
    Requires a currently verified platform admin and records the read.
    """
    try:
        verified = verify_platform_admin(principal)
    except HTTPException as exc:
        emit(
            "authz.admin_denied", severity="warning", outcome="denied",
            reason="not_admin" if exc.status_code == 403 else "admin_check_unavailable",
            principal=principal, request=request, resource=resource,
        )
        raise
    emit("admin.cross_tenant_read", outcome="allowed", reason="all_tenants",
         principal=verified, request=request, resource=resource)
    return verified
