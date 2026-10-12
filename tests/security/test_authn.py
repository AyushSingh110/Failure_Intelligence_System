"""
Authentication: I-2, I-11, I-12, I-16 and attacks 9–12 of Artifact C.

Tokens are built in the pre-WP-002 format on purpose (they still carry the
`api_key` claim): a token issued before the change must keep verifying, and a
forged or stale claim inside it must not be believed.
"""
from __future__ import annotations

import pytest

from .fakes import (
    DEV_CONSTANT_SECRET, PLATFORM_ADMIN, TENANT_A, TENANT_B, bearer, key_headers, make_token,
)

OWN = "/api/v1/inferences"
ADMIN_ROUTE = "/api/v1/monitor/signal-logs"


# ── Credentials that must be refused ──────────────────────────────────────────

@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer not.a.token"},
    {"Authorization": "Basic Zm9vOmJhcg=="},
    {"X-API-Key": ""},
    {"X-API-Key": "fie-0000000000000000"},
])
def test_malformed_or_unknown_credentials_are_401(client, headers):
    assert client.get(OWN, headers=headers).status_code == 401


def test_expired_token_is_401(client):
    assert client.get(OWN, headers=bearer(make_token(TENANT_A, expires_in_s=-60))).status_code == 401


def test_token_signed_with_another_secret_is_401(client):
    forged = make_token(TENANT_A, secret="some-other-secret-that-is-long-enough-000000")
    assert client.get(OWN, headers=bearer(forged)).status_code == 401


def test_token_signed_with_the_source_constant_is_401(client):
    assert client.get(OWN, headers=bearer(make_token(TENANT_A, secret=DEV_CONSTANT_SECRET))).status_code == 401


def test_token_without_a_tenant_is_401(client):
    assert client.get(OWN, headers=bearer(make_token(TENANT_A, tenant_id=None))).status_code == 401


@pytest.mark.parametrize("tenant", ["", "anonymous"])
def test_a_user_record_without_a_real_tenant_cannot_authenticate(client, fakedb, tenant):
    fakedb["users"].insert_one({**TENANT_A, "email": "ghost@x.test", "api_key": "fie-ghost00000000000",
                                "tenant_id": tenant})
    assert client.get(OWN, headers={"X-API-Key": "fie-ghost00000000000"}).status_code == 401


# ── Credentials that must work ────────────────────────────────────────────────

def test_api_key_authenticates_its_tenant(a):
    assert a.get("/api/v1/auth/me").json()["tenant_id"] == TENANT_A["tenant_id"]


def test_session_token_authenticates_its_tenant(client):
    response = client.get("/api/v1/auth/me", headers=bearer(make_token(TENANT_B)))
    assert response.status_code == 200
    assert response.json()["tenant_id"] == TENANT_B["tenant_id"]


def test_environment_key_is_a_platform_admin(client, monkeypatch):
    monkeypatch.setenv("FIE_API_KEY", "env-admin-key-for-the-suite-0001")
    assert client.get(ADMIN_ROUTE, headers={"X-API-Key": "env-admin-key-for-the-suite-0001"}).status_code == 200
    assert client.get(ADMIN_ROUTE, headers={"X-API-Key": "env-admin-key-for-the-suite-0002"}).status_code == 401


# ── S10 / I-11: the signing secret ────────────────────────────────────────────

def _deploy_without_secret(monkeypatch):
    import app.auth as auth

    monkeypatch.setenv("JWT_SECRET_KEY", "")
    monkeypatch.setattr(auth, "_jwt_secret_raw", "", raising=False)
    monkeypatch.setattr(auth, "JWT_SECRET", DEV_CONSTANT_SECRET, raising=False)


def test_no_secret_no_tokens(client, monkeypatch):
    """Attack 9: with no secret configured, a token signed with the constant in the source is refused."""
    _deploy_without_secret(monkeypatch)
    forged = make_token(PLATFORM_ADMIN, secret=DEV_CONSTANT_SECRET, tenant_id="forged-tenant")
    assert client.get(OWN, headers=bearer(forged)).status_code == 401
    assert client.get(ADMIN_ROUTE, headers=bearer(forged)).status_code == 401


def test_no_token_is_issued_without_a_secret(monkeypatch):
    import app.auth as auth

    _deploy_without_secret(monkeypatch)
    with pytest.raises(Exception):
        auth.create_session_token(dict(TENANT_A))


def test_a_short_secret_counts_as_no_secret(monkeypatch):
    import app.auth as auth

    monkeypatch.setenv("JWT_SECRET_KEY", "too-short")
    monkeypatch.setattr(auth, "_jwt_secret_raw", "too-short", raising=False)
    monkeypatch.setattr(auth, "JWT_SECRET", "too-short", raising=False)
    with pytest.raises(Exception):
        auth.create_session_token(dict(TENANT_A))
    assert auth.verify_session_token(make_token(TENANT_A, secret="too-short")) is None


def test_startup_fails_without_secret(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    import storage.database as database

    _deploy_without_secret(monkeypatch)
    monkeypatch.setattr(main, "_warm_models_in_background", lambda: None)
    monkeypatch.setattr(database, "initialize_vault", lambda: None)
    with pytest.raises(Exception):
        with TestClient(main.app):
            pass


def test_the_development_switch_is_explicit_and_logged(monkeypatch, events):
    import app.auth as auth

    _deploy_without_secret(monkeypatch)
    monkeypatch.setenv("FIE_ALLOW_INSECURE_DEV_SECRET", "1")
    token = auth.create_session_token(dict(TENANT_A))
    assert auth.verify_session_token(token)["tenant_id"] == TENANT_A["tenant_id"]
    from app.main import enforce_startup_security
    enforce_startup_security()
    assert events.named("startup.insecure_secret")


# ── S9: what a token contains and what it is trusted for ──────────────────────

def test_session_token_contains_no_api_key():
    import jwt
    import app.auth as auth

    payload = jwt.decode(auth.create_session_token(dict(TENANT_A)), options={"verify_signature": False})
    assert "api_key" not in payload
    assert TENANT_A["api_key"] not in str(payload)
    assert payload["tenant_id"] == TENANT_A["tenant_id"]


def test_admin_claim_in_a_token_is_not_believed(client):
    """Attack 10: a validly signed token that says is_admin, for a user who is not."""
    stale = make_token(TENANT_A, is_admin=True)
    assert client.get(ADMIN_ROUTE, headers=bearer(stale)).status_code == 403
    assert client.get("/api/v1/auth/users", headers=bearer(stale)).status_code == 403


def test_admin_flag_is_checked_in_database(client, fakedb, events):
    token = make_token(PLATFORM_ADMIN)
    assert client.get(ADMIN_ROUTE, headers=bearer(token)).status_code == 200
    fakedb["users"].update_one({"email": PLATFORM_ADMIN["email"]}, {"$set": {"is_admin": False}})
    assert client.get(ADMIN_ROUTE, headers=bearer(token)).status_code == 403
    assert events.named("authz.admin_denied")


def test_admin_routes_fail_closed_when_the_user_store_is_down(client, monkeypatch):
    import app.auth as auth

    token = make_token(PLATFORM_ADMIN)
    monkeypatch.setattr(auth, "_get_users_collection", lambda: None)
    assert client.get(ADMIN_ROUTE, headers=bearer(token)).status_code == 503


def test_admin_routes_fail_closed_when_the_user_store_raises(client, fakedb):
    token = make_token(PLATFORM_ADMIN)
    fakedb["users"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = client.get(ADMIN_ROUTE, headers=bearer(token))
    assert response.status_code == 503
    assert "ZQ-INTERNAL-DETAIL" not in response.text


def test_tenant_routes_never_fall_back_to_anonymous_when_the_user_store_is_down(client, monkeypatch, fakedb):
    import app.auth as auth

    monkeypatch.setattr(auth, "_get_users_collection", lambda: None)
    response = client.post("/api/v1/monitor", headers=key_headers(TENANT_A),
                           json={"prompt": "hello", "primary_output": "hi"})
    assert response.status_code == 401
    assert fakedb["inferences"].docs == []


# ── N5 / I-16: key rotation ───────────────────────────────────────────────────

def test_key_rotation_replaces_key(client, events):
    old = key_headers(TENANT_A)
    response = client.post("/api/v1/auth/regenerate-key", headers=bearer(make_token(TENANT_A)))
    assert response.status_code == 200, response.text
    new_key = response.json()["api_key"]
    assert new_key and new_key != TENANT_A["api_key"]
    assert client.get(OWN, headers=old).status_code == 401
    assert client.get(OWN, headers={"X-API-Key": new_key}).status_code == 200
    assert events.named("auth.key_rotated")


def test_key_rotation_reports_failure_when_nothing_was_stored(client, monkeypatch):
    import app.auth as auth

    token = make_token(TENANT_A)
    monkeypatch.setattr(auth, "_get_users_collection", lambda: None)
    response = client.post("/api/v1/auth/regenerate-key", headers=bearer(token))
    assert response.status_code == 503
    assert "api_key" not in response.text


# ── N6: login ─────────────────────────────────────────────────────────────────

class _GoogleResponse:
    def __init__(self, payload: dict, ok: bool = True, text: str = ""):
        self._payload, self.ok, self.text = payload, ok, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.text)


def _fake_google(monkeypatch, userinfo: dict, token_ok: bool = True, token_text: str = ""):
    import app.auth_routes as auth_routes

    monkeypatch.setattr(auth_routes.http_requests, "post",
                        lambda *a, **k: _GoogleResponse({"access_token": "t"}, ok=token_ok, text=token_text))
    monkeypatch.setattr(auth_routes.http_requests, "get", lambda *a, **k: _GoogleResponse(userinfo))


LOGIN = "/api/v1/auth/google-callback"


def test_login_with_a_verified_email_creates_a_tenant(client, monkeypatch, fakedb):
    _fake_google(monkeypatch, {"email": "new.user@fresh.test", "name": "New", "verified_email": True})
    response = client.post(LOGIN, json={"code": "c"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_admin"] is False and body["tenant_id"]
    assert client.get(OWN, headers=bearer(body["token"])).status_code == 200


@pytest.mark.parametrize("userinfo", [
    {"name": "No Email", "verified_email": True},
    {"email": "", "name": "Empty Email", "verified_email": True},
    {"email": "unverified@x.test", "name": "Unverified", "verified_email": False},
    {"email": "unverified@x.test", "name": "No Flag"},
])
def test_login_requires_verified_email(client, monkeypatch, fakedb, userinfo):
    before = len(fakedb["users"].docs)
    _fake_google(monkeypatch, userinfo)
    response = client.post(LOGIN, json={"code": "c"})
    assert response.status_code in (400, 401, 403)
    assert "token" not in response.text
    assert len(fakedb["users"].docs) == before


def test_empty_admin_email_grants_nothing(monkeypatch, fakedb):
    import app.auth as auth

    monkeypatch.setattr(auth, "ADMIN_EMAIL", "")
    monkeypatch.setenv("ADMIN_EMAIL", "")
    with pytest.raises(Exception):
        auth.get_or_create_user(email="", name="nobody")
    user = auth.get_or_create_user(email="someone@x.test", name="someone")
    assert user["is_admin"] is False
    assert all(not u.get("is_admin") or u["email"] == PLATFORM_ADMIN["email"] for u in fakedb["users"].docs)


def test_login_error_does_not_return_the_provider_response(client, monkeypatch):
    _fake_google(monkeypatch, {}, token_ok=False, token_text="ZQ-GOOGLE-INTERNAL-DETAIL client_secret=abc")
    response = client.post(LOGIN, json={"code": "c"})
    assert response.status_code == 400
    assert "ZQ-GOOGLE-INTERNAL-DETAIL" not in response.text


def test_auth_me_returns_only_the_callers_own_key(a, b):
    assert a.get("/api/v1/auth/me").json()["api_key"] == TENANT_A["api_key"]
    assert TENANT_A["api_key"] not in b.get("/api/v1/auth/me").text
