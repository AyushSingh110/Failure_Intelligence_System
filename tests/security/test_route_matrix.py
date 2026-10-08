"""
I-1 / AC-15: every route has exactly one declared policy, and the policy is enforced.

EXPECTED is Artifact A of PLAN_002. A route that exists in the application and is
missing here fails `test_every_route_is_listed`, so a new route cannot be added
without someone deciding who may call it.
"""
from __future__ import annotations

import pytest

from .fakes import TENANT_A, key_headers

PUBLIC, TENANT, ADMIN, FRAMEWORK = "public", "tenant", "admin", "framework"

EXPECTED: dict[tuple[str, str], str] = {
    ("GET", "/"): PUBLIC,
    ("GET", "/health"): PUBLIC,
    ("GET", "/ready"): PUBLIC,
    ("GET", "/health/deep"): PUBLIC,
    ("GET", "/docs"): FRAMEWORK,
    ("GET", "/docs/oauth2-redirect"): FRAMEWORK,
    ("GET", "/openapi.json"): FRAMEWORK,
    ("GET", "/redoc"): FRAMEWORK,
    ("POST", "/api/v1/auth/google-callback"): PUBLIC,
    ("GET", "/api/v1/auth/me"): TENANT,
    ("POST", "/api/v1/auth/regenerate-key"): TENANT,
    ("GET", "/api/v1/auth/users"): ADMIN,
    ("POST", "/api/v1/track"): TENANT,
    ("POST", "/api/v1/track-and-analyze"): TENANT,
    ("POST", "/api/v1/analyze"): TENANT,
    ("POST", "/api/v1/analyze/v2"): TENANT,
    ("POST", "/api/v1/diagnose"): TENANT,
    ("GET", "/api/v1/inferences"): TENANT,
    ("GET", "/api/v1/inferences/export/csv"): TENANT,
    ("GET", "/api/v1/inferences/grouped/by-question"): TENANT,
    ("GET", "/api/v1/inferences/{request_id}"): TENANT,
    ("DELETE", "/api/v1/inferences/{request_id}"): TENANT,
    ("DELETE", "/api/v1/inferences"): TENANT,
    ("POST", "/api/v1/monitor"): TENANT,
    ("GET", "/api/v1/monitor/status"): TENANT,
    ("GET", "/api/v1/monitor/model-info"): TENANT,
    ("GET", "/api/v1/monitor/calibration"): ADMIN,
    ("GET", "/api/v1/monitor/signal-logs"): ADMIN,
    ("POST", "/api/v1/feedback/{request_id}"): TENANT,
    ("GET", "/api/v1/trend"): TENANT,
    ("GET", "/api/v1/clusters"): TENANT,
    ("DELETE", "/api/v1/clusters/reset"): TENANT,
    ("POST", "/api/v1/telemetry"): PUBLIC,
    ("GET", "/api/v1/analytics/usage"): ADMIN,
    ("GET", "/api/v1/analytics/model-performance"): ADMIN,
    ("GET", "/api/v1/analytics/calibration"): ADMIN,
    ("GET", "/api/v1/analytics/question-breakdown"): ADMIN,
    ("GET", "/api/v1/analytics/paper-metrics"): ADMIN,
    ("GET", "/api/v1/analytics/sdk-telemetry"): ADMIN,
    ("GET", "/api/v1/admin/guard/config"): ADMIN,
    ("POST", "/api/v1/admin/guard/config"): ADMIN,
    ("POST", "/api/v1/notifications/digest"): TENANT,
    ("POST", "/api/v1/playground"): TENANT,
    ("GET", "/api/v1/flags"): ADMIN,
    ("POST", "/api/v1/flags/{event_id}/label"): ADMIN,
    ("GET", "/api/v1/flags/export"): ADMIN,
    ("GET", "/api/v1/flags/hard-positives/stats"): ADMIN,
    ("GET", "/api/v1/flags/hard-positives/export"): ADMIN,
    ("POST", "/api/v1/community/feedback"): PUBLIC,
    ("GET", "/api/v1/community/stats"): PUBLIC,
    ("GET", "/api/v1/community/export"): ADMIN,
}

PROTECTED = sorted(k for k, v in EXPECTED.items() if v in (TENANT, ADMIN))
ADMIN_ONLY = sorted(k for k, v in EXPECTED.items() if v == ADMIN)


def _application_routes(app) -> set[tuple[str, str]]:
    found = set()
    for route in app.routes:
        for method in (getattr(route, "methods", None) or set()) - {"HEAD", "OPTIONS"}:
            found.add((method, route.path))
    return found


def _concrete(path: str) -> str:
    return path.replace("{request_id}", "does-not-exist").replace("{event_id}", "does-not-exist")


def declared_policies(app) -> dict[tuple[str, str], str | None]:
    """Read each route's policy from its dependency tree. None = no policy declared."""
    from app import auth_guard

    names = {auth_guard.public: PUBLIC, auth_guard.require_tenant: TENANT,
             auth_guard.require_platform_admin: ADMIN}

    def walk(dependant) -> set[str]:
        hits = set()
        for dep in getattr(dependant, "dependencies", []):
            if dep.call in names:
                hits.add(names[dep.call])
            hits |= walk(dep)
        return hits

    out: dict[tuple[str, str], str | None] = {}
    for route in app.routes:
        methods = (getattr(route, "methods", None) or set()) - {"HEAD", "OPTIONS"}
        dependant = getattr(route, "dependant", None)
        for method in methods:
            if dependant is None:
                out[(method, route.path)] = FRAMEWORK
                continue
            hits = walk(dependant)
            out[(method, route.path)] = next(iter(hits)) if len(hits) == 1 else None
    return out


def test_every_route_is_listed(client):
    from app.main import app

    actual = _application_routes(app)
    assert actual - set(EXPECTED) == set(), "routes with no entry in the security matrix"
    assert set(EXPECTED) - actual == set(), "matrix entries for routes that no longer exist"


def test_every_route_declares_its_policy(client):
    from app.main import app

    declared = declared_policies(app)
    wrong = {k: (declared.get(k), v) for k, v in EXPECTED.items() if declared.get(k) != v}
    assert wrong == {}, f"(declared, expected) differs: {wrong}"


def test_a_route_without_a_policy_is_detected():
    from fastapi import FastAPI

    probe = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @probe.get("/forgotten")
    def forgotten() -> dict:
        return {}

    assert declared_policies(probe) == {("GET", "/forgotten"): None}


@pytest.mark.parametrize("method,path", PROTECTED)
def test_no_credential_is_401(anon, method, path):
    response = anon.request(method, _concrete(path), json={})
    assert response.status_code == 401, f"{method} {path} answered {response.status_code} without a credential"


@pytest.mark.parametrize("method,path", PROTECTED)
def test_unknown_key_is_401(client, method, path):
    response = client.request(method, _concrete(path), json={}, headers={"X-API-Key": "fie-not-a-real-key00"})
    assert response.status_code == 401


@pytest.mark.parametrize("method,path", ADMIN_ONLY)
def test_tenant_is_403_on_admin_routes(a, method, path):
    response = a.request(method, _concrete(path), json={"label": "true_positive"})
    assert response.status_code == 403, f"{method} {path} answered {response.status_code} to a non-admin"


@pytest.mark.parametrize("path", ["/", "/health", "/health/deep", "/api/v1/community/stats"])
def test_public_reads_stay_public(anon, monkeypatch, path):
    import engine.encoder as encoder

    monkeypatch.setattr(encoder, "get_encoder", lambda: type("E", (), {"available": True})())
    assert anon.get(path).status_code == 200


def test_ready_stays_public(anon):
    assert anon.get("/ready").status_code in (200, 503)


def test_public_writes_stay_public(anon):
    assert anon.post("/api/v1/telemetry", json={"event": "suite"}).status_code == 200
    r = anon.post("/api/v1/community/feedback", json={"prompt": "hello there", "kind": "false_positive"})
    assert r.status_code == 200


def test_authenticated_tenant_reaches_tenant_routes(a):
    assert a.get("/api/v1/inferences").status_code == 200
    assert a.get("/api/v1/trend").status_code == 200
    assert a.get("/api/v1/clusters").status_code == 200
    assert a.get("/api/v1/monitor/model-info").status_code == 200
    assert a.get("/api/v1/auth/me").json()["tenant_id"] == TENANT_A["tenant_id"]


def test_openapi_schema_still_builds(client):
    assert client.get("/openapi.json").status_code == 200
    assert key_headers(TENANT_A)["X-API-Key"] not in client.get("/openapi.json").text
