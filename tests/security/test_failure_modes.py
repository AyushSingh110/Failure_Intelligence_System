"""I-14: when identity, scope or a store is missing, the request fails. It is never served as global."""
from __future__ import annotations

import pytest

from .conftest import scope_for
from .fakes import PLATFORM_ADMIN, TENANT_A, TENANT_B, key_headers, monitor_body


# ── Principal and scope ───────────────────────────────────────────────────────

@pytest.mark.parametrize("tenant", ["", "anonymous", "   ", None, 42])
def test_principal_cannot_exist_without_a_real_tenant(tenant):
    from app.auth_guard import Principal

    with pytest.raises(ValueError):
        Principal(tenant_id=tenant, subject="s", role="tenant", credential_kind="api_key")


@pytest.mark.parametrize("not_a_principal", [None, "", TENANT_A["tenant_id"], {"tenant_id": "x"}, object()])
def test_scope_cannot_be_built_from_request_data(not_a_principal):
    from app.tenancy import TenantScope, TenantScopeError

    with pytest.raises(TenantScopeError):
        TenantScope(not_a_principal)


def test_scope_is_immutable_and_carries_exactly_the_principals_tenant():
    scope = scope_for(TENANT_A)
    assert scope.tenant_id == TENANT_A["tenant_id"]
    with pytest.raises(Exception):
        scope.tenant_id = TENANT_B["tenant_id"]


def test_tenant_of_accepts_only_a_scope():
    from app.tenancy import tenant_of

    assert tenant_of(scope_for(TENANT_A)) == TENANT_A["tenant_id"]
    for other in (None, "", TENANT_A["tenant_id"], {"tenant_id": TENANT_A["tenant_id"]}, object()):
        assert tenant_of(other) is None


def test_an_admin_scope_is_still_one_tenant():
    assert scope_for(PLATFORM_ADMIN).tenant_id == PLATFORM_ADMIN["tenant_id"]


def test_a_handler_called_without_a_resolved_principal_refuses(fakedb):
    """A direct call that bypasses dependency resolution must not run as anybody."""
    from fastapi import HTTPException
    from starlette.requests import Request
    from app.routes.monitor import monitor
    from app.schemas import MonitorRequest

    request = Request({"type": "http", "method": "POST", "path": "/api/v1/monitor", "headers": [],
                       "client": ("127.0.0.1", 1)})
    with pytest.raises(HTTPException) as caught:
        monitor(request, MonitorRequest(prompt="hello", primary_output="hi"))
    assert caught.value.status_code == 401
    assert fakedb["inferences"].docs == []


# ── Stores ────────────────────────────────────────────────────────────────────

def test_user_store_error_is_never_an_anonymous_request(client, fakedb):
    fakedb["users"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = client.post("/api/v1/monitor", headers=key_headers(TENANT_A), json=monitor_body("hello", "hi"))
    assert response.status_code in (401, 503)
    assert "ZQ-INTERNAL-DETAIL" not in response.text
    fakedb["users"].fail_with = None
    assert fakedb["inferences"].docs == []


def test_inference_store_error_is_not_a_silent_success(a, fakedb):
    from .fakes import track_body

    fakedb["inferences"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = a.post("/api/v1/track", json=track_body("r", "p", "o"))
    assert response.status_code == 500 and "ZQ-INTERNAL-DETAIL" not in response.text


def test_session_store_error_does_not_fail_or_cross(a, b, fakedb):
    fakedb["session_context"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    for actor in (a, b):
        response = actor.post("/api/v1/monitor", json=monitor_body("hello there", "hi", session_id="s"))
        assert response.status_code == 200 and "ZQ-INTERNAL-DETAIL" not in response.text


def test_database_fallback_mode_is_also_scoped(a, b, monkeypatch):
    """When MongoDB is down the server keeps records in memory. That store is per tenant too."""
    import storage.database as database
    from .conftest import inference_ids
    from .fakes import track_body

    monkeypatch.setattr(database, "_fallback_mode", True)
    monkeypatch.setattr(database, "_db", None)
    monkeypatch.setattr(database, "_collection", None)
    assert a.post("/api/v1/track", json=track_body("same", "from a", "o")).status_code == 200
    assert b.post("/api/v1/track", json=track_body("same", "from b", "o")).status_code == 200
    assert a.get("/api/v1/inferences/same").json()["input_text"] == "from a"
    assert b.get("/api/v1/inferences/same").json()["input_text"] == "from b"
    assert b.delete("/api/v1/inferences").json()["deleted_count"] == 1
    assert inference_ids(a) == ["same"]


def test_dependency_is_the_only_place_that_reads_credentials():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "app"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "auth_guard.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in ("X-API-Key",):
                offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id in ("verify_session_token", "get_user_by_api_key") \
                    and path.name not in ("auth.py",):
                offenders.append(f"{path.name}:{node.lineno} {node.id}")
    assert offenders == []
