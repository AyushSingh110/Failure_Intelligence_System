"""AC-21: the boundary costs microseconds and no extra database round trip on tenant routes."""
from __future__ import annotations

import time

from .conftest import scope_for
from .fakes import PLATFORM_ADMIN, TENANT_A, bearer, key_headers, make_token


def _p95_ms(fn, iterations: int = 1000) -> float:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return samples[int(len(samples) * 0.95)]


def test_authentication_and_scope_cost(fakedb):
    from app.auth_guard import authenticate
    from app.tenancy import TenantScope

    token = make_token(TENANT_A)

    def by_key():
        TenantScope(authenticate(None, TENANT_A["api_key"]))

    def by_token():
        TenantScope(authenticate(f"Bearer {token}", None))

    assert _p95_ms(by_key) <= 1.0
    assert _p95_ms(by_token) <= 1.0


def test_cache_key_cost():
    from engine.ground_truth_cache import _question_id
    from engine.groq_service import _cache_key

    scope = scope_for(TENANT_A)
    question = "Who currently chairs the Zorblat harbour committee? " * 4
    assert _p95_ms(lambda: _question_id(question, scope.tenant_id)) <= 0.1
    assert _p95_ms(lambda: _cache_key("model", question, scope.tenant_id, "system message")) <= 0.1


def test_user_store_round_trips_per_request(client, fakedb):
    users = fakedb["users"]

    def lookups(headers, path="/api/v1/inferences") -> int:
        users.calls.clear()
        assert client.get(path, headers=headers).status_code == 200
        return users.calls["find_one"]

    assert lookups(key_headers(TENANT_A)) == 1                       # as before WP-002
    assert lookups(bearer(make_token(TENANT_A))) == 0                # a tenant's token needs no lookup
    assert lookups(key_headers(PLATFORM_ADMIN), "/api/v1/monitor/calibration") == 1
    assert lookups(bearer(make_token(PLATFORM_ADMIN)), "/api/v1/monitor/calibration") == 1   # the admin check


def test_scoped_queries_start_with_the_tenant(a, fakedb, monkeypatch):
    from .fakes import monitor_body

    filters: dict[str, list] = {}
    for name in ("inferences", "signal_logs", "session_context", "conversation_turns"):
        col = fakedb[name]
        for op in ("find_one", "find", "update_one", "delete_one", "delete_many"):
            original = getattr(col, op)
            monkeypatch.setattr(col, op, lambda flt=None, *args, _o=original, _n=name, **kw: (
                filters.setdefault(_n, []).append(flt), _o(flt, *args, **kw))[1])
    assert a.post("/api/v1/monitor", json=monitor_body("hello there", "hi", session_id="s",
                                                       conversation_id="c")).status_code == 200
    rid = a.get("/api/v1/inferences").json()[0]["request_id"]
    assert a.post(f"/api/v1/feedback/{rid}", json={"is_correct": True}).status_code == 200
    assert a.delete(f"/api/v1/inferences/{rid}").status_code == 200
    assert a.delete("/api/v1/inferences").status_code == 200
    assert set(filters) == {"inferences", "signal_logs", "session_context", "conversation_turns"}
    for name, seen in filters.items():
        missing = [f for f in seen if (f or {}).get("tenant_id") != TENANT_A["tenant_id"]]
        assert missing == [], f"{name}: query without the tenant: {missing}"
