"""I-8, I-13, I-17 / attacks 8a, 8c, 13, 14: what a response, an error or an outbound call may reveal."""
from __future__ import annotations

import socket

import pytest

from .conftest import inference_ids
from .fakes import TENANT_A, TENANT_B, monitor_body

SECRET_DETAIL = "ZQ-INTERNAL-DETAIL-do-not-return"


# ── No existence oracle ───────────────────────────────────────────────────────

@pytest.mark.parametrize("method,template,body", [
    ("GET", "/api/v1/inferences/{id}", None),
    ("DELETE", "/api/v1/inferences/{id}", None),
    ("POST", "/api/v1/feedback/{id}", {"is_correct": True}),
])
def test_no_existence_oracle_on_inference_id(a, b, method, template, body):
    assert a.post("/api/v1/monitor", json=monitor_body("hello there", "hi")).status_code == 200
    real_id = inference_ids(a)[0]
    missing_id = "0" * len(real_id)
    kwargs = {"json": body} if body is not None else {}
    theirs = b.request(method, template.format(id=real_id), **kwargs)
    nothing = b.request(method, template.format(id=missing_id), **kwargs)
    assert theirs.status_code == nothing.status_code == 404
    assert theirs.text.replace(real_id, "<id>") == nothing.text.replace(missing_id, "<id>")
    assert inference_ids(a) == [real_id]


# ── Errors carry no internals ─────────────────────────────────────────────────

@pytest.mark.parametrize("path,collection", [
    ("/api/v1/analytics/usage", "signal_logs"),
    ("/api/v1/analytics/model-performance", "signal_logs"),
    ("/api/v1/analytics/calibration", "signal_logs"),
    ("/api/v1/analytics/question-breakdown", "signal_logs"),
    ("/api/v1/analytics/paper-metrics", "signal_logs"),
    ("/api/v1/analytics/sdk-telemetry", "sdk_telemetry"),
    ("/api/v1/flags", "flagged_events"),
    ("/api/v1/flags/export", "flagged_events"),
    ("/api/v1/monitor/signal-logs", "signal_logs"),
    ("/api/v1/monitor/calibration", "signal_logs"),
])
def test_admin_route_errors_are_generic(admin, fakedb, path, collection):
    fakedb[collection].fail_with = RuntimeError(SECRET_DETAIL)
    response = admin.get(path)
    assert response.status_code in (200, 500, 503)
    assert SECRET_DETAIL not in response.text


def test_tenant_route_errors_are_generic(a, fakedb):
    fakedb["inferences"].fail_with = RuntimeError(SECRET_DETAIL)
    for response in (a.post("/api/v1/track", json={"request_id": "r", "timestamp": "2026-01-01T00:00:00",
                                                   "model_name": "m", "model_version": "v", "temperature": 0,
                                                   "latency_ms": 1, "input_text": "p", "output_text": "o"}),
                     a.post("/api/v1/notifications/digest")):
        assert SECRET_DETAIL not in response.text


# ── /health/deep (N12) ────────────────────────────────────────────────────────

@pytest.fixture
def probes(monkeypatch):
    import engine.encoder as encoder
    import engine.groq_service as groq_service

    calls = []

    class _Groq:
        _api_key = "suite-key"

        def _call_single_model(self, *args, **kwargs):
            calls.append(args)
            return type("R", (), {"success": True, "latency_ms": 1.0, "error": ""})()

    monkeypatch.setattr(groq_service, "get_groq_service", lambda: _Groq())
    monkeypatch.setattr(encoder, "get_encoder", lambda: type("E", (), {"available": True})())
    return calls


def test_health_deep_anonymous_makes_no_outbound_call(anon, probes, fakedb):
    fakedb.ping_error = RuntimeError(SECRET_DETAIL)
    response = anon.get("/health/deep")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "version", "components"}
    assert {"mongodb", "groq", "faiss", "encoder", "xgboost", "detector"} <= set(body["components"])
    assert "mode" in body["components"]["detector"] and "status" in body["components"]["detector"]
    assert probes == [], "an anonymous health check spent provider quota"
    assert SECRET_DETAIL not in response.text
    assert body["components"]["mongodb"]["status"] == "down"


def test_health_deep_gives_detail_to_a_platform_admin(admin, probes, fakedb):
    fakedb.ping_error = RuntimeError(SECRET_DETAIL)
    response = admin.get("/health/deep")
    assert response.status_code == 200
    assert len(probes) == 1
    assert SECRET_DETAIL[:20] in response.text


def test_health_deep_treats_a_tenant_like_an_anonymous_caller(a, probes):
    assert a.get("/health/deep").status_code == 200
    assert probes == []


# ── Request id (DL-14) ────────────────────────────────────────────────────────

@pytest.mark.parametrize("supplied", ["x" * 300, "has space", "new\tline", "<script>", ""])
def test_unsafe_request_id_is_replaced(client, supplied):
    echoed = client.get("/health", headers={"X-Request-ID": supplied}).headers["X-Request-ID"]
    assert echoed != supplied and 1 <= len(echoed) <= 64 and echoed.replace("-", "").replace("_", "").isalnum()


def test_safe_request_id_is_kept(client):
    assert client.get("/health", headers={"X-Request-ID": "abc_DEF-123"}).headers["X-Request-ID"] == "abc_DEF-123"


# ── Playground custom endpoint (N11) ──────────────────────────────────────────

@pytest.fixture
def outbound(monkeypatch):
    import app.routes.playground as playground

    sent = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "custom model answer"}}]}

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(playground._http, "post", post)
    return sent


def _playground(actor, endpoint: str):
    return actor.post("/api/v1/playground", json={"prompt": "hello", "custom_endpoint": endpoint,
                                                  "custom_api_key": "k"})


@pytest.mark.parametrize("endpoint", [
    "http://api.example.com/v1/chat/completions",
    "https://localhost/v1",
    "https://127.0.0.1/v1",
    "https://127.0.0.1:8080/v1",
    "https://[::1]/v1",
    "https://0.0.0.0/v1",
    "https://10.1.2.3/v1",
    "https://172.16.0.9/v1",
    "https://192.168.1.1/v1",
    "https://169.254.169.254/latest/meta-data/",
    "https://2130706433/v1",
    "https://0x7f000001/v1",
    "https://[::ffff:10.0.0.1]/v1",
    "https://[fd00::1]/v1",
    "https://user:pass@127.0.0.1/v1",
    "ftp://example.com/x",
    "file:///etc/passwd",
    "https:///no-host",
    "gopher://example.com/",
])
def test_playground_blocks_internal_endpoints(a, outbound, endpoint):
    response = _playground(a, endpoint)
    assert outbound == [], f"the server sent a request to {endpoint}"
    assert response.status_code in (200, 400, 422)


def test_playground_blocks_names_that_resolve_inside(a, outbound, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 443))])
    _playground(a, "https://looks-public.example.com/v1")
    assert outbound == []


def test_playground_blocks_names_that_do_not_resolve(a, outbound, monkeypatch):
    def fail(host, *args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    _playground(a, "https://nowhere.invalid/v1")
    assert outbound == []


def test_playground_allows_a_public_https_endpoint_without_following_redirects(a, outbound, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda host, *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    response = _playground(a, "https://api.example.com/v1/chat/completions")
    assert response.status_code == 200
    ((url, kwargs),) = outbound
    assert url == "https://api.example.com/v1/chat/completions"
    assert kwargs.get("allow_redirects") is False
    assert response.json()["raw_response"] == "custom model answer"


# ── Rate limiting (S12) ───────────────────────────────────────────────────────

def test_rate_limit_key_is_tenant_for_authenticated(client):
    from types import SimpleNamespace
    from app.auth_guard import Principal
    from app.limiter import rate_key

    def request_for(principal):
        return SimpleNamespace(state=SimpleNamespace(principal=principal), client=SimpleNamespace(host="203.0.113.9"),
                               headers={}, scope={"client": ("203.0.113.9", 1)})

    pa = Principal(tenant_id=TENANT_A["tenant_id"], subject="s", role="tenant", credential_kind="api_key")
    pb = Principal(tenant_id=TENANT_B["tenant_id"], subject="s", role="tenant", credential_kind="api_key")
    assert rate_key(request_for(pa)) != rate_key(request_for(pb))
    assert rate_key(request_for(pa)) == rate_key(request_for(pa))
    assert TENANT_A["tenant_id"] not in rate_key(request_for(pa))
    anonymous = SimpleNamespace(state=SimpleNamespace(), client=SimpleNamespace(host="203.0.113.9"), headers={},
                                scope={"client": ("203.0.113.9", 1)})
    assert rate_key(anonymous) == "203.0.113.9"


def test_two_tenants_behind_one_address_have_independent_limits(fakedb):
    """One shared source address, as behind a proxy: A exhausting its limit must not block B."""
    from fastapi import Depends, FastAPI, Request
    from fastapi.testclient import TestClient
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from app.auth_guard import Principal, require_tenant
    from app.limiter import limiter, rate_limit
    from .fakes import key_headers

    # The limiter resolves the handler's annotations in this module's globals.
    globals().update(Request=Request, Principal=Principal, Depends=Depends)

    probe = FastAPI()
    probe.state.limiter = limiter
    probe.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @probe.get("/limited")
    @rate_limit("2/minute")
    def limited(request: Request, principal: Principal = Depends(require_tenant)) -> dict:
        return {"tenant": "ok"}

    http = TestClient(probe)
    statuses_a = [http.get("/limited", headers=key_headers(TENANT_A)).status_code for _ in range(3)]
    statuses_b = [http.get("/limited", headers=key_headers(TENANT_B)).status_code for _ in range(2)]
    assert statuses_a == [200, 200, 429]
    assert statuses_b == [200, 200]


def test_monitor_keeps_its_rate_limit(a):
    import inspect
    import app.routes.monitor as routes

    assert '@rate_limit("60/minute")' in inspect.getsource(routes)
