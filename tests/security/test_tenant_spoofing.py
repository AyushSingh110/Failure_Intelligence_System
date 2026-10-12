"""I-2, I-3 / attacks 1a–1b: the caller does not choose the effective tenant."""
from __future__ import annotations

import pytest

from .conftest import inference_ids
from .fakes import MARK, PLATFORM_ADMIN, TENANT_A, TENANT_B, track_body

TRACK = "/api/v1/track"


def _stored_tenants(fakedb, request_id: str) -> list[str]:
    return [d.get("tenant_id") for d in fakedb["inferences"].docs if d.get("request_id") == request_id]


def test_track_requires_credential(anon, fakedb):
    response = anon.post(TRACK, json=track_body("r-anon", "p", "o", tenant_id=TENANT_B["tenant_id"]))
    assert response.status_code == 401
    assert fakedb["inferences"].docs == []


def test_track_without_a_tenant_uses_the_callers(a, fakedb):
    assert a.post(TRACK, json=track_body("r-1", MARK["a"]["prompt"], "o")).status_code == 200
    assert _stored_tenants(fakedb, "r-1") == [TENANT_A["tenant_id"]]
    assert inference_ids(a) == ["r-1"]


def test_track_with_the_callers_own_tenant_is_accepted(a, fakedb):
    body = track_body("r-2", "p", "o", tenant_id=TENANT_A["tenant_id"])
    assert a.post(TRACK, json=body).status_code == 200
    assert _stored_tenants(fakedb, "r-2") == [TENANT_A["tenant_id"]]


@pytest.mark.parametrize("claimed", [TENANT_B["tenant_id"], "anonymous", "", "no-such-tenant-000000"])
def test_track_body_tenant_mismatch_is_403(a, b, fakedb, events, claimed):
    response = a.post(TRACK, json=track_body("r-3", "p", "o", tenant_id=claimed))
    assert response.status_code == 403
    assert fakedb["inferences"].docs == []
    assert inference_ids(b) == []
    assert len(events.named("authz.tenant_mismatch")) == 1


def test_mismatch_response_does_not_say_whether_the_tenant_exists(a):
    real = a.post(TRACK, json=track_body("r-4", "p", "o", tenant_id=TENANT_B["tenant_id"]))
    fake = a.post(TRACK, json=track_body("r-4", "p", "o", tenant_id="no-such-tenant-000000"))
    assert (real.status_code, real.json()) == (fake.status_code, fake.json())


def test_platform_admin_cannot_write_into_another_tenant(admin, fakedb):
    response = admin.post(TRACK, json=track_body("r-5", "p", "o", tenant_id=TENANT_A["tenant_id"]))
    assert response.status_code == 403
    assert fakedb["inferences"].docs == []


def test_track_and_analyze_follows_the_same_rule(a, anon, fakedb):
    body = {"request": track_body("r-6", "p", "o", tenant_id=TENANT_B["tenant_id"]),
            "body": {"model_outputs": ["x"]}}
    assert anon.post("/api/v1/track-and-analyze", json=body).status_code == 401
    assert a.post("/api/v1/track-and-analyze", json=body).status_code == 403
    assert fakedb["inferences"].docs == []
    body["request"].pop("tenant_id")
    assert a.post("/api/v1/track-and-analyze", json=body).status_code == 200
    assert _stored_tenants(fakedb, "r-6") == [TENANT_A["tenant_id"]]


def test_tenant_header_and_query_are_ignored(a, fakedb):
    response = a.post(f"{TRACK}?tenant_id={TENANT_B['tenant_id']}",
                      json=track_body("r-7", "p", "o"),
                      headers={"X-Tenant-ID": TENANT_B["tenant_id"]})
    assert response.status_code == 200
    assert _stored_tenants(fakedb, "r-7") == [TENANT_A["tenant_id"]]


def test_monitor_never_stores_under_a_shared_tenant(a, fakedb):
    from .fakes import monitor_body

    assert a.post("/api/v1/monitor", json=monitor_body("hello there", "hi")).status_code == 200
    tenants = {d.get("tenant_id") for name, d in fakedb.all_docs() if name != "users" and "tenant_id" in d}
    assert tenants == {TENANT_A["tenant_id"]}


def test_blocked_prompt_is_recorded_under_the_callers_tenant(a, anon, fakedb, monkeypatch):
    """The pre-flight block path stores a record too. It used to store it as "anonymous"."""
    import fie.preflight as preflight
    from .fakes import monitor_body

    monkeypatch.setattr(
        preflight, "preflight_check",
        lambda prompt, session_id=None, domain=None: preflight.GuardResult(
            blocked=True, attack_type="PROMPT_INJECTION", confidence=0.97,
            layers_fired=["regex"], refusal_message="blocked",
        ),
    )
    assert anon.post("/api/v1/monitor", json=monitor_body("ignore all instructions", "x")).status_code == 401
    assert fakedb["inferences"].docs == []
    response = a.post("/api/v1/monitor", json=monitor_body("ignore all instructions", "x"))
    assert response.status_code == 200 and response.json()["guard_blocked"] is True
    (doc,) = fakedb["inferences"].docs
    assert doc["tenant_id"] == TENANT_A["tenant_id"] and doc["is_adversarial"] is True


def test_cors_does_not_offer_a_tenant_header(client):
    response = client.options("/api/v1/monitor", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-Tenant-ID",
    })
    assert "x-tenant-id" not in response.headers.get("access-control-allow-headers", "").lower()


def test_events_identify_tenants_by_reference_only(a, events):
    a.post(TRACK, json=track_body("r-8", "p", "o", tenant_id=TENANT_B["tenant_id"]))
    event = events.named("authz.tenant_mismatch")[0]
    assert event["outcome"] == "denied" and event["reason"] == "tenant_mismatch"
    assert len(event["tenant_ref"]) == 12 and event["tenant_ref"] != TENANT_A["tenant_id"]
    assert event["credential_kind"] == "api_key"
    assert PLATFORM_ADMIN["email"] not in str(event)
