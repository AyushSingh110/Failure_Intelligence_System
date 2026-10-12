"""I-12, I-15 / attacks 4c, 10: what a platform admin can do, and how it is checked and recorded."""
from __future__ import annotations

from .fakes import PLATFORM_ADMIN, TENANT_A, TENANT_B, bearer, make_token, monitor_body


def test_user_list_never_contains_api_keys(admin, client):
    for response in (admin.get("/api/v1/auth/users"),
                     client.get("/api/v1/auth/users", headers=bearer(make_token(PLATFORM_ADMIN)))):
        assert response.status_code == 200, response.text
        rows = response.json()
        assert {r["email"] for r in rows} >= {TENANT_A["email"], TENANT_B["email"]}
        assert all("api_key" not in r for r in rows)
        for user in (TENANT_A, TENANT_B, PLATFORM_ADMIN):
            assert user["api_key"] not in response.text


def test_flag_routes_work_for_a_platform_admin(admin, fakedb, monkeypatch):
    """The review queue was unreachable for everyone: its auth helper imported a name that does not exist."""
    import fie.feedback_store as feedback_store

    monkeypatch.setattr(feedback_store, "_KNOWN_ATTACK_HASHES", set())
    monkeypatch.setattr(feedback_store, "_WHITELIST_HASHES", set())
    fakedb["flagged_events"].insert_one({"id": "ev-1", "kind": "input_block", "flag_type": "PROMPT_INJECTION",
                                         "confidence": 0.9, "prompt_hash": "h" * 64, "matched": "x",
                                         "session_id": None, "timestamp": "2026-01-01T00:00:00+00:00",
                                         "label": None, "labeled_at": None})
    listed = admin.get("/api/v1/flags")
    assert listed.status_code == 200 and [e["id"] for e in listed.json()["events"]] == ["ev-1"]
    assert admin.post("/api/v1/flags/ev-1/label", json={"label": "nonsense"}).status_code == 400
    # An unknown event id changes nothing. (With the MongoDB backend the SDK's
    # apply_label() reports success for an id it did not find, so the route
    # answers 200; `fie/` is outside this package. What matters here is that no
    # hash is learned and no stored event is touched.)
    admin.post("/api/v1/flags/missing/label", json={"label": "true_positive"})
    assert feedback_store._KNOWN_ATTACK_HASHES == set() and feedback_store._WHITELIST_HASHES == set()
    assert [e["label"] for e in fakedb["flagged_events"].docs] == [None]


def test_flag_label_requires_platform_admin(a, admin, fakedb, events, monkeypatch):
    """Attack 4c / S8: a label changes verdicts for every tenant, so only the platform may set it."""
    import fie.feedback_store as feedback_store

    monkeypatch.setattr(feedback_store, "_KNOWN_ATTACK_HASHES", set())
    monkeypatch.setattr(feedback_store, "_WHITELIST_HASHES", set())
    fakedb["flagged_events"].insert_one({"id": "ev-2", "prompt_hash": "f" * 64, "label": None,
                                         "timestamp": "2026-01-01T00:00:00+00:00"})
    assert a.post("/api/v1/flags/ev-2/label", json={"label": "false_positive"}).status_code == 403
    assert feedback_store._WHITELIST_HASHES == set()
    assert admin.post("/api/v1/flags/ev-2/label", json={"label": "false_positive"}).status_code == 200
    assert feedback_store._WHITELIST_HASHES == {"f" * 64}
    (event,) = events.named("admin.flag_labelled")
    assert event["label"] == "false_positive" and event["outcome"] == "changed"


def test_guard_configuration_needs_a_current_admin_and_is_recorded(client, admin, fakedb, events):
    stale = bearer(make_token(TENANT_A, is_admin=True))
    assert client.post("/api/v1/admin/guard/config", headers=stale,
                       json={"block_enabled": False}).status_code == 403
    assert client.get("/api/v1/admin/guard/config", headers=stale).status_code == 403
    assert admin.get("/api/v1/admin/guard/config").json()["block_enabled"] is True
    assert admin.post("/api/v1/admin/guard/config", json={"block_enabled": False}).status_code == 200
    (event,) = events.named("admin.config_change")
    assert event["outcome"] == "changed" and event["block_enabled"] is False


def test_reading_all_tenants_signal_logs_is_recorded(a, admin, events):
    assert a.post("/api/v1/monitor", json=monitor_body("hello there", "hi")).status_code == 200
    response = admin.get("/api/v1/monitor/signal-logs")
    assert response.status_code == 200 and len(response.json()) == 1
    assert response.json()[0]["tenant_id"] == TENANT_A["tenant_id"]
    assert events.named("admin.cross_tenant_read")


def test_admin_analytics_cover_all_tenants_but_need_a_current_admin(a, admin, client, fakedb):
    assert a.post("/api/v1/monitor", json=monitor_body("hello there", "hi")).status_code == 200
    assert admin.get("/api/v1/analytics/usage").json()["total_requests"] == 1
    fakedb["users"].update_one({"email": PLATFORM_ADMIN["email"]}, {"$set": {"is_admin": False}})
    assert admin.get("/api/v1/analytics/usage").status_code == 403
    assert client.get("/api/v1/analytics/usage",
                      headers=bearer(make_token(PLATFORM_ADMIN))).status_code == 403


def test_internal_explanation_is_for_current_admins_only(a, admin, client, fakedb):
    body = monitor_body("hello there", "hi")
    assert a.post("/api/v1/monitor", json=body).json().get("explanation_internal") is None
    stale = bearer(make_token(TENANT_A, is_admin=True))
    assert client.post("/api/v1/monitor", headers=stale, json=body).json().get("explanation_internal") is None
