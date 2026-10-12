"""I-5, I-10, I-15 / attacks 1c, 6d, 6e, 8a: stored records stay inside their tenant."""
from __future__ import annotations

import pytest

from .conftest import inference_ids
from .fakes import MARK, PLATFORM_ADMIN, TENANT_A, TENANT_B, monitor_body, track_body


def _seed(actor, key: str, n: int = 1) -> list[str]:
    for i in range(n):
        body = monitor_body(f"{MARK[key]['prompt']} question {i}", f"{MARK[key]['answer']} {i}")
        assert actor.post("/api/v1/monitor", json=body).status_code == 200
    return inference_ids(actor)


def test_a_tenant_sees_only_its_own_records(a, b):
    a_ids, b_ids = _seed(a, "a", 2), _seed(b, "b", 1)
    assert len(a_ids) == 2 and len(b_ids) == 1 and not set(a_ids) & set(b_ids)
    assert MARK["a"]["prompt"] in a.get("/api/v1/inferences").text
    for path in ("/api/v1/inferences", "/api/v1/inferences/export/csv", "/api/v1/inferences/grouped/by-question"):
        assert b.get(path).status_code == 200  # the marker fixture fails the test if A's data is in it


def test_read_and_delete_of_another_tenants_record_is_404(a, b):
    (a_id,) = _seed(a, "a")
    assert b.get(f"/api/v1/inferences/{a_id}").status_code == 404
    assert b.delete(f"/api/v1/inferences/{a_id}").status_code == 404
    assert a.get(f"/api/v1/inferences/{a_id}").status_code == 200


def test_clear_removes_only_the_callers_records(a, b):
    _seed(a, "a", 2)
    _seed(b, "b", 2)
    assert a.delete("/api/v1/inferences").json()["deleted_count"] == 2
    assert inference_ids(a) == [] and len(inference_ids(b)) == 2


def test_track_cannot_overwrite_other_tenant_record(a, b, fakedb):
    """N1: B reuses an id that A already holds."""
    (a_id,) = _seed(a, "a")
    response = b.post("/api/v1/track", json=track_body(a_id, MARK["b"]["prompt"], MARK["b"]["answer"]))
    assert response.status_code in (200, 409)
    mine = a.get(f"/api/v1/inferences/{a_id}")
    assert mine.status_code == 200, "A's record is gone"
    assert MARK["a"]["prompt"] in mine.json()["input_text"], "A's record was overwritten"
    owners = {d["tenant_id"] for d in fakedb["inferences"].docs if d["request_id"] == a_id}
    assert TENANT_A["tenant_id"] in owners


def test_id_collision_is_reported_without_detail_while_the_old_index_exists(a, b):
    (a_id,) = _seed(a, "a")
    response = b.post("/api/v1/track", json=track_body(a_id, "p", "o"))
    assert response.status_code == 409
    assert TENANT_A["tenant_id"] not in response.text


def test_two_tenants_can_hold_the_same_request_id_after_the_index_migration(a, b, fakedb):
    fakedb["inferences"].indexes.clear()   # the owner-run migration replaced the single-field unique index
    for actor, key in ((a, "a"), (b, "b")):
        body = track_body("shared-id", MARK[key]["prompt"], MARK[key]["answer"])
        assert actor.post("/api/v1/track", json=body).status_code == 200
    assert len([d for d in fakedb["inferences"].docs if d["request_id"] == "shared-id"]) == 2
    assert MARK["a"]["prompt"] in a.get("/api/v1/inferences/shared-id").json()["input_text"]
    assert MARK["b"]["prompt"] in b.get("/api/v1/inferences/shared-id").json()["input_text"]
    assert b.delete("/api/v1/inferences/shared-id").status_code == 200
    assert a.get("/api/v1/inferences/shared-id").status_code == 200


def test_tracking_the_same_id_twice_updates_the_callers_own_record(a, fakedb):
    assert a.post("/api/v1/track", json=track_body("mine", "first", "o")).status_code == 200
    assert a.post("/api/v1/track", json=track_body("mine", "second", "o")).status_code == 200
    docs = [d for d in fakedb["inferences"].docs if d["request_id"] == "mine"]
    assert len(docs) == 1 and docs[0]["input_text"] == "second"


def test_a_legacy_record_of_the_same_tenant_is_updated_not_duplicated(a, fakedb):
    fakedb["inferences"].insert_one({**track_body("old", "legacy", "o"), "_id": "old",
                                     "tenant_id": TENANT_A["tenant_id"]})
    assert a.post("/api/v1/track", json=track_body("old", "updated", "o")).status_code == 200
    docs = [d for d in fakedb["inferences"].docs if d["request_id"] == "old"]
    assert len(docs) == 1 and docs[0]["input_text"] == "updated"


# ── Platform admin on data routes (N7) ────────────────────────────────────────

def test_admin_default_view_is_own_tenant(a, admin):
    _seed(a, "a", 2)
    assert admin.get("/api/v1/inferences").json() == []
    assert MARK["a"]["prompt"] not in admin.get("/api/v1/inferences/export/csv").text
    assert admin.get("/api/v1/inferences/grouped/by-question").json() == {}


def test_admin_cross_tenant_read_is_explicit_and_audited(a, admin, events):
    (a_id,) = _seed(a, "a")
    assert admin.get(f"/api/v1/inferences/{a_id}").status_code == 404
    listed = admin.get("/api/v1/inferences?all_tenants=true")
    assert [r["request_id"] for r in listed.json()] == [a_id]
    assert admin.get(f"/api/v1/inferences/{a_id}?all_tenants=true").status_code == 200
    assert MARK["a"]["prompt"] in admin.get("/api/v1/inferences/export/csv?all_tenants=true").text
    assert len(events.named("admin.cross_tenant_read")) == 3


def test_all_tenants_is_refused_for_a_tenant(a, b, events):
    _seed(a, "a")
    for path in ("/api/v1/inferences", "/api/v1/inferences/export/csv",
                 "/api/v1/inferences/grouped/by-question"):
        assert b.get(f"{path}?all_tenants=true").status_code == 403
    assert events.named("authz.admin_denied")


def test_admin_clear_deletes_only_own_tenant(a, b, admin, fakedb):
    _seed(a, "a", 2)
    _seed(b, "b", 1)
    assert admin.delete("/api/v1/inferences").json()["deleted_count"] == 0
    assert admin.delete("/api/v1/inferences?all_tenants=true").json()["deleted_count"] == 0
    assert len(fakedb["inferences"].docs) == 3


def test_admin_cannot_delete_another_tenants_record(a, admin, fakedb):
    (a_id,) = _seed(a, "a")
    assert admin.delete(f"/api/v1/inferences/{a_id}").status_code == 404
    assert admin.delete(f"/api/v1/inferences/{a_id}?all_tenants=true").status_code == 404
    assert len(fakedb["inferences"].docs) == 1


# ── I-10: attribution ─────────────────────────────────────────────────────────

def test_rows_carry_tenant(a, fakedb):
    (a_id,) = _seed(a, "a")
    assert a.post(f"/api/v1/feedback/{a_id}", json={"is_correct": False,
                                                     "correct_answer": MARK["a"]["fix"]}).status_code == 200
    for name in ("inferences", "signal_logs", "feedback", "ground_truth_cache", "retraining_buffer",
                 "model_extraction_tracking"):
        docs = fakedb[name].docs
        assert docs, f"{name} is empty"
        assert {d.get("tenant_id") for d in docs} == {TENANT_A["tenant_id"]}, name


def test_requested_indexes_start_with_the_tenant(fakedb):
    import storage.database as database

    fakedb["inferences"].indexes.clear()
    database.request_indexes(fakedb["inferences"])
    keys = fakedb["inferences"].index_keys()
    assert ("tenant_id", "request_id") in keys and ("tenant_id", "timestamp") in keys
    assert ("request_id",) not in keys, "the single-field unique index must no longer be requested"
    unique = [k for k, o in fakedb["inferences"].indexes if o.get("unique")]
    assert unique == [("tenant_id", "request_id")]


# ── The scoped store itself ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, "", TENANT_A["tenant_id"], {"tenant_id": TENANT_A["tenant_id"]}])
def test_store_cannot_be_built_without_a_scope(bad):
    from app.tenancy import TenantScopeError
    from storage.tenant_store import TenantStore

    with pytest.raises(TenantScopeError):
        TenantStore(bad)


def test_store_forces_its_own_tenant_onto_every_write(fakedb):
    from app.schemas import InferenceRequest
    from storage.tenant_store import TenantStore
    from .conftest import scope_for

    store = TenantStore(scope_for(TENANT_A))
    record = InferenceRequest(**track_body("x1", "p", "o", tenant_id=TENANT_B["tenant_id"]))
    assert store.save_inference(record) is True
    store.save_feedback({"request_id": "x1", "tenant_id": TENANT_B["tenant_id"], "is_correct": True})
    for name in ("inferences", "feedback"):
        assert {d["tenant_id"] for d in fakedb[name].docs} == {TENANT_A["tenant_id"]}
    assert TenantStore(scope_for(TENANT_B)).get_inference("x1") is None
    assert TenantStore(scope_for(PLATFORM_ADMIN)).list_inferences() == []


def test_store_error_is_not_answered_from_an_unscoped_query(a, b, fakedb):
    _seed(a, "a")
    fakedb["inferences"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = b.get("/api/v1/inferences")
    assert response.status_code in (200, 500, 503)
    assert "ZQ-INTERNAL-DETAIL" not in response.text
    if response.status_code == 200:
        assert response.json() == []
