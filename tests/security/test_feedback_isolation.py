"""I-6, I-9 / attacks 3a–3d: feedback is recorded, and changes nothing outside its tenant."""
from __future__ import annotations

import time

from .conftest import inference_ids
from .fakes import MARK, TENANT_A, TENANT_B, monitor_body, track_body


def _seed_labels(fakedb, n: int = 60) -> None:
    """Enough labelled history for a recalibration to run if anything triggers one."""
    for i in range(n):
        fakedb["signal_logs"].insert_one({
            "log_id": f"seed-{i}", "request_id": f"seed-req-{i}", "tenant_id": TENANT_B["tenant_id"],
            "feedback_received": True, "question_type": "FACTUAL",
            "classifier_probability": 0.05 + 0.9 * (i / n), "fie_was_correct": i % 3 != 0,
        })


def _monitor_then_feedback(actor, key: str, correct: bool = False):
    assert actor.post("/api/v1/monitor", json=monitor_body(f"{MARK[key]['prompt']} what is it",
                                                           MARK[key]["answer"])).status_code == 200
    request_id = inference_ids(actor)[0]
    body = {"is_correct": correct}
    if not correct:
        body["correct_answer"] = MARK[key]["fix"]
    response = actor.post(f"/api/v1/feedback/{request_id}", json=body)
    assert response.status_code == 200, response.text
    return request_id


def _wait_for_background_work(seconds: float = 1.5) -> None:
    import engine.fie_config as fie_config

    deadline = time.time() + seconds
    while time.time() < deadline and fie_config.get_config_version() == "default":
        time.sleep(0.05)


def test_feedback_does_not_change_thresholds(a, b, fakedb):
    """Attack 3a."""
    import engine.fie_config as fie_config

    _seed_labels(fakedb)
    before_info = b.get("/api/v1/monitor/model-info").json()
    before = fie_config.get_all_thresholds()
    _monitor_then_feedback(a, "a")
    _wait_for_background_work()
    assert fie_config.get_all_thresholds() == before
    assert fie_config.get_config_version() == "default"
    assert fakedb["fie_config"].docs == [], "feedback wrote platform configuration"
    after_info = b.get("/api/v1/monitor/model-info").json()
    assert after_info["thresholds_per_type"] == before_info["thresholds_per_type"]
    assert after_info["config_version"] == before_info["config_version"]


def test_feedback_is_still_recorded(a, fakedb):
    request_id = _monitor_then_feedback(a, "a")
    (doc,) = fakedb["feedback"].docs
    assert doc["request_id"] == request_id and doc["tenant_id"] == TENANT_A["tenant_id"]
    assert doc["correct_answer"] == MARK["a"]["fix"]
    (log,) = [d for d in fakedb["signal_logs"].docs if d.get("request_id") == request_id]
    assert log["feedback_received"] is True and log["tenant_id"] == TENANT_A["tenant_id"]


def test_feedback_does_not_start_retraining(a, fakedb, monkeypatch):
    import engine.retraining.buffer as buffer

    started = []
    monkeypatch.setattr(buffer, "RETRAIN_THRESHOLD", 1)
    monkeypatch.setattr(buffer, "_run_retrain", lambda: started.append("retrain"))
    _monitor_then_feedback(a, "a")
    time.sleep(0.2)
    assert started == []
    assert {d.get("tenant_id") for d in fakedb["retraining_buffer"].docs} == {TENANT_A["tenant_id"]}


def test_recalibration_preserves_attack_thresholds(fakedb):
    """S6: a deliberate recalibration must not drop the operator's guard overrides."""
    import engine.fie_config as fie_config

    _seed_labels(fakedb)
    fakedb["fie_config"].insert_one({"_id": "thresholds", "attack_thresholds": {"PROMPT_INJECTION": 0.71},
                                     "scan_threshold": 0.5})
    result = fie_config.recalibrate()
    assert result["status"] == "ok"
    (doc,) = fakedb["fie_config"].docs
    assert doc["attack_thresholds"] == {"PROMPT_INJECTION": 0.71}
    assert doc["version"].startswith("calibrated-")


def test_manual_recalibration_is_recorded(fakedb, events):
    import engine.fie_config as fie_config

    _seed_labels(fakedb)
    fie_config.recalibrate()
    (event,) = events.named("platform.recalibration")
    assert event["outcome"] == "changed"


def test_signal_log_lookup_is_tenant_scoped(a, fakedb):
    """N13: a log that belongs to B is not labelled because A used the same request id."""
    fakedb["inferences"].indexes.clear()
    fakedb["signal_logs"].insert_one({"log_id": "b-log", "request_id": "shared-1",
                                      "tenant_id": TENANT_B["tenant_id"], "feedback_received": False,
                                      "high_failure_risk": True, "fix_applied": False})
    body = track_body("shared-1", "p", "o", tenant_id=TENANT_A["tenant_id"])
    assert a.post("/api/v1/track", json=body).status_code == 200
    assert a.post("/api/v1/feedback/shared-1", json={"is_correct": True}).status_code == 200
    (log,) = fakedb["signal_logs"].docs
    assert log["feedback_received"] is False, "B's signal log was labelled by A"
    assert fakedb["retraining_buffer"].docs == []


def test_admin_cannot_write_feedback_cross_tenant(a, admin, fakedb):
    """N8."""
    assert a.post("/api/v1/monitor", json=monitor_body("what is it", "something")).status_code == 200
    request_id = inference_ids(a)[0]
    response = admin.post(f"/api/v1/feedback/{request_id}",
                          json={"is_correct": False, "correct_answer": "admin correction"})
    assert response.status_code == 404
    assert fakedb["feedback"].docs == [] and fakedb["ground_truth_cache"].docs == []
    assert all(not d.get("feedback_received") for d in fakedb["signal_logs"].docs)


def test_feedback_on_another_tenants_record_changes_nothing(a, b, fakedb):
    assert a.post("/api/v1/monitor", json=monitor_body("what is it", "something")).status_code == 200
    request_id = inference_ids(a)[0]
    response = b.post(f"/api/v1/feedback/{request_id}",
                      json={"is_correct": False, "correct_answer": MARK["b"]["fix"]})
    assert response.status_code == 404
    assert fakedb["feedback"].docs == [] and fakedb["ground_truth_cache"].docs == []


def test_signal_log_failure_does_not_lose_the_feedback(a, fakedb):
    assert a.post("/api/v1/monitor", json=monitor_body("what is it", "something")).status_code == 200
    request_id = inference_ids(a)[0]
    fakedb["signal_logs"].fail_with = RuntimeError("ZQ-INTERNAL-DETAIL")
    response = a.post(f"/api/v1/feedback/{request_id}", json={"is_correct": True})
    assert response.status_code == 200 and "ZQ-INTERNAL-DETAIL" not in response.text
    assert [d["tenant_id"] for d in fakedb["feedback"].docs] == [TENANT_A["tenant_id"]]


def test_automatic_switches_are_off_by_default(monkeypatch):
    import engine.fie_config as fie_config
    import engine.retraining.buffer as buffer

    started = []
    monkeypatch.setattr(fie_config, "recalibrate", lambda: started.append("recalibrate"))
    monkeypatch.setattr(buffer, "_run_retrain", lambda: started.append("retrain"))
    fie_config.maybe_recalibrate()
    buffer.maybe_trigger_retrain(10_000)
    time.sleep(0.2)
    assert started == []
