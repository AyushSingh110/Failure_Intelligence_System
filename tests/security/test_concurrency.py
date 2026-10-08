"""
Two tenants hammer the server at the same time. Nothing may land in the wrong scope.

Scope is passed explicitly through every helper, so there is no per-thread or
per-task "current tenant" that a pool could lose. This test is the evidence.
"""
from __future__ import annotations

import json
import threading

from .conftest import squash
from .fakes import MARK, TENANT_A, TENANT_B, key_headers, monitor_body, track_body

WORKERS_PER_TENANT = 8
ROUNDS = 4


def _run(client_factory, user: dict, key: str, worker: int, errors: list) -> None:
    http = client_factory()
    try:
        for n in range(ROUNDS):
            body = monitor_body(f"{MARK[key]['prompt']} w{worker} r{n}", f"{MARK[key]['answer']} w{worker} r{n}",
                                session_id="shared-session", conversation_id="shared-conversation")
            r = http.post("/api/v1/monitor", json=body, headers=key_headers(user))
            if r.status_code != 200:
                errors.append(f"{key} monitor {r.status_code} {r.text[:120]}")
            r = http.post("/api/v1/track", headers=key_headers(user),
                          json=track_body(f"same-id-{worker}-{n}", MARK[key]["prompt"], MARK[key]["answer"]))
            if r.status_code not in (200, 409):
                errors.append(f"{key} track {r.status_code}")
    except Exception as exc:   # pragma: no cover - reported through `errors`
        errors.append(f"{key} worker {worker}: {type(exc).__name__}: {exc}")


def test_concurrent_tenants_never_mix(fakedb):
    from fastapi.testclient import TestClient
    from app.main import app

    fakedb["inferences"].indexes.clear()   # both tenants deliberately reuse the same request ids
    factory = lambda: TestClient(app, raise_server_exceptions=False)  # noqa: E731
    errors: list[str] = []
    threads = [threading.Thread(target=_run, args=(factory, user, key, w, errors))
               for w in range(WORKERS_PER_TENANT) for user, key in ((TENANT_A, "a"), (TENANT_B, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert errors == []

    own = {TENANT_A["tenant_id"]: squash(MARK["a"]["prompt"]), TENANT_B["tenant_id"]: squash(MARK["b"]["prompt"])}
    other = {TENANT_A["tenant_id"]: squash(MARK["b"]["prompt"]), TENANT_B["tenant_id"]: squash(MARK["a"]["prompt"])}

    expected = WORKERS_PER_TENANT * ROUNDS
    for tenant in own:
        records = [d for d in fakedb["inferences"].docs if d.get("tenant_id") == tenant]
        assert len(records) == 2 * expected, f"{tenant}: {len(records)} inference records"
    for name in ("inferences", "signal_logs", "session_context", "conversation_turns", "model_extraction_tracking"):
        for doc in fakedb[name].docs:
            tenant = doc.get("tenant_id")
            assert tenant in own, f"{name}: row without a tenant: {str(doc)[:120]}"
            assert other[tenant] not in squash(json.dumps(doc, default=str)), f"{name}: foreign text in {tenant}'s row"
    assert len(fakedb["session_context"].docs) == 2
    for user in (TENANT_A, TENANT_B):
        ids = [d["request_id"] for d in fakedb["inferences"].docs if d["tenant_id"] == user["tenant_id"]]
        assert len(ids) == len(set(ids))

    http = factory()
    for user, key, foreign in ((TENANT_A, "a", "b"), (TENANT_B, "b", "a")):
        clusters = http.get("/api/v1/clusters", headers=key_headers(user)).text
        trend = http.get("/api/v1/trend", headers=key_headers(user)).json()
        assert squash(MARK[foreign]["answer"]) not in squash(clusters)
        assert trend["signals_recorded"] == expected
        listing = http.get("/api/v1/inferences?limit=500", headers=key_headers(user)).text
        assert squash(MARK[foreign]["prompt"]) not in squash(listing)


def test_concurrent_writes_to_one_request_id_keep_both_owners_intact(fakedb):
    from fastapi.testclient import TestClient
    from app.main import app

    fakedb["inferences"].indexes.clear()
    barrier = threading.Barrier(2)
    results = {}

    def write(user, key):
        http = TestClient(app, raise_server_exceptions=False)
        barrier.wait(timeout=30)
        for n in range(25):
            http.post("/api/v1/track", headers=key_headers(user),
                      json=track_body("contested", f"{MARK[key]['prompt']} {n}", MARK[key]["answer"]))
        results[key] = http.get("/api/v1/inferences/contested", headers=key_headers(user)).json()

    threads = [threading.Thread(target=write, args=(TENANT_A, "a")),
               threading.Thread(target=write, args=(TENANT_B, "b"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert MARK["a"]["prompt"] in results["a"]["input_text"]
    assert MARK["b"]["prompt"] in results["b"]["input_text"]
    assert len([d for d in fakedb["inferences"].docs if d["request_id"] == "contested"]) == 2


def test_concurrent_cache_writes_and_reads_stay_scoped(fakedb):
    from engine.ground_truth_cache import lookup_cache, save_to_cache
    from .conftest import scope_for

    question = "Which harbour does the Zorblat ferry leave from?"
    scopes = {"a": scope_for(TENANT_A), "b": scope_for(TENANT_B)}
    wrong: list[str] = []

    def worker(key: str):
        for n in range(40):
            save_to_cache(question, f"{MARK[key]['fix']} {n}", scope=scopes[key], source_class="tenant_feedback")
            hit = lookup_cache(question, scope=scopes[key])
            if hit is None or MARK[key]["fix"] not in hit.verified_answer:
                wrong.append(f"{key}: {hit}")

    threads = [threading.Thread(target=worker, args=(k,)) for k in ("a", "b") for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert wrong == []
    assert len(fakedb["ground_truth_cache"].docs) == 2
