from types import SimpleNamespace


def test_list_inferences_returns_only_current_tenant_records(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")

    from app.routes import list_inferences
    from app.schemas import InferenceRequest
    import app.routes as routes
    from datetime import datetime

    tenant_record = InferenceRequest(
        request_id="req-1",
        tenant_id="tenant-a",
        timestamp=datetime.utcnow(),
        model_name="m1",
        model_version="v1",
        temperature=0.0,
        latency_ms=1.0,
        input_text="hello",
        output_text="world",
    )

    monkeypatch.setattr(
        routes,
        "require_user",
        lambda authorization, x_api_key: {"tenant_id": "tenant-a", "is_admin": False},
    )
    monkeypatch.setattr(routes, "get_inferences_for_tenant", lambda tenant_id: [tenant_record])

    result = list_inferences(authorization="Bearer token", x_api_key=None)

    assert len(result) == 1
    assert result[0].tenant_id == "tenant-a"


def test_list_inferences_returns_all_records_for_admin(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")

    from app.routes import list_inferences
    from app.schemas import InferenceRequest
    import app.routes as routes
    from datetime import datetime

    record = InferenceRequest(
        request_id="req-1",
        tenant_id="tenant-a",
        timestamp=datetime.utcnow(),
        model_name="m1",
        model_version="v1",
        temperature=0.0,
        latency_ms=1.0,
        input_text="hello",
        output_text="world",
    )

    monkeypatch.setattr(
        routes,
        "require_user",
        lambda authorization, x_api_key: {"tenant_id": "admin", "is_admin": True},
    )
    monkeypatch.setattr(routes, "get_all_inferences_admin", lambda: [record])

    result = list_inferences(authorization="Bearer token", x_api_key=None)

    assert len(result) == 1
    assert result[0].tenant_id == "tenant-a"


def test_monitor_hides_internal_explanation_for_non_admin(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")

    from app.routes import monitor
    from app.schemas import FailureSignalVector, JuryVerdict, MonitorRequest
    import app.routes as routes
    import engine.groq_service as groq_service

    captured = {}

    class FakeGroqService:
        def fan_out(self, prompt: str):
            return []

    monkeypatch.setattr(routes.settings, "groq_enabled", True)
    monkeypatch.setattr(routes.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(groq_service, "get_groq_service", lambda: FakeGroqService())
    monkeypatch.setattr(
        routes,
        "_build_failure_signal",
        lambda outputs: FailureSignalVector(
            agreement_score=1.0,
            fsd_score=0.0,
            answer_counts={"ok": 1},
            entropy_score=0.0,
            ensemble_disagreement=False,
            ensemble_similarity=1.0,
            high_failure_risk=False,
        ),
    )
    monkeypatch.setattr(routes, "label_failure_archetype", lambda signal: "STABLE")
    monkeypatch.setattr(routes, "compute_embedding_distance", lambda p, s: {"embedding_distance": 0.0})
    monkeypatch.setattr(routes.archetype_registry, "assign", lambda signal: {})
    monkeypatch.setattr(routes.evolution_tracker, "record", lambda signal: None)
    monkeypatch.setattr(routes, "resolve_user", lambda authorization, x_api_key: {"tenant_id": "tenant-a", "is_admin": False})
    monkeypatch.setattr(
        routes.failure_agent,
        "run_diagnostic",
        lambda request: SimpleNamespace(
            jury=JuryVerdict(
                verdicts=[],
                primary_verdict=None,
                jury_confidence=0.0,
                failure_summary="No diagnostic agents reached a conclusion.",
            ),
            explanation_internal=None,
            explanation_external=None,
        ),
    )
    monkeypatch.setattr(
        routes,
        "save_inference",
        lambda inference: captured.setdefault("tenant_id", inference.tenant_id) or True,
    )

    response = monitor(
        MonitorRequest(
            prompt="What is the capital of France?",
            primary_output="Paris",
            primary_model_name="user-gpt4",
            run_full_jury=True,
        ),
        authorization="Bearer token",
        x_api_key=None,
    )

    assert captured["tenant_id"] == "tenant-a"
    assert response.explanation_external is not None
    assert response.explanation_internal is None


def test_monitor_keeps_internal_explanation_for_admin(monkeypatch):
    monkeypatch.setenv("DEBUG", "false")

    from app.routes import monitor
    from app.schemas import FailureSignalVector, JuryVerdict, MonitorRequest
    import app.routes as routes
    import engine.groq_service as groq_service

    class FakeGroqService:
        def fan_out(self, prompt: str):
            return []

    monkeypatch.setattr(routes.settings, "groq_enabled", True)
    monkeypatch.setattr(routes.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(groq_service, "get_groq_service", lambda: FakeGroqService())
    monkeypatch.setattr(
        routes,
        "_build_failure_signal",
        lambda outputs: FailureSignalVector(
            agreement_score=1.0,
            fsd_score=0.0,
            answer_counts={"ok": 1},
            entropy_score=0.0,
            ensemble_disagreement=False,
            ensemble_similarity=1.0,
            high_failure_risk=False,
        ),
    )
    monkeypatch.setattr(routes, "label_failure_archetype", lambda signal: "STABLE")
    monkeypatch.setattr(routes, "compute_embedding_distance", lambda p, s: {"embedding_distance": 0.0})
    monkeypatch.setattr(routes.archetype_registry, "assign", lambda signal: {})
    monkeypatch.setattr(routes.evolution_tracker, "record", lambda signal: None)
    monkeypatch.setattr(routes, "resolve_user", lambda authorization, x_api_key: {"tenant_id": "tenant-a", "is_admin": True})
    monkeypatch.setattr(
        routes.failure_agent,
        "run_diagnostic",
        lambda request: SimpleNamespace(
            jury=JuryVerdict(
                verdicts=[],
                primary_verdict=None,
                jury_confidence=0.0,
                failure_summary="No diagnostic agents reached a conclusion.",
            ),
            explanation_internal=None,
            explanation_external=None,
        ),
    )
    monkeypatch.setattr(routes, "save_inference", lambda inference: True)

    response = monitor(
        MonitorRequest(
            prompt="What is the capital of France?",
            primary_output="Paris",
            primary_model_name="user-gpt4",
            run_full_jury=True,
        ),
        authorization="Bearer token",
        x_api_key=None,
    )

    assert response.explanation_external is not None
    assert response.explanation_internal is not None
