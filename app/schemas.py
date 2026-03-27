from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class MathematicalMetrics(BaseModel):
    confidence:           Optional[float] = None
    entropy:              Optional[float] = None
    logit_margin:         Optional[float] = None
    agreement_score:      Optional[float] = None
    fsd_score:            Optional[float] = None
    consistency_entropy:  Optional[float] = None
    embedding_distance:   Optional[float] = None


class InferenceRequest(BaseModel):
    request_id:    str
    tenant_id:     str = "anonymous"  # identifies which user this belongs to
    timestamp:     datetime
    model_name:    str
    model_version: str
    temperature:   float
    latency_ms:    float

    input_text:   str
    output_text:  str
    ground_truth: Optional[str]  = None
    is_correct:   Optional[bool] = None

    metrics:      Optional[MathematicalMetrics] = None
    embedding_id: Optional[str] = None
    human_explanation: Optional["HumanExplanation"] = None
    explanation_external: Optional["ExplanationBundle"] = None


class FailureSignalVector(BaseModel):
    """Phase 1 output: per-inference uncertainty signals."""
    agreement_score:      float = Field(..., ge=0.0, le=1.0)
    fsd_score:            float = Field(..., ge=0.0, le=1.0)
    answer_counts:        dict[str, int]
    entropy_score:        float = Field(..., ge=0.0, le=1.0)
    ensemble_disagreement: bool
    ensemble_similarity:  float = Field(..., ge=0.0, le=1.0)
    high_failure_risk:    bool  = False


# ── Phase 2 schemas ───────────────────────────────────────────────────────

class ClusterAssignment(BaseModel):
    cluster_id:       Optional[str] = None
    status:           str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    archetype:        str


class LabelResult(BaseModel):
    archetype:      str
    confidence:     str
    conditions_met: list[str]


class ArchetypeAnalysisResponse(BaseModel):
    failure_signal_vector: FailureSignalVector
    cluster_assignment:    ClusterAssignment
    label_detail:          LabelResult
    embedding_distance:    float
    trend_summary:         Optional[dict] = None


class AnalyzeRequest(BaseModel):
    """
    Input for /analyze and /analyze/v2 endpoints.
    model_outputs[0] = primary, model_outputs[1] = reference.
    All outputs are used by every detector.
    """
    model_outputs: list[str] = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    failure_signal_vector: FailureSignalVector


class TrackResponse(BaseModel):
    status:     str
    request_id: str


class InferenceRecord(BaseModel):
    inference:      InferenceRequest
    failure_signal: Optional[FailureSignalVector] = None
    recorded_at:    datetime = Field(default_factory=datetime.utcnow)


class TrendResponse(BaseModel):
    signals_recorded:      int
    decay_alpha:           float
    ema_entropy:           float
    ema_agreement:         float
    ema_disagreement_rate: float
    ema_high_risk_rate:    float
    degradation_velocity:  float
    is_degrading:          bool


class ClusterSummaryResponse(BaseModel):
    total_clusters: int
    clusters:       list[dict]


# ── Phase 3 schemas ───────────────────────────────────────────────────────

class AgentVerdict(BaseModel):
    agent_name:          str
    root_cause:          str
    confidence_score:    float = Field(..., ge=0.0, le=1.0)
    mitigation_strategy: str
    evidence:            Optional[dict] = None
    skipped:             bool           = False
    skip_reason:         Optional[str]  = None


class JuryVerdict(BaseModel):
    verdicts:           list[AgentVerdict]
    primary_verdict:    Optional[AgentVerdict] = None
    jury_confidence:    float                  = Field(default=0.0, ge=0.0, le=1.0)
    is_adversarial:     bool                   = False
    is_complex_prompt:  bool                   = False
    failure_summary:    str                    = ""


class DiagnosticRequest(BaseModel):
    """
    Input to the /diagnose endpoint (Phase 3).
    model_outputs[0] = primary (model under test)
    model_outputs[1] = secondary / reference model (if present)
    model_outputs[2..N] = additional ensemble members
    """
    prompt:        str
    model_outputs: list[str] = Field(..., min_length=1)
    latency_ms:    Optional[float] = None


class DiagnosticResponse(BaseModel):
    failure_signal_vector: FailureSignalVector
    archetype:             str
    embedding_distance:    float
    jury:                  JuryVerdict
    explanation_internal:  Optional["ExplanationBundle"] = None
    explanation_external:  Optional["ExplanationBundle"] = None
    human_explanation:     Optional["HumanExplanation"] = None


# ── Phase 4 schemas — Real-time Monitor ───────────────────────────────────

class OllamaModelResult(BaseModel):
    """Response from a single Ollama shadow model."""
    model_name:  str
    output_text: str
    latency_ms:  float
    success:     bool
    error:       str = ""


class MonitorRequest(BaseModel):
    """
    Input to the /monitor endpoint.

    The user sends a prompt and their primary model output.
    FIE automatically calls Ollama shadow models to get additional
    outputs, then runs the full pipeline on all outputs together.

    prompt               : the original user prompt
    primary_output       : response from the user's main model
    primary_model_name   : name of the main model (for logging)
    run_full_jury        : if True, runs Phase 3 DiagnosticJury as well
                           if False, only runs Phase 1 + Phase 2 (faster)
    """
    prompt:             str
    primary_output:     str
    primary_model_name: str            = "primary"
    run_full_jury:      bool           = True
    latency_ms:         Optional[float] = None


class FixResult(BaseModel):
    """Result from the auto-fix engine."""
    fixed_output:      str
    fix_applied:       bool
    fix_strategy:      str
    fix_explanation:   str
    original_output:   str
    root_cause:        str
    fix_confidence:    float = Field(default=0.0, ge=0.0, le=1.0)
    improvement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warning:           str   = ""


class ExplanationSignal(BaseModel):
    """Normalized signal used to justify a diagnosis or mitigation."""
    name: str
    source: str
    value: str
    normalized_score: float = Field(default=0.0, ge=0.0, le=1.0)
    direction: str
    contribution_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str


class ExplanationEvidence(BaseModel):
    """Evidence item that supports an explanation."""
    type: str
    source: str
    content_preview: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supports: str
    safe_to_expose: bool = False


class ExplanationStep(BaseModel):
    """One step in the system's decision trace."""
    stage: str
    decision: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    inputs_used: list[str] = Field(default_factory=list)


class ExplanationAttribution(BaseModel):
    """Ranked factor showing which inputs influenced the explanation most."""
    factor: str
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    polarity: str
    details: str


class ExplanationBundle(BaseModel):
    """Structured explainability payload for audit, dashboard, or user views."""
    explanation_version: str = "v1"
    mode: str
    request_id: Optional[str] = None
    final_label: str
    final_fix_strategy: str
    explanation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str
    decision_trace: list[ExplanationStep] = Field(default_factory=list)
    signals: list[ExplanationSignal] = Field(default_factory=list)
    evidence: list[ExplanationEvidence] = Field(default_factory=list)
    attributions: list[ExplanationAttribution] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    internal_only: bool = False


class HumanExplanation(BaseModel):
    """Plain-language explanation generated from structured XAI signals."""
    summary: str
    why_risky: str
    recommended_action: str
    severity: str = "medium"
    generated_by: str = "template"
    safe_for_user: bool = True


class MonitorResponse(BaseModel):
    """
    Full response from the /monitor endpoint.
    Contains Phase 1 + Phase 2 results and optionally Phase 3 jury.
    Also includes the raw Ollama model responses for transparency.
    """
    # What models responded and what they said
    shadow_model_results:  list[OllamaModelResult]
    all_model_outputs:     list[str]   # primary + all shadow outputs
    ollama_available:      bool

    # Phase 1 + 2
    failure_signal_vector: FailureSignalVector
    archetype:             str
    embedding_distance:    float

    # Phase 3 (only if run_full_jury=True)
    jury:                  Optional[JuryVerdict] = None

    # Quick summary
    high_failure_risk:     bool
    failure_summary:       str

    # Auto-fix result (None if no failure detected or fix not applied)
    fix_result:            Optional[FixResult] = None
    explanation_internal:  Optional[ExplanationBundle] = None
    explanation_external:  Optional[ExplanationBundle] = None
    human_explanation:     Optional[HumanExplanation] = None
