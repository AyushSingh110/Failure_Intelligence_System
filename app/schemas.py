from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

#Provenance types 
ProvenanceCategory = Literal[
    "GENERAL_KNOWLEDGE",
    "LIVE_WORLD_STATE",
    "USER_SPECIFIC_STATE",
    "MIXED_SYNTHESIS",
]

# ProvenanceLabel: how well was the source of this specific response verified?
ProvenanceLabel = Literal[
    "FULLY_PROVENANCED",
    "PARTIALLY_PROVENANCED",
    "UNVERIFIED_MODEL_INFERENCE",
    "REQUIRES_TOOL_VERIFICATION",
    "NULL_REQUIRED_BUT_MISSING",
]


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

    # Adversarial guard fields — set when the pre-flight guard or adversarial_guard fires
    is_adversarial:     Optional[bool]  = None
    guard_attack_type:  Optional[str]   = None
    guard_confidence:   Optional[float] = None
    archetype:          Optional[str]   = None


class FailureSignalVector(BaseModel):
    """Phase 1 output: per-inference uncertainty signals."""
    agreement_score:      float = Field(..., ge=0.0, le=1.0)
    fsd_score:            float = Field(..., ge=0.0, le=1.0)
    answer_counts:        dict[str, int]
    entropy_score:        float = Field(..., ge=0.0, le=1.0)
    ensemble_disagreement: bool
    ensemble_similarity:  float = Field(..., ge=0.0, le=1.0)
    high_failure_risk:    bool  = False
    # Question-type classification (set at routing time, before jury)
    question_type:        str   = "UNKNOWN"   # FACTUAL|TEMPORAL|REASONING|CODE|OPINION|UNKNOWN

    # Phase 1 Provenance
    provenance_category: str = "GENERAL_KNOWLEDGE"
    provenance_label:    str = "UNVERIFIED_MODEL_INFERENCE"


# Phase 2 schemas 
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


# Phase 3 schemas 
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

#Input to the /diagnose endpoint 
class DiagnosticRequest(BaseModel):
    prompt:        str
    model_outputs: list[str] = Field(..., min_length=1)
    latency_ms:    Optional[float] = None
    canary_token:  Optional[str]   = None


class DiagnosticResponse(BaseModel):
    failure_signal_vector: FailureSignalVector
    archetype:             str
    embedding_distance:    float
    jury:                  JuryVerdict
    explanation_internal:  Optional["ExplanationBundle"] = None
    explanation_external:  Optional["ExplanationBundle"] = None
    human_explanation:     Optional["HumanExplanation"] = None


#Phase 4 schemas — Real-time Monitor 

class OllamaModelResult(BaseModel):
    """Response from a single Ollama shadow model."""
    model_name:  str
    output_text: str
    latency_ms:  float
    success:     bool
    error:       str = ""

class MonitorRequest(BaseModel):
    prompt:                    str
    primary_output:            str
    primary_model_name:        str             = "primary"
    run_full_jury:             bool            = True
    latency_ms:                Optional[float] = None
    conversation_id:           Optional[str]   = Field(None, max_length=128, pattern=r"^[a-zA-Z0-9_\-]{1,128}$")
    session_id:                Optional[str]   = Field(None, max_length=128, pattern=r"^[a-zA-Z0-9_\-]{1,128}$", description="When provided, FIE auto-threads conversation history. Prior turns are fetched and injected as context[] automatically — no need to pass context manually.")
    context:                   Optional[list[dict]] = Field(None, description="Prior conversation turns [{role, content}] to prime shadow models with the same history VEXR had. Auto-populated from session_id if not provided.")
    is_constitutional_refusal: bool            = Field(False, description="Set True when the primary output is an intentional refusal (Article 6 / sovereign right). FIE will classify as CONSTITUTIONAL_REFUSAL instead of a failure archetype.")

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

class GroundTruthVerification(BaseModel):
    verified_answer:     str   = ""
    confidence:          float = Field(default=0.0, ge=0.0, le=1.0)
    source:              str   = "none"
    from_cache:          bool  = False
    requires_escalation: bool  = False
    escalation_reason:   str   = ""
    pipeline_trace:      list[str] = Field(default_factory=list)
class ReasoningStepResult(BaseModel):
    """API-safe representation of one verified reasoning step."""
    index:        int
    text:         str
    step_type:    str
    is_correct:   bool  = True
    confidence:   float = Field(default=0.0, ge=0.0, le=1.0)
    failure_note: str   = ""


class ReasoningVerification(BaseModel):
    failure_detected:    bool  = False
    failure_type:        str   = "NO_FAILURE_DETECTED"
    confidence:          float = Field(default=0.0, ge=0.0, le=1.0)
    total_steps:         int   = 0
    first_failed_step:   Optional[int] = None
    steps:               list[ReasoningStepResult] = Field(default_factory=list)
    pipeline_trace:      list[str] = Field(default_factory=list)
    # Socratic probe summary
    socratic_probes_run:         int   = 0
    socratic_contradiction_found: bool  = False
    socratic_score:              float = Field(default=0.0, ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    """
    Step 8 — User submits ground truth feedback for a specific inference.

    is_correct     : True = the model's answer was right
    correct_answer : (when is_correct=False) what the right answer actually is
    notes          : optional free-text comment
    """
    is_correct:     bool
    correct_answer: Optional[str] = None
    notes:          Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response to a feedback submission."""
    status:          str
    request_id:      str
    cache_updated:   bool  = False
    message:         str   = ""


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

    # Step 7+10 — Ground truth verification and escalation
    ground_truth:          Optional[GroundTruthVerification] = None
    requires_human_review: bool = False
    escalation_reason:     str  = ""

    # Reasoning verification (populated when question_type = REASONING)
    reasoning_verification: Optional[ReasoningVerification] = None

    # Classifier metadata (useful for debugging and research paper)
    classifier_probability: Optional[float] = None
    model_version:          str             = "xgboost-v2"
    config_version:         str             = "default"

    # Multi-turn adversarial tracking (populated when conversation_id is provided)
    multi_turn_escalation:  Optional[dict]  = None

    # Pre-flight guard — populated when the prompt was blocked BEFORE the LLM ran
    guard_blocked:      bool         = False
    guard_attack_type:  Optional[str]   = None
    guard_confidence:   Optional[float] = None


class TelemetryPing(BaseModel):
    """Anonymized usage ping from fie-sdk clients (FIE_TELEMETRY=true)."""
    event:            str   = Field("unknown", max_length=64)
    sdk_version:      str   = Field("unknown", max_length=32)
    high_failure_risk: bool = False
    fix_applied:      bool  = False
    question_type:    str   = Field("UNKNOWN", max_length=32)
    model_version:    str   = Field("unknown", max_length=32)
    mode:             str   = Field("unknown", max_length=32)
