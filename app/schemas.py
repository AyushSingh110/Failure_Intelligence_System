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