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
    agreement_score:     float = Field(..., ge=0.0, le=1.0)
    fsd_score:           float = Field(..., ge=0.0, le=1.0)
    answer_counts:       dict[str, int]
    entropy_score:       float = Field(..., ge=0.0, le=1.0)
    ensemble_disagreement: bool
    ensemble_similarity: float = Field(..., ge=0.0, le=1.0)
    high_failure_risk:   bool  = False


# ── Phase 2 schemas ───────────────────────────────────────────────────────

class ClusterAssignment(BaseModel):
    """Result of assigning a signal to the archetype registry."""
    cluster_id:       Optional[str] = None
    status:           str            # KNOWN_FAILURE | NOVEL_ANOMALY | AMBIGUOUS
    similarity_score: float          = Field(..., ge=0.0, le=1.0)
    archetype:        str


class LabelResult(BaseModel):
    """Detailed archetype label with diagnostic conditions."""
    archetype:       str
    confidence:      str            # HIGH | MEDIUM | LOW
    conditions_met:  list[str]


class ArchetypeAnalysisResponse(BaseModel):
    """
    Full Phase 2 response — signal + cluster assignment + label detail.
    Returned by the /analyze/v2 endpoint.
    """
    failure_signal_vector: FailureSignalVector
    cluster_assignment:    ClusterAssignment
    label_detail:          LabelResult
    embedding_distance:    float
    trend_summary:         Optional[dict] = None


class AnalyzeRequest(BaseModel):
    model_outputs:    list[str] = Field(..., min_length=1)
    primary_output:   str
    secondary_output: str


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
    """Response from the /trend endpoint."""
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
