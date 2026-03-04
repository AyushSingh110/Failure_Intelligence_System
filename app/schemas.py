from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MathematicalMetrics(BaseModel):
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    logit_margin: Optional[float] = None
    agreement_score: Optional[float] = None
    fsd_score: Optional[float] = None
    consistency_entropy: Optional[float] = None
    embedding_distance: Optional[float] = None


class InferenceRequest(BaseModel):
    request_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    temperature: float
    latency_ms: float

    input_text: str
    output_text: str
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None

    metrics: Optional[MathematicalMetrics] = None

    embedding_id: Optional[str] = None    