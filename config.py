"""
config.py — Failure Intelligence Engine: Centralized Configuration

All runtime parameters for every engine module live here.
No magic numbers anywhere else in the codebase.

Environment Variable Override
------------------------------
Every field maps directly to an env var of the same name (uppercased).
For example:
    HIGH_ENTROPY_THRESHOLD=0.80 uvicorn app.main:app

Or place them in a .env file at the project root (see .env.example).

Validation
----------
pydantic-settings validates types and ranges at startup.
The app will refuse to start with an invalid configuration rather than
silently misbehaving at runtime.

Usage
-----
    from config import get_settings
    settings = get_settings()
    threshold = settings.high_entropy_threshold

get_settings() is cached via @lru_cache — safe to call at module level
anywhere in the codebase without performance concern.
"""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ------------------------------------------------------------------
    # Application identity
    # ------------------------------------------------------------------

    app_name: str = "Failure Intelligence Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default=["*"])

    # ------------------------------------------------------------------
    # Detector thresholds
    # Used by: ensemble.py, labeling.py, failure_agent.py, routes.py
    # ------------------------------------------------------------------

    # ensemble.py — cosine similarity below this → models disagree
    ensemble_disagreement_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # labeling.py / routes.py — entropy above this → UNSTABLE_OUTPUT or HALLUCINATION_RISK
    high_entropy_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # labeling.py / routes.py — agreement below this → LOW_CONFIDENCE
    low_agreement_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Clustering
    # Used by: clustering.py
    # ------------------------------------------------------------------

    # Base similarity required to merge a signal into an existing cluster
    cluster_base_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # Similarity below this → signal is a NOVEL_ANOMALY, not a cluster member
    cluster_novel_anomaly_ceiling: float = Field(default=0.45, ge=0.0, le=1.0)

    # How much the threshold grows per existing cluster (adaptive threshold)
    cluster_threshold_growth_rate: float = Field(default=0.002, ge=0.0, le=0.05)

    # Hard ceiling on adaptive threshold growth
    cluster_threshold_max: float = Field(default=0.92, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Evolution tracker
    # Used by: tracker.py
    # ------------------------------------------------------------------

    # Number of signals to keep in the rolling window
    tracker_window_size: int = Field(default=100, ge=2, le=10_000)

    # Exponential decay factor (0 < alpha ≤ 1)
    # Higher → slower decay (history matters more)
    # Lower  → faster decay (recency dominates)
    tracker_decay_alpha: float = Field(default=0.94, gt=0.0, le=1.0)

    # high_risk_rate above this → is_degrading() returns True
    tracker_degradation_risk_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # degradation_velocity above this → is_degrading() returns True
    tracker_degradation_velocity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Vault (storage)
    # Used by: database.py
    # ------------------------------------------------------------------

    # Path to the JSON vault file, relative to project root
    vault_path: str = "storage/vault.json"

    # Hard cap on in-memory + persisted records (oldest evicted when exceeded)
    max_vault_records: int = Field(default=10_000, ge=1)

    # How often the background flush thread writes to disk (seconds)
    vault_flush_interval_seconds: float = Field(default=5.0, ge=0.5, le=300.0)

    # ------------------------------------------------------------------
    # Embedding detector
    # Used by: embedding.py
    # ------------------------------------------------------------------

    # N-gram size for character-level similarity (Phase 1)
    embedding_ngram_size: int = Field(default=3, ge=1, le=6)

    # Phase 2 toggle: when True, embedding.py uses sentence-transformers
    # instead of n-grams (requires: pip install sentence-transformers)
    embedding_use_transformer: bool = False

    # HuggingFace model ID to use when embedding_use_transformer=True
    embedding_transformer_model: str = "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------
    # Dashboard
    # Used by: dashboard/ui.py
    # ------------------------------------------------------------------

    dashboard_auto_refresh_seconds: int = Field(default=10, ge=5, le=300)
    dashboard_max_chart_records: int = Field(default=500, ge=10)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("cluster_novel_anomaly_ceiling")
    @classmethod
    def novel_ceiling_below_base_threshold(cls, v: float, info) -> float:
        base = info.data.get("cluster_base_similarity_threshold", 0.80)
        if v >= base:
            raise ValueError(
                f"cluster_novel_anomaly_ceiling ({v}) must be strictly less than "
                f"cluster_base_similarity_threshold ({base})"
            )
        return v

    @field_validator("high_entropy_threshold")
    @classmethod
    def entropy_above_agreement(cls, v: float, info) -> float:
        low_agree = info.data.get("low_agreement_threshold", 0.50)
        # These operate on different axes so no hard constraint,
        # but warn if they're identical (likely a misconfiguration).
        if v == low_agree:
            import warnings
            warnings.warn(
                "high_entropy_threshold equals low_agreement_threshold. "
                "These thresholds operate on different metrics — verify this is intentional.",
                UserWarning,
                stacklevel=2,
            )
        return v

    # ------------------------------------------------------------------
    # Pydantic-settings config
    # ------------------------------------------------------------------

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
