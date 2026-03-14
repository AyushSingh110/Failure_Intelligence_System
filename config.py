from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    
    # Application identity
    app_name:    str  = "Failure Intelligence Engine"
    app_version: str  = "3.0.0"
    debug:       bool = False
    log_level:   str  = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    
    # API server
    api_host:     str       = "0.0.0.0"
    api_port:     int       = Field(default=8000, ge=1, le=65535)
    api_prefix:   str       = "/api/v1"
    cors_origins: list[str] = Field(default=["*"])

    # ensemble.py — cosine similarity below this threshold → models disagree
    ensemble_disagreement_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # labeling.py / routes.py — entropy above this → UNSTABLE_OUTPUT or HALLUCINATION_RISK
    high_entropy_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # labeling.py / routes.py — agreement below this → LOW_CONFIDENCE
    low_agreement_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    
    # Clustering
    # Used by: clustering.py
    

    # Base similarity required to merge a signal into an existing cluster
    cluster_base_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # Similarity below this → signal is NOVEL_ANOMALY, not a cluster member
    cluster_novel_anomaly_ceiling: float = Field(default=0.45, ge=0.0, le=1.0)

    # How much the threshold grows per existing cluster (adaptive threshold)
    cluster_threshold_growth_rate: float = Field(default=0.002, ge=0.0, le=0.05)

    # Hard ceiling on adaptive threshold growth
    cluster_threshold_max: float = Field(default=0.92, ge=0.0, le=1.0)

    
    # Evolution tracker
    # Used by: tracker.py
    

    # Number of signals to keep in the rolling window
    tracker_window_size: int = Field(default=100, ge=2, le=10_000)

    # Exponential decay factor  (0 < alpha <= 1)
    # Higher -> slower decay (history matters more)
    # Lower  -> faster decay (recency dominates)
    tracker_decay_alpha: float = Field(default=0.94, gt=0.0, le=1.0)

    # high_risk_rate above this -> is_degrading() returns True
    tracker_degradation_risk_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # degradation_velocity above this -> is_degrading() returns True
    tracker_degradation_velocity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    
    # Vault (storage)
    # Used by: database.py
    

    # Path to the JSON vault file, relative to project root
    vault_path: str = "storage/vault.json"

    # Hard cap on in-memory + persisted records (oldest evicted when exceeded)
    max_vault_records: int = Field(default=10_000, ge=1)

    # How often the background flush thread writes to disk (seconds)
    vault_flush_interval_seconds: float = Field(default=5.0, ge=0.5, le=300.0)

    
    # Embeddings
    # Used by: embedding.py, encoder.py
    

    # N-gram size for character-level similarity (Phase 1/2 fallback)
    embedding_ngram_size: int = Field(default=3, ge=1, le=6)

    # Phase 3: enable sentence-transformer embeddings
    # Set True for DiagnosticJury + FAISS semantic search
    # Requires: pip install sentence-transformers
    embedding_use_transformer: bool = True

    # HuggingFace model ID — 384-dim, fast, fits on RTX 3050 (4 GB VRAM)
    embedding_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Dimension of all-MiniLM-L6-v2 output vectors
    # Must match the model above — change only if you swap to a different model
    embedding_dimension: int = Field(default=384, ge=64)

    
    # FAISS Vector Index
    # Used by: registry.py (AdversarialRegistry)
    

    # Path where the FAISS adversarial prompt index is persisted to disk
    # Created automatically on first run, reloaded on subsequent starts
    faiss_index_path: str = "storage/faiss_adversarial.index"

    # Sidecar JSON mapping each FAISS row -> { prompt, label, category, source }
    faiss_meta_path: str = "storage/faiss_adversarial_meta.json"

    # Cosine similarity above which a query prompt is flagged adversarial
    # Range: 0.0-1.0  |  Default: 0.82 (high precision, low false-positive rate)
    faiss_adversarial_similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)

    # Number of nearest neighbours FAISS returns per query
    faiss_top_k: int = Field(default=5, ge=1, le=50)

    
    # Phase 3 -- DiagnosticJury agent thresholds
    # Used by: linguistic_auditor.py, adversarial_specialist.py
    

    # -- LinguisticAuditor ---------------------------------------------

    # Minimum prompt complexity score to activate the LinguisticAuditor.
    # Below this score the agent skips entirely.
    # Range: 0.0-1.0  |  Lower = more sensitive (catches more edge cases)
    jury_linguistic_complexity_threshold: float = Field(default=0.20, ge=0.0, le=1.0)

    # Minimum entropy for the LinguisticAuditor to confirm PROMPT_COMPLEXITY_OOD.
    # Should sit below high_entropy_threshold so it catches moderate instability.
    # Range: 0.0-1.0  |  Default: 0.45
    jury_linguistic_entropy_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # -- AdversarialSpecialist -----------------------------------------

    # FAISS cosine similarity threshold used inside the agent's confidence formula.
    # Kept separate from faiss_adversarial_similarity_threshold so the jury
    # agent can be tuned independently from the raw FAISS index.
    # Range: 0.0-1.0  |  Default: 0.82
    jury_adversarial_faiss_threshold: float = Field(default=0.82, ge=0.0, le=1.0)

    # Confidence cap when ONLY the regex layer fires (no FAISS confirmation).
    # Prevents regex-only matches from claiming full 0.91 base confidence.
    # Range: 0.0-1.0  |  Default: 0.75
    jury_adversarial_pattern_confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    
    # Phase 3 — DomainCritic (Agent 3) thresholds
    # Used by: engine/agents/domain_critic.py
    

    # Minimum scaled confidence for DomainCritic to return a verdict.
    # Below this, it returns DOMAIN_CORRECT or skips entirely.
    # Range: 0.0-1.0  |  Default: 0.30
    jury_domain_confidence_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # Cosine similarity below this value triggers self-contradiction in Layer 2.
    # 0.70 means primary and secondary outputs differ meaningfully.
    # Range: 0.0-1.0  |  Default: 0.70
    jury_domain_self_contradiction_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    
    # Dashboard
    # Used by: dashboard/ui.py
    

    dashboard_auto_refresh_seconds: int = Field(default=10, ge=5, le=300)
    dashboard_max_chart_records:    int = Field(default=500, ge=10)

    
    # Validators
    

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
        if v == low_agree:
            import warnings
            warnings.warn(
                "high_entropy_threshold equals low_agreement_threshold. "
                "These thresholds operate on different metrics — verify this is intentional.",
                UserWarning,
                stacklevel=2,
            )
        return v

    
    # Pydantic-settings model config
    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive":    False,
        "extra":             "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()