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

    # Detector thresholds
    ensemble_disagreement_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    high_entropy_threshold:          float = Field(default=0.75, ge=0.0, le=1.0)

    # Raised to 0.80 so 1 outlier in 4 models (agreement=0.75) is caught
    low_agreement_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # Clustering
    cluster_base_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    cluster_novel_anomaly_ceiling:      float = Field(default=0.45, ge=0.0, le=1.0)
    cluster_threshold_growth_rate:      float = Field(default=0.002, ge=0.0, le=0.05)
    cluster_threshold_max:              float = Field(default=0.92, ge=0.0, le=1.0)

    # Evolution tracker
    tracker_window_size:                   int   = Field(default=100, ge=2, le=10_000)
    tracker_decay_alpha:                   float = Field(default=0.94, gt=0.0, le=1.0)
    tracker_degradation_risk_threshold:    float = Field(default=0.40, ge=0.0, le=1.0)
    tracker_degradation_velocity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    # Vault
    vault_path:                  str   = "storage/vault.json"
    max_vault_records:           int   = Field(default=10_000, ge=1)
    vault_flush_interval_seconds: float = Field(default=5.0, ge=0.5, le=300.0)

    # MongoDB
    mongodb_uri:     str = Field(default="")
    mongodb_db_name: str = Field(default="fie_database")

    # Ollama shadow model service
    ollama_base_url:        str       = Field(default="http://localhost:11434")
    ollama_models:          list[str] = Field(default=["mistral", "llama3.2", "phi3"])
    ollama_timeout_seconds: int       = Field(default=180, ge=5, le=300)
    # Ollama disabled — using Groq now (faster, no GPU needed)
    # To re-enable: set ollama_enabled=True in .env
    ollama_enabled:         bool      = Field(default=False)

    #Groq API settings 
    groq_api_key:           str       = Field(default="")
    groq_models:            list[str] = Field(default=[
                                            "llama-3.3-70b-versatile",       
                                            "deepseek-r1-distill-llama-70b",  
                                            "qwen-qwq-32b",                
                                        ])
    groq_timeout_seconds:   int       = Field(default=30, ge=5, le=120)
    groq_enabled:           bool      = Field(default=True)

    # Ground Truth Verification settings 
    wikidata_enabled:                 bool  = Field(default=True)
    wikidata_timeout_seconds:         int   = Field(default=8, ge=2, le=30)

    # Serper: real-time Google search for temporal questions
    serper_api_key:                   str   = Field(default="")
    serper_enabled:                   bool  = Field(default=False)
    serper_timeout_seconds:           int   = Field(default=8, ge=2, le=30)

    # Ground truth cache 
    ground_truth_cache_enabled:       bool  = Field(default=True)
    ground_truth_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    # Confidence weights for shadow model families
    confidence_weight_high:           float = Field(default=3.0)
    confidence_weight_medium:         float = Field(default=2.0)
    confidence_weight_low:            float = Field(default=1.0)

    # Escalation: when total weighted confidence is below this
    escalation_confidence_threshold:  float = Field(default=0.40, ge=0.0, le=1.0)

    # Embeddings
    embedding_ngram_size:        int  = Field(default=3, ge=1, le=6)
    embedding_use_transformer:   bool = True
    embedding_transformer_model: str  = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension:         int  = Field(default=384, ge=64)

    # FAISS
    faiss_index_path:                    str   = "storage/faiss_adversarial.index"
    faiss_meta_path:                     str   = "storage/faiss_adversarial_meta.json"
    faiss_adversarial_similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    faiss_top_k:                         int   = Field(default=5, ge=1, le=50)

    # DiagnosticJury thresholds
    jury_linguistic_complexity_threshold:    float = Field(default=0.20, ge=0.0, le=1.0)
    jury_linguistic_entropy_threshold:       float = Field(default=0.45, ge=0.0, le=1.0)
    jury_adversarial_faiss_threshold:        float = Field(default=0.82, ge=0.0, le=1.0)
    jury_adversarial_pattern_confidence:     float = Field(default=0.75, ge=0.0, le=1.0)
    # Lowered to 0.08 so DomainCritic fires when 1 model disagrees with majority
    jury_domain_confidence_threshold:        float = Field(default=0.06, ge=0.0, le=1.0)
    jury_domain_self_contradiction_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # Dashboard
    dashboard_auto_refresh_seconds: int = Field(default=10, ge=5, le=300)
    dashboard_max_chart_records:    int = Field(default=500, ge=10)

    # Validators
    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_flag(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

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
                "high_entropy_threshold equals low_agreement_threshold.",
                UserWarning,
                stacklevel=2,
            )
        return v

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive":    False,
        "extra":             "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
