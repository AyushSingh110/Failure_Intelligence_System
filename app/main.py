from __future__ import annotations
import threading
import time
import uuid
from contextlib import asynccontextmanager

# Load .env before any module-level
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from config import get_settings
from engine.logging_config import configure_logging, bind_request_id
# Logging must be configured before any other FIE import emits log records
configure_logging()
import logging
import os
logger = logging.getLogger("fie.server")

# Error tracking — strictly opt-in. Does nothing unless SENTRY_DSN is set,
# and degrades silently if sentry-sdk is not installed, so local development
# and existing deployments are unaffected.
_SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn                      = _SENTRY_DSN,
            environment              = os.getenv("SENTRY_ENVIRONMENT", "production"),
            traces_sample_rate       = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            send_default_pii         = False,   # never ship prompts/user data to Sentry
        )
        logger.info("startup=sentry status=enabled")
    except ImportError:
        logger.warning("startup=sentry status=skipped reason=sentry-sdk not installed")
    except Exception as _sentry_exc:
        logger.warning("startup=sentry status=failed reason=%s", _sentry_exc)

#Route packages
from app.routes import router
from app.auth_routes import router as auth_router

# Rate limiting
from app.limiter import limiter as _limiter, available as _rate_limiting_available
try:
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler
except ImportError:
    # slowapi is optional. Without it the app runs UNRATE-LIMITED, which is a
    # real exposure for a public endpoint — hence warning, not debug.
    logger.warning(
        "startup=rate_limiting status=disabled reason='slowapi not installed' "
        "impact='no per-IP request limits'"
    )
    RateLimitExceeded = None  # type: ignore[assignment, misc]
    _rate_limit_exceeded_handler = None  # type: ignore[assignment]

settings = get_settings()

_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:8000",
    ).split(",")
    if o.strip()
]


# ── Background warm-up ────────────────────────────────────────────────────────
#
# Model loading is lazy, so without an explicit warm-up the FIRST request pays
# it — roughly 10 s, against a 10 s layer deadline. That means a cold container's
# first real request is served with the PAIR layer marked degraded, i.e. with
# materially reduced recall, while still returning a confident-looking verdict.
#
# Warm-up runs in a background thread so the port starts accepting connections
# immediately (platform health checks do not wait for a transformer to load),
# and readiness is reported separately from liveness — see /health vs /ready.

# Set once warm-up finishes. Read by /ready.
_WARMUP_STATE: dict[str, object] = {"done": False, "detector": {}, "encoder": "pending"}


def _warm_models_in_background() -> None:
    """Preload the sentence encoder and every detector artifact. Never raises."""
    logger.info("background_task=warmup status=started")

    # Sentence encoder (shadow-ensemble / consistency scoring).
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        _ = encoder.encode("warmup")
        _WARMUP_STATE["encoder"] = "ready" if encoder.available else "degraded"
        if encoder.available:
            logger.info("background_task=warmup component=encoder status=ready backend=transformer")
        else:
            logger.warning(
                "background_task=warmup component=encoder status=unavailable "
                "backend=zero_vector_fallback"
            )
    except Exception as exc:
        _WARMUP_STATE["encoder"] = "failed"
        logger.error("background_task=warmup component=encoder status=failed error=%s", exc)

    # Adversarial detector (PAIR classifier, meta-classifier, layer pool).
    try:
        from fie.adversarial import warmup as _detector_warmup
        status = _detector_warmup()
        _WARMUP_STATE["detector"] = status
        if status.get("pair_classifier") != "ready":
            logger.warning(
                "background_task=warmup component=detector status=degraded detail=%s "
                "impact='reduced detection recall'", status,
            )
        else:
            logger.info("background_task=warmup component=detector status=ready detail=%s", status)
    except Exception as exc:
        _WARMUP_STATE["detector"] = {"error": str(exc)}
        logger.error("background_task=warmup component=detector status=failed error=%s", exc)

    _WARMUP_STATE["done"] = True
    logger.info("background_task=warmup status=complete")


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from storage.database import initialize_vault
    initialize_vault()

    try:
        from engine.fie_config import load_from_db
        load_from_db()
        logger.info("startup=fie_config status=loaded")
    except Exception as _cfg_exc:
        logger.warning("startup=fie_config status=skipped reason=%s", _cfg_exc)

    threading.Thread(
        target=_warm_models_in_background, name="fie-warmup", daemon=True,
    ).start()

    try:
        from fie.feedback_store import _load_confirmed_from_db
        threading.Thread(target=_load_confirmed_from_db, daemon=True).start()
        logger.info("startup=feedback_store status=loading")
    except Exception as _fb_exc:
        logger.warning("startup=feedback_store status=skipped reason=%s", _fb_exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    # Release the detector's shared thread pool before the process exits.
    # Without this, the 16 worker threads are only reclaimed by the atexit hook,
    # which does not run on SIGKILL and can leave a container lingering past its
    # grace period during a rolling deploy.
    try:
        from fie.adversarial import shutdown_layer_pool
        shutdown_layer_pool(wait=False)
        logger.info("shutdown=layer_pool status=released")
    except Exception as _pool_exc:
        logger.warning("shutdown=layer_pool status=failed reason=%s", _pool_exc)

    from storage.database import flush_vault
    flush_vault()
    logger.info("shutdown=vault status=flushed")


# FastAPI application
app = FastAPI(
    title    = settings.app_name,
    version  = settings.app_version,
    debug    = settings.debug,
    lifespan = lifespan,
)

# Rate limiting
if _rate_limiting_available and _limiter is not None:
    app.state.limiter = _limiter
    if RateLimitExceeded and _rate_limit_exceeded_handler:
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"],
)

app.include_router(router,      prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")

# Middleware: security headers + structured request logging
@app.middleware("http")
async def security_and_logging(request: Request, call_next):
    """
    Per-request middleware:
    1. Extract or generate a correlation ID (X-Request-ID header).
    2. Bind the ID to the logging ContextVar so every log line in this
       request carries `rid` automatically.
    3. Add production-grade security headers to every response.
    4. Log method, path, status, and latency at INFO level.
    """
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    request.state.request_id = rid
    start = time.perf_counter()

    with bind_request_id(rid):
        response = await call_next(request)

    elapsed = round((time.perf_counter() - start) * 1000, 1)

    # Security headers
    response.headers["X-Request-ID"]              = rid
    response.headers["X-Content-Type-Options"]    = "nosniff"
    response.headers["X-Frame-Options"]           = "DENY"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"]   = "default-src 'none'; frame-ancestors 'none'"

    if request.url.path not in ("/health", "/"):
        logger.info(
            "rid=%s method=%s path=%s status=%d latency_ms=%.1f",
            rid, request.method, request.url.path, response.status_code, elapsed,
        )

    return response


#Root endpoints
@app.get("/")
def root() -> dict[str, str]:
    return {
        "system":  settings.app_name,
        "version": settings.app_version,
        "status":  "operational",
    }

@app.get("/health")
def health() -> dict:
    """
    LIVENESS probe. "Is this process alive and able to serve?"

    Deliberately cheap and dependency-light: it makes no network calls and never
    triggers a model load. A liveness probe that touches slow dependencies will
    eventually time out under load and get the container killed — turning a
    partial outage into a restart loop. Use /ready for "should traffic come
    here?" and /health/deep for on-call diagnosis.
    """
    from storage import database as _db_module
    db_ok = not _db_module._fallback_mode and _db_module._db is not None
    components = {
        "api":      "ok",
        "database": "ok" if db_ok else "degraded",
    }
    overall = "healthy" if all(v == "ok" for v in components.values()) else "degraded"
    return {
        "status":     overall,
        "version":    settings.app_version,
        "components": components,
    }


@app.get("/ready")
def ready(response: Response) -> dict:
    """
    READINESS probe. "Should this instance receive traffic yet?"

    Returns 503 until background warm-up finishes. This is the endpoint a load
    balancer or `gcloud run deploy` health check should watch: routing traffic
    to a cold instance means the first users get scanned by a pipeline whose
    main classifier has not loaded, and the guard reports a confident verdict
    produced with reduced coverage.

    Separating this from /health is what makes a zero-downtime rolling deploy
    possible — the old instance keeps serving until the new one is genuinely
    warm, not merely running.
    """
    detector = _WARMUP_STATE.get("detector") or {}
    is_ready = bool(_WARMUP_STATE.get("done"))

    if not is_ready:
        response.status_code = 503

    return {
        "ready":    is_ready,
        "encoder":  _WARMUP_STATE.get("encoder"),
        "detector": detector,
        "version":  settings.app_version,
    }


@app.get("/health/deep")
def health_deep() -> dict:
    """
    Deep health check — actively pings all critical dependencies.
    Returns per-component status, latency, and error detail.
    Use for readiness probes and on-call dashboards.
    """
    import time as _time
    results: dict = {}
    # MongoDB
    try:
        from storage import database as _db_module
        t0 = _time.time()
        if _db_module._db is not None:
            _db_module._db.command("ping")
            results["mongodb"] = {"status": "ok", "latency_ms": round((_time.time() - t0) * 1000, 1)}
        else:
            results["mongodb"] = {"status": "degraded", "error": "not connected"}
    except Exception as exc:
        logger.warning("health_deep: mongodb ping failed: %s", exc)
        results["mongodb"] = {"status": "down", "error": str(exc)[:120]}

    # Groq
    try:
        from engine.groq_service import get_groq_service
        t0   = _time.time()
        groq = get_groq_service()
        if groq and groq._api_key:
            r = groq._call_single_model("llama-3.1-8b-instant", "Say ok", max_tokens=5)
            if r.success:
                results["groq"] = {"status": "ok", "latency_ms": r.latency_ms}
            else:
                results["groq"] = {"status": "degraded", "error": r.error[:120]}
        else:
            results["groq"] = {"status": "not_configured"}
    except Exception as exc:
        logger.warning("health_deep: groq probe failed: %s", exc)
        results["groq"] = {"status": "down", "error": str(exc)[:120]}

    # FAISS index
    try:
        from engine.archetypes.registry import adversarial_registry
        size = adversarial_registry.size
        results["faiss"] = {
            "status":  "ok" if size > 0 else "degraded",
            "vectors": size,
        }
    except Exception as exc:
        logger.warning("health_deep: faiss probe failed: %s", exc)
        results["faiss"] = {"status": "down", "error": str(exc)[:120]}

    # Sentence encoder
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        results["encoder"] = {
            "status":  "ok" if encoder.available else "degraded",
            "backend": "transformer" if encoder.available else "ngram_fallback",
        }
    except Exception as exc:
        logger.warning("health_deep: encoder probe failed: %s", exc)
        results["encoder"] = {"status": "down", "error": str(exc)[:120]}

    # XGBoost classifier
    try:
        from engine.failure_classifier import _model
        results["xgboost"] = {
            "status": "ok" if _model is not None else "degraded",
            "mode":   "xgboost" if _model is not None else "rule_based_fallback",
        }
    except Exception as exc:
        logger.warning("health_deep: xgboost probe failed: %s", exc)
        results["xgboost"] = {"status": "down", "error": str(exc)[:120]}

    # Adversarial detector — the guard itself. Reported last because it is the
    # component whose degradation is least visible from the outside: a scan with
    # PAIR unloaded still returns 200 with a confident-looking verdict, so this
    # is the only place an operator can see recall has silently dropped.
    try:
        from fie.adversarial import health as _detector_health
        det = _detector_health()   # non-blocking; never triggers a model load
        pair_ok = det["pair_classifier"]["loaded"]
        results["detector"] = {
            "status":  "ok" if pair_ok else "degraded",
            "mode":    "full_pipeline" if pair_ok else "reduced_recall",
            "detail":  det,
            "warmup":  _WARMUP_STATE.get("detector"),
        }
    except Exception as exc:
        logger.warning("health_deep: detector probe failed: %s", exc)
        results["detector"] = {"status": "down", "error": str(exc)[:120]}

    overall = (
        "healthy"  if all(v.get("status") == "ok"   for v in results.values()) else
        "degraded" if any(v.get("status") in ("ok", "degraded") for v in results.values()) else
        "down"
    )
    return {
        "status":     overall,
        "version":    settings.app_version,
        "components": results,
    }
