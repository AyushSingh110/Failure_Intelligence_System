"""
Failure Intelligence Engine — ASGI application entry point.

Startup sequence
----------------
1. configure_logging()       — structured JSON to stdout (replaces all handlers)
2. initialize_vault()        — MongoDB connection + in-memory fallback
3. load_from_db()            — hot-configurable thresholds from MongoDB
4. _warm_encoder_in_background() — lazy sentence-encoder warm-up (non-blocking)

Middleware
----------
security_and_logging  — injects X-Request-ID, binds correlation ID into every
                        log line via bind_request_id(), adds security headers.

Rate limiting
-------------
Powered by slowapi (optional); gracefully disabled if not installed.
The limiter singleton lives in app.limiter to avoid the circular import that
existed when routes.py imported _limiter from app.main.
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from engine.logging_config import configure_logging, bind_request_id

# ── Logging must be configured before any other FIE import emits log records ──
configure_logging()

import logging
import os

logger = logging.getLogger("fie.server")

# ── Route packages ─────────────────────────────────────────────────────────────
from app.routes import router
from app.auth_routes import router as auth_router

# ── Rate limiting (optional) ───────────────────────────────────────────────────
from app.limiter import limiter as _limiter, available as _rate_limiting_available
try:
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler
except ImportError:
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


# ── Background encoder warm-up ─────────────────────────────────────────────────

def _warm_encoder_in_background() -> None:
    """Warm the sentence encoder after startup without blocking the web server."""
    logger.info("background_task=encoder_warmup status=started")
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        _ = encoder.encode("warmup")
        if encoder.available:
            logger.info("background_task=encoder_warmup status=ready backend=transformer")
        else:
            logger.warning(
                "background_task=encoder_warmup status=unavailable backend=ngram_fallback"
            )
    except Exception as exc:
        logger.error("background_task=encoder_warmup status=failed error=%s", exc)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    from storage.database import initialize_vault
    initialize_vault()

    try:
        from engine.fie_config import load_from_db
        load_from_db()
        logger.info("startup=fie_config status=loaded")
    except Exception as _cfg_exc:
        logger.warning("startup=fie_config status=skipped reason=%s", _cfg_exc)

    threading.Thread(target=_warm_encoder_in_background, daemon=True).start()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    from storage.database import flush_vault
    flush_vault()
    logger.info("shutdown=vault status=flushed")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title    = settings.app_name,
    version  = settings.app_version,
    debug    = settings.debug,
    lifespan = lifespan,
)

# Rate limiting — gracefully disabled if slowapi not installed
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


# ── Middleware: security headers + structured request logging ──────────────────

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


# ── Root endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict[str, str]:
    return {
        "system":  settings.app_name,
        "version": settings.app_version,
        "status":  "operational",
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe — returns component status for uptime monitors."""
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
        results["encoder"] = {"status": "down", "error": str(exc)[:120]}

    # XGBoost classifier
    try:
        from engine.failure_classifier import _model
        results["xgboost"] = {
            "status": "ok" if _model is not None else "degraded",
            "mode":   "xgboost" if _model is not None else "rule_based_fallback",
        }
    except Exception as exc:
        results["xgboost"] = {"status": "down", "error": str(exc)[:120]}

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
