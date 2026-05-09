from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import threading
import uuid
import time
import logging
import os

logger = logging.getLogger("fie.server")

from app.routes import router
from app.auth_routes import router as auth_router
from config import get_settings

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _limiter = Limiter(key_func=get_remote_address)
    _rate_limiting_available = True
except ImportError:
    _limiter = None
    _rate_limiting_available = False

settings = get_settings()

# Allowed origins: env-configurable, defaults to localhost in dev
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8000",
).split(",") if o.strip()]


def _warm_encoder_in_background() -> None:
    """Warm the sentence encoder after startup without blocking the web server."""
    print("[startup] Background encoder warmup started...")
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        _ = encoder.encode("warmup")
        if encoder.available:
            print("[startup] Sentence encoder ready.")
        else:
            print("[startup] WARNING: Sentence encoder unavailable.")
            print("[startup]   Consistency and embedding will use n-gram fallback.")
    except Exception as exc:
        print(f"[startup] Encoder warmup failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    #  Startup
    from storage.database import initialize_vault
    initialize_vault()

    # Load classifier thresholds from MongoDB (hot-configurable, no restart needed)
    try:
        from engine.fie_config import load_from_db
        load_from_db()
        print("[startup] FIE config loaded.")
    except Exception as _cfg_exc:
        print(f"[startup] fie_config load skipped: {_cfg_exc}")

    threading.Thread(target=_warm_encoder_in_background, daemon=True).start()

    yield

    #  Shutdown ───────────────────────────────────────────────────────
    from storage.database import flush_vault
    flush_vault()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Rate limiting — gracefully disabled if slowapi not installed
if _rate_limiting_available:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")


@app.middleware("http")
async def security_and_logging(request: Request, call_next):
    """Security headers + request-ID logging on every response."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)

    # ── Security headers ────────────────────────────────────────────────
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Tight CSP for API server (no HTML pages served, only JSON)
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

    if request.url.path not in ("/health", "/"):
        logger.info(
            "rid=%s method=%s path=%s status=%d latency=%.1fms",
            rid, request.method, request.url.path, response.status_code, elapsed,
        )
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {
        "system": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe — returns component status for uptime monitors."""
    from storage.database import get_db
    db_ok = False
    try:
        db = get_db()
        db_ok = db is not None
    except Exception:
        pass

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
