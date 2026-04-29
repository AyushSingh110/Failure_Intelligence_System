from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading

from app.routes import router
from app.auth_routes import router as auth_router
from config import get_settings


settings = get_settings()


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "system": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
