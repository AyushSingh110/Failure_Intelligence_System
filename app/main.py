from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import router
from config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────
    # 1. Initialize vault
    from storage.database import initialize_vault
    initialize_vault()

    # 2. Warm up sentence encoder so errors appear at boot, not silently
    #    on first request.
    print("[startup] Warming up sentence encoder...")
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        _ = encoder.encode("warmup")
        if encoder.available:
            print("[startup] Sentence encoder ready.")
        else:
            print("[startup] WARNING: Sentence encoder unavailable.")
            print("[startup]   Consistency and embedding will use n-gram fallback.")
            print("[startup]   Check [encoder] lines above for the exact error.")
    except Exception as exc:
        print(f"[startup] Encoder warmup failed: {exc}")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────
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