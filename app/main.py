from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import router
from config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize vault on disk if it doesn't exist
    from storage.database import initialize_vault
    initialize_vault()
    yield
    # Shutdown: flush in-memory store to disk
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
