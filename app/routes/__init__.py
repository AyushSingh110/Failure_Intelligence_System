"""
app.routes — FastAPI route package for the Failure Intelligence Engine.

This package replaces the monolithic app/routes.py (1863 lines) with four
focused modules, each owning a single domain:

  inference.py  — track / analyze / inferences CRUD / diagnose (Phases 1–3)
  monitor.py    — real-time monitoring, feedback loop, calibration (Phase 4)
  analytics.py  — trend, clusters, telemetry, analytics dashboards
  admin.py      — notifications and digest endpoints

A single `router` is re-exported here so app/main.py needs no changes to its
import statement (`from app.routes import router`).
"""
from fastapi import APIRouter

from app.routes.inference import router as _inference_router
from app.routes.monitor   import router as _monitor_router
from app.routes.analytics import router as _analytics_router
from app.routes.admin     import router as _admin_router

router = APIRouter()
router.include_router(_inference_router)
router.include_router(_monitor_router)
router.include_router(_analytics_router)
router.include_router(_admin_router)

__all__ = ["router"]
