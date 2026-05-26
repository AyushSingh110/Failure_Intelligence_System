from fastapi import APIRouter
from app.routes.inference  import router as _inference_router
from app.routes.monitor    import router as _monitor_router
from app.routes.analytics  import router as _analytics_router
from app.routes.admin      import router as _admin_router
from app.routes.playground import router as _playground_router

router = APIRouter()
router.include_router(_inference_router)
router.include_router(_monitor_router)
router.include_router(_analytics_router)
router.include_router(_admin_router)
router.include_router(_playground_router)

__all__ = ["router"]
