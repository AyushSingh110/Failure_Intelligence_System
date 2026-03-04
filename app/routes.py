from fastapi import APIRouter
from app.schemas import InferenceRequest
from app.storage import save_inference

router = APIRouter()



@router.post("/track")
def track_inference(request: InferenceRequest):
    success = save_inference(request)

    if not success:
        return {"status": "error", "message": "Failed to store inference"}

    return {
        "status": "stored",
        "request_id": request.request_id
    }