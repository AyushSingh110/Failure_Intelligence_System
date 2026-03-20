import os
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException


def _resolve_api_base() -> str:
    env_override = os.getenv("FIE_API_URL")
    if env_override:
        return env_override.rstrip("/")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from config import get_settings
        s = get_settings()
        host = s.api_host
        if host in ("0.0.0.0", ""):
            host = "127.0.0.1"
        return f"http://{host}:{s.api_port}{s.api_prefix}"
    except Exception:
        return "http://127.0.0.1:8000/api/v1"


API_BASE = _resolve_api_base()
_TIMEOUT_SHORT = 3
_TIMEOUT_LONG  = 12



# Connection check
def check_connection() -> bool:
    try:
        health_url = API_BASE.replace("/api/v1", "") + "/health"
        r = requests.get(health_url, timeout=_TIMEOUT_SHORT)
        return r.status_code == 200
    except (ConnectionError, Timeout, RequestException):
        return False



# Inferences
def fetch_inferences() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/inferences", timeout=_TIMEOUT_LONG)
        r.raise_for_status()
        return r.json()
    except (ConnectionError, Timeout):
        return []
    except RequestException:
        return []


def fetch_questions_with_outputs() -> dict[str, list[dict]]:
    """
    Returns all stored questions grouped by input_text.
    """
    try:
        r = requests.get(
            f"{API_BASE}/inferences/grouped/by-question",
            timeout=_TIMEOUT_LONG,
        )
        r.raise_for_status()
        return r.json()
    except (ConnectionError, Timeout):
        return {}
    except RequestException:
        return {}


def fetch_inference_by_id(request_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/inferences/{request_id}", timeout=_TIMEOUT_LONG)
        r.raise_for_status()
        return r.json()
    except RequestException:
        return None


def delete_inference(request_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/inferences/{request_id}", timeout=_TIMEOUT_SHORT)
        return r.status_code in (200, 204)
    except RequestException:
        return False



# Analysis


def analyze_outputs(model_outputs: list[str]) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/analyze",
            json={"model_outputs": model_outputs},
            timeout=_TIMEOUT_LONG,
        )
        r.raise_for_status()
        return r.json()
    except RequestException:
        return None



# Trend / evolution


def fetch_trend() -> dict:
    try:
        r = requests.get(f"{API_BASE}/trend", timeout=_TIMEOUT_SHORT)
        r.raise_for_status()
        return r.json()
    except RequestException:
        return {}



# Phase 3 — DiagnosticJury


def run_diagnostic(
    prompt:        str,
    model_outputs: list[str],
    latency_ms:    float | None = None,
) -> dict | None:
    """
    Calls POST /diagnose — full Phase 3 pipeline.
    model_outputs[0] = primary, model_outputs[1] = secondary (derived server-side).
    Returns DiagnosticResponse as a dict, or None on error.
    """
    payload: dict = {
        "prompt":        prompt,
        "model_outputs": model_outputs,
    }
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms

    try:
        r = requests.post(
            f"{API_BASE}/diagnose",
            json=payload,
            timeout=_TIMEOUT_LONG,
        )
        r.raise_for_status()
        return r.json()
    except RequestException:
        return None