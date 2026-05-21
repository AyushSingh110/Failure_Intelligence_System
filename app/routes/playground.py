"""
app/routes/playground.py

FIE Playground — side-by-side comparison of raw primary model output vs
FIE-protected response.  Lets developers see exactly what FIE intercepts
and corrects before their users ever see it.

Two parallel paths per request
--------------------------------
1. Raw   — calls a small primary model (llama-3.1-8b-instant) with NO guard.
           This is what your users would receive without FIE.
2. FIE   — runs the pre-flight guard first.
           If blocked  → returns guard verdict, LLM never runs.
           If safe     → fan-out to three large shadow models and picks the
                         highest-confidence answer.  If it differs from the
                         raw answer the status is CORRECTED; if they agree
                         it is VALIDATED.

Security notes
--------------
* Requires a valid user session (bearer token or API key).  Anonymous access
  is rejected with 401.
* Rate-limited to 20 req/min to protect Groq TPD quota.
* Playground results are NOT persisted to MongoDB — this is a sandbox tool,
  not a production monitoring path.  It adds zero noise to analytics.
* The raw model call and shadow fan-out run in parallel so total latency is
  bounded by the slowest single model call, not their sum.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.auth_guard import require_user

logger = logging.getLogger("fie.playground")

router = APIRouter(tags=["playground"])

# Small model used as the "raw primary" — intentionally weaker so corrections
# are clearly visible when shadow models disagree.
_RAW_MODEL = "llama-3.1-8b-instant"

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
    "on", "at", "by", "with", "that", "this", "be", "are", "was", "were",
    "i", "you", "he", "she", "we", "they", "do", "does", "did", "have",
    "has", "had", "not", "but", "if", "as", "so", "from", "about",
})


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PlaygroundRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)


class ShadowResult(BaseModel):
    model:      str
    answer:     str
    confidence: str
    latency_ms: float


class PlaygroundResponse(BaseModel):
    # Pre-flight guard
    preflight_blocked:     bool       = False
    preflight_attack_type: str        = ""
    preflight_confidence:  float      = 0.0
    preflight_layers:      list[str]  = []

    # Raw primary model (no guard)
    raw_response:  str   = ""
    raw_model:     str   = ""
    raw_latency_ms: float = 0.0
    raw_success:   bool  = False

    # FIE-protected response
    fie_response:  str   = ""
    fie_status:    str   = "UNAVAILABLE"   # BLOCKED | VALIDATED | CORRECTED | UNAVAILABLE
    fie_latency_ms: float = 0.0

    # Individual shadow model responses (for transparency panel)
    shadow_results: list[ShadowResult] = []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_preflight(prompt: str) -> tuple[bool, str, float, list[str]]:
    try:
        from fie.preflight import preflight_check
        r = preflight_check(prompt)
        return r.blocked, r.attack_type, r.confidence, r.layers_fired
    except Exception as exc:
        logger.warning("playground: preflight check failed (%s) — allowing through", exc)
        return False, "", 0.0, []


def _run_raw_model(prompt: str, groq) -> tuple[str, str, float, bool]:
    t0 = time.perf_counter()
    try:
        r = groq._call_single_model(
            _RAW_MODEL, prompt, max_tokens=500, temperature=0.2,
        )
        return r.output_text, r.model_name, (time.perf_counter() - t0) * 1000, r.success
    except Exception as exc:
        logger.error("playground: raw model call failed: %s", exc)
        return "", _RAW_MODEL, (time.perf_counter() - t0) * 1000, False


def _run_shadow_ensemble(prompt: str, groq) -> tuple[str, list[ShadowResult]]:
    try:
        results   = groq.fan_out_with_confidence(prompt)
        shadows   = []
        best_text = ""
        best_w    = -1.0

        for r in results:
            if r.success and r.output_text:
                shadows.append(ShadowResult(
                    model      = r.model_name,
                    answer     = r.output_text,
                    confidence = r.model_confidence,
                    latency_ms = r.latency_ms,
                ))
                if r.confidence_weight > best_w:
                    best_w    = r.confidence_weight
                    best_text = r.output_text

        return best_text, shadows
    except Exception as exc:
        logger.error("playground: shadow ensemble failed: %s", exc)
        return "", []


def _content_words(text: str) -> set[str]:
    return {
        w for w in text.lower().split()
        if w.isalpha() and w not in _STOPWORDS and len(w) > 2
    }


def _answers_match(raw: str, fie: str) -> bool:
    """Jaccard similarity on content words.  >0.55 → same answer."""
    a = _content_words(raw)
    b = _content_words(fie)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) > 0.55


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/playground", response_model=PlaygroundResponse)
def playground(
    body:          PlaygroundRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> PlaygroundResponse:
    """
    FIE Playground — compare raw primary model output vs FIE-protected response.

    Useful for developers who want to understand exactly what FIE catches and
    corrects before it reaches their users.  Results are NOT saved to MongoDB.
    """
    require_user(authorization, x_api_key)

    if not body.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    from engine.groq_service import get_groq_service
    groq = get_groq_service()

    t_total = time.perf_counter()

    # ── Step 1: Pre-flight guard (fast, synchronous) ──────────────────────────
    blocked, attack_type, pf_conf, pf_layers = _run_preflight(body.prompt)

    if blocked:
        raw_text, raw_model_name, raw_latency, raw_ok = ("", _RAW_MODEL, 0.0, False)
        if groq:
            raw_text, raw_model_name, raw_latency, raw_ok = _run_raw_model(body.prompt, groq)

        fie_msg = (
            f"Prompt blocked by FIE pre-flight guard — the model was never called. "
            f"Attack type: {attack_type}. "
            f"Confidence: {pf_conf:.0%}. "
            f"Layers fired: {', '.join(pf_layers) if pf_layers else 'none'}."
        )
        return PlaygroundResponse(
            preflight_blocked     = True,
            preflight_attack_type = attack_type,
            preflight_confidence  = pf_conf,
            preflight_layers      = pf_layers,
            raw_response          = raw_text,
            raw_model             = raw_model_name,
            raw_latency_ms        = round(raw_latency, 1),
            raw_success           = raw_ok,
            fie_response          = fie_msg,
            fie_status            = "BLOCKED",
            fie_latency_ms        = round((time.perf_counter() - t_total) * 1000, 1),
        )

    if not groq:
        raise HTTPException(status_code=503, detail="Groq service not configured on this server.")

    # ── Step 2: Raw + shadow ensemble in parallel ─────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_raw    = exe.submit(_run_raw_model, body.prompt, groq)
        fut_shadow = exe.submit(_run_shadow_ensemble, body.prompt, groq)
        raw_text, raw_model_name, raw_latency, raw_ok = fut_raw.result()
        fie_text, shadow_results                       = fut_shadow.result()

    fie_latency = (time.perf_counter() - t_total) * 1000

    if not fie_text:
        fie_status = "UNAVAILABLE"
    elif raw_ok and _answers_match(raw_text, fie_text):
        fie_status = "VALIDATED"
    else:
        fie_status = "CORRECTED"

    return PlaygroundResponse(
        preflight_blocked     = False,
        preflight_attack_type = "",
        preflight_confidence  = pf_conf,
        preflight_layers      = pf_layers,
        raw_response          = raw_text,
        raw_model             = raw_model_name,
        raw_latency_ms        = round(raw_latency, 1),
        raw_success           = raw_ok,
        fie_response          = fie_text,
        fie_status            = fie_status,
        fie_latency_ms        = round(fie_latency, 1),
        shadow_results        = shadow_results,
    )
