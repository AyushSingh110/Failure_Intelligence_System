"""
app/routes/playground.py

FIE Playground — full pipeline comparison.

Two parallel paths per request
--------------------------------
Raw path  : calls the selected primary model with NO guard, NO correction.
            This is exactly what users receive without FIE.

FIE path  : runs the complete FIE architecture in order:
              1. Pre-flight adversarial guard
              2. Shadow model fan-out (3 Groq models in parallel)
              3. Signal analysis  (agreement score, entropy)
              4. DiagnosticJury   (3 specialist agents)
              5. Correction decision based on jury verdict + signals

Primary model choices
---------------------
* Any preset Groq model (llama, deepseek, qwen, gemma)
* Any custom OpenAI-compatible endpoint (enterprise integration)

Security
--------
* Requires valid session token or API key — 401 otherwise.
* Custom endpoint calls are isolated; credentials never logged.
* Results are NOT persisted to MongoDB — zero noise in analytics.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests as _http
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.auth_guard import require_user

logger = logging.getLogger("fie.playground")

router = APIRouter(tags=["playground"])

_ADVERSARIAL_CAUSES = frozenset({
    "PROMPT_INJECTION", "JAILBREAK_ATTEMPT", "TOKEN_SMUGGLING",
    "INSTRUCTION_OVERRIDE", "INDIRECT_INJECTION",
})

_HALLUCINATION_CAUSES = frozenset({
    "FACTUAL_HALLUCINATION", "TEMPORAL_KNOWLEDGE_CUTOFF",
    "KNOWLEDGE_BOUNDARY_FAILURE", "MODEL_BLIND_SPOT",
})


# ── Schemas ───────────────────────────────────────────────────────────────────

class PlaygroundRequest(BaseModel):
    prompt:          str            = Field(..., min_length=1, max_length=8000)
    primary_model:   str            = Field("llama-3.1-8b-instant", max_length=128)
    custom_endpoint: Optional[str]  = Field(None, max_length=512)
    custom_api_key:  Optional[str]  = Field(None, max_length=256)


class ShadowResult(BaseModel):
    model:      str
    answer:     str
    confidence: str
    latency_ms: float


class PlaygroundResponse(BaseModel):
    # Pre-flight
    preflight_blocked:     bool      = False
    preflight_attack_type: str       = ""
    preflight_confidence:  float     = 0.0
    preflight_layers:      list[str] = []

    # Raw primary model (no guard, no correction)
    raw_response:   str   = ""
    raw_model:      str   = ""
    raw_latency_ms: float = 0.0
    raw_success:    bool  = False

    # FIE-protected response (full pipeline result)
    fie_response:   str   = ""
    fie_status:     str   = "UNAVAILABLE"  # BLOCKED | VALIDATED | CORRECTED | UNAVAILABLE
    fie_latency_ms: float = 0.0

    # Pipeline transparency
    shadow_results:  list[ShadowResult] = []
    jury_verdict:    str                = ""
    jury_confidence: float              = 0.0
    failure_summary: str                = ""
    is_adversarial:  bool               = False
    agreement_score: float              = 0.0
    entropy_score:   float              = 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_preflight(prompt: str) -> tuple[bool, str, float, list[str]]:
    try:
        from fie.preflight import preflight_check
        r = preflight_check(prompt)
        return r.blocked, r.attack_type, r.confidence, r.layers_fired
    except Exception as exc:
        logger.warning("preflight failed (%s) — allowing through", exc)
        return False, "", 0.0, []


def _call_groq(model_name: str, prompt: str, groq) -> tuple[str, str, float, bool]:
    t0 = time.perf_counter()
    try:
        r = groq._call_single_model(model_name, prompt, max_tokens=500, temperature=0.2)
        return r.output_text, r.model_name, (time.perf_counter() - t0) * 1000, r.success
    except Exception as exc:
        logger.error("groq call failed (%s): %s", model_name, exc)
        return "", model_name, (time.perf_counter() - t0) * 1000, False


def _call_custom(endpoint: str, api_key: str, prompt: str) -> tuple[str, str, float, bool]:
    """Call any OpenAI-compatible endpoint the user provides."""
    t0 = time.perf_counter()
    try:
        resp = _http.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.2},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text, "custom-model", (time.perf_counter() - t0) * 1000, True
    except Exception as exc:
        logger.error("custom model call failed: %s", exc)
        return "", "custom-model", (time.perf_counter() - t0) * 1000, False


def _run_shadow_ensemble(prompt: str, groq) -> tuple[str, list[ShadowResult], list[str]]:
    """Fan-out to all shadow models. Returns (best_answer, shadow_details, all_texts)."""
    try:
        results   = groq.fan_out_with_confidence(prompt)
        shadows   = []
        all_texts = []
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
                all_texts.append(r.output_text)
                if r.confidence_weight > best_w:
                    best_w    = r.confidence_weight
                    best_text = r.output_text

        return best_text, shadows, all_texts
    except Exception as exc:
        logger.error("shadow ensemble failed: %s", exc)
        return "", [], []


def _run_signals(raw: str, shadow_texts: list[str]) -> tuple[float, float]:
    """Compute agreement score and entropy across all model outputs."""
    try:
        from engine.detector.consistency import compute_consistency
        from engine.detector.entropy import compute_entropy_from_counts

        all_outputs = [raw] + shadow_texts
        consistency = compute_consistency(all_outputs)
        entropy     = compute_entropy_from_counts(
            consistency["answer_counts"], len(all_outputs)
        )
        return round(consistency.get("agreement_score", 0.0), 3), round(entropy, 3)
    except Exception:
        return 0.0, 0.0


def _run_jury(prompt: str, raw: str, shadow_texts: list[str]):
    """Run DiagnosticJury on all outputs. Returns jury result or None."""
    try:
        from engine.agents.failure_agent import failure_agent
        from app.schemas import DiagnosticRequest

        all_outputs = [raw] + shadow_texts
        return failure_agent.run_diagnostic(
            DiagnosticRequest(prompt=prompt, model_outputs=all_outputs)
        )
    except Exception as exc:
        logger.warning("jury failed: %s", exc)
        return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/playground", response_model=PlaygroundResponse)
def playground(
    body:          PlaygroundRequest,
    authorization: str | None = Header(None),
    x_api_key:     str | None = Header(None, alias="X-API-Key"),
) -> PlaygroundResponse:
    """
    FIE Playground — full pipeline comparison.

    Raw  : primary model with no protection.
    FIE  : preflight → shadow ensemble → signal analysis → DiagnosticJury → correction.

    Supports preset Groq models and any custom OpenAI-compatible endpoint.
    Results are NOT saved to MongoDB.
    """
    require_user(authorization, x_api_key)

    if not body.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")

    is_custom = bool(body.custom_endpoint and body.custom_api_key)
    if body.custom_endpoint and not body.custom_api_key:
        raise HTTPException(status_code=422, detail="custom_api_key required with custom_endpoint.")

    from engine.groq_service import get_groq_service
    groq = get_groq_service()

    if not groq and not is_custom:
        raise HTTPException(status_code=503, detail="Groq service not configured on this server.")

    t_total = time.perf_counter()

    # ── Step 1: Pre-flight ────────────────────────────────────────────────────
    blocked, attack_type, pf_conf, pf_layers = _run_preflight(body.prompt)

    if blocked:
        raw_text = raw_model_name = ""
        raw_latency = 0.0
        raw_ok = False
        if is_custom:
            raw_text, raw_model_name, raw_latency, raw_ok = _call_custom(
                body.custom_endpoint, body.custom_api_key, body.prompt
            )
        elif groq:
            raw_text, raw_model_name, raw_latency, raw_ok = _call_groq(
                body.primary_model, body.prompt, groq
            )
        return PlaygroundResponse(
            preflight_blocked     = True,
            preflight_attack_type = attack_type,
            preflight_confidence  = pf_conf,
            preflight_layers      = pf_layers,
            raw_response          = raw_text,
            raw_model             = raw_model_name or body.primary_model,
            raw_latency_ms        = round(raw_latency, 1),
            raw_success           = raw_ok,
            fie_response          = (
                f"Blocked by FIE pre-flight guard — model was never called. "
                f"Attack: {attack_type} | Confidence: {pf_conf:.0%} | "
                f"Layers: {', '.join(pf_layers) if pf_layers else 'none'}."
            ),
            fie_status            = "BLOCKED",
            fie_latency_ms        = round((time.perf_counter() - t_total) * 1000, 1),
            is_adversarial        = True,
        )

    # ── Step 2: Raw primary + shadow ensemble in parallel ─────────────────────
    def _get_raw():
        if is_custom:
            return _call_custom(body.custom_endpoint, body.custom_api_key, body.prompt)
        return _call_groq(body.primary_model, body.prompt, groq)

    raw_text = raw_model_name = ""
    raw_latency = 0.0
    raw_ok = False
    shadow_best = ""
    shadow_results: list[ShadowResult] = []
    shadow_texts: list[str] = []

    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_raw    = exe.submit(_get_raw)
        fut_shadow = exe.submit(_run_shadow_ensemble, body.prompt, groq) if groq else None

        raw_text, raw_model_name, raw_latency, raw_ok = fut_raw.result()
        if fut_shadow:
            shadow_best, shadow_results, shadow_texts = fut_shadow.result()

    # ── Step 3: Signal analysis + DiagnosticJury ─────────────────────────────
    agreement = entropy = jury_conf = 0.0
    jury_root = summary = ""
    is_adv    = False

    if raw_ok and shadow_texts:
        agreement, entropy = _run_signals(raw_text, shadow_texts)

        jury_result = _run_jury(body.prompt, raw_text, shadow_texts)
        if jury_result and jury_result.jury:
            j          = jury_result.jury
            jury_conf  = j.jury_confidence or 0.0
            summary    = j.failure_summary or ""
            is_adv     = j.is_adversarial or False
            if j.primary_verdict:
                jury_root = j.primary_verdict.root_cause or ""

    # ── Step 4: Determine FIE status and response ─────────────────────────────
    fie_text = shadow_best or raw_text

    if not shadow_best:
        fie_status = "UNAVAILABLE"
        fie_text   = raw_text

    elif is_adv or jury_root in _ADVERSARIAL_CAUSES:
        fie_status = "BLOCKED"
        fie_text   = (
            f"FIE DiagnosticJury flagged this as adversarial "
            f"({jury_root.replace('_', ' ') if jury_root else 'adversarial content'}). "
            f"Jury confidence: {jury_conf:.0%}. "
            f"The response below was blocked before reaching your users."
        )
        is_adv = True

    elif jury_root in _HALLUCINATION_CAUSES or (agreement < 0.4 and entropy > 0.5):
        fie_status = "CORRECTED"
        fie_text   = shadow_best

    else:
        fie_status = "VALIDATED"
        fie_text   = raw_text if raw_ok else shadow_best

    return PlaygroundResponse(
        preflight_blocked     = False,
        preflight_confidence  = pf_conf,
        preflight_layers      = pf_layers,
        raw_response          = raw_text,
        raw_model             = raw_model_name or body.primary_model,
        raw_latency_ms        = round(raw_latency, 1),
        raw_success           = raw_ok,
        fie_response          = fie_text,
        fie_status            = fie_status,
        fie_latency_ms        = round((time.perf_counter() - t_total) * 1000, 1),
        shadow_results        = shadow_results,
        jury_verdict          = jury_root.replace("_", " ") if jury_root else "",
        jury_confidence       = round(jury_conf, 3),
        failure_summary       = summary,
        is_adversarial        = is_adv,
        agreement_score       = agreement,
        entropy_score         = entropy,
    )
