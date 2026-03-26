"""
engine/groq_service.py

Groq API ke through shadow models call karta hai.
Ollama ka replacement hai — same interface, same output format.

Groq kyun?
  - Ollama: 15-40 seconds per model (local GPU)
  - Groq:   0.5-2 seconds per model (cloud API)
  - Groq free tier: 14,400 requests/day
  - Internet chahiye, lekin bahut fast hai

"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class GroqModelResponse:
    """
    Single model ka response.
    Same structure as OllamaModelResponse — easy to swap.
    """
    model_name:  str
    output_text: str   = ""
    success:     bool  = False
    latency_ms:  float = 0.0
    error:       str   = ""


# Groq pe available free models
# In teeno ko shadow models ki tarah use karenge
GROQ_MODELS = [
    "llama-3.1-8b-instant",   # fastest — good for simple facts
    "llama-3.3-70b-versatile",
    "llama-3.2-3b-preview",
]

# Groq API endpoint
_MODEL_ALIASES = {
    "mixtral-8x7b-32768": "llama-3.3-70b-versatile",
    "gemma2-9b-it": "llama-3.1-8b-instant",
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqService:
    """
    Groq API se multiple shadow models call karta hai parallel mein.

    Usage:
        service = GroqService(api_key="gsk_xxx")
        results = service.fan_out("Who invented the telephone?")
        for r in results:
            print(r.model_name, r.output_text)
    """

    def __init__(
        self,
        api_key:         str,
        models:          list[str] = None,
        timeout_seconds: int       = 30,
    ) -> None:
        self._api_key  = api_key
        self._models   = self._normalize_models(models or GROQ_MODELS)
        self._timeout  = timeout_seconds

        # Session with auth header — created once, reused for all calls
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        })

    @staticmethod
    def _normalize_models(models: list[str]) -> list[str]:
        normalized: list[str] = []

        for model_name in models:
            resolved_name = _MODEL_ALIASES.get(model_name, model_name)
            if resolved_name != model_name:
                logger.warning(
                    "Remapping deprecated Groq model '%s' to '%s'",
                    model_name,
                    resolved_name,
                )
            if resolved_name not in normalized:
                normalized.append(resolved_name)

        return normalized or GROQ_MODELS

    def _call_single_model(
        self,
        model_name: str,
        prompt:     str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> GroqModelResponse:
        """
        Ek Groq model ko call karta hai aur response return karta hai.

        Groq OpenAI-compatible API use karta hai — same format as GPT-4 API.
        Isliye requests.post se direct call kar sakte hain.
        """
        start = time.time()

        try:
            response = self._session.post(
                GROQ_API_URL,
                json={
                    "model":       model_name,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()

            data        = response.json()
            output_text = data["choices"][0]["message"]["content"].strip()
            latency_ms  = round((time.time() - start) * 1000, 1)

            logger.debug(
                "Groq %s responded in %.0fms: %s...",
                model_name, latency_ms, output_text[:60],
            )

            return GroqModelResponse(
                model_name  = model_name,
                output_text = output_text,
                success     = True,
                latency_ms  = latency_ms,
            )

        except requests.exceptions.Timeout:
            logger.warning("Groq model %s timed out after %ds", model_name, self._timeout)
            return GroqModelResponse(
                model_name = model_name,
                success    = False,
                error      = f"Timeout after {self._timeout}s",
                latency_ms = round((time.time() - start) * 1000, 1),
            )

        except requests.exceptions.HTTPError as exc:
            # Rate limit hit kiya?
            if exc.response.status_code == 429:
                logger.warning("Groq rate limit hit for model %s", model_name)
                error = "Rate limit exceeded — free tier limit reached"
            else:
                response_text = ""
                try:
                    response_text = exc.response.text
                except Exception:
                    response_text = ""
                error = str(exc) if not response_text else f"{exc} | {response_text}"
            return GroqModelResponse(
                model_name = model_name,
                success    = False,
                error      = error,
                latency_ms = round((time.time() - start) * 1000, 1),
            )

        except Exception as exc:
            logger.error("Groq model %s error: %s", model_name, exc)
            return GroqModelResponse(
                model_name = model_name,
                success    = False,
                error      = str(exc),
                latency_ms = round((time.time() - start) * 1000, 1),
            )

    def fan_out(self, prompt: str) -> list[GroqModelResponse]:
        """
        Saare configured Groq models ko PARALLEL mein call karta hai.

        Parallel kyun?
          3 models sequentially: 0.5s + 1.0s + 0.8s = 2.3s total
          3 models parallel:     max(0.5s, 1.0s, 0.8s) = 1.0s total

        Returns list of GroqModelResponse sorted by model name.
        Failed models bhi include hote hain (success=False).
        """
        results: list[GroqModelResponse] = []

        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            futures = {
                executor.submit(self._call_single_model, model, prompt): model
                for model in self._models
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result.success:
                    logger.info(
                        "Groq %s: %.0fms | %s...",
                        result.model_name,
                        result.latency_ms,
                        result.output_text[:50],
                    )
                else:
                    logger.warning(
                        "Groq %s failed: %s",
                        result.model_name,
                        result.error,
                    )

        # Sort by model name for consistent ordering
        results.sort(key=lambda r: r.model_name)
        return results

    def complete(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.1,
    ) -> GroqModelResponse:
        """
        Run a single grounded completion through one Groq model.
        Used by focused workflows like RAG verification/fallback.
        """
        target_model = model_name or self._models[0]
        return self._call_single_model(
            target_model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def is_available(self) -> bool:
        """
        Groq API reachable hai kya? Simple health check.
        Returns True if at least one model responds.
        """
        if not self._api_key:
            return False

        try:
            # Quick test with fastest model
            result = self._call_single_model(
                "llama-3.1-8b-instant",
                "Say 'ok' in one word.",
            )
            return result.success
        except Exception:
            return False


# ── Singleton instance ─────────────────────────────────────────────────────
# Created once when module loads, reused for all requests
# (same pattern as ollama_service.py)

_groq_service_instance: Optional[GroqService] = None


def get_groq_service() -> Optional[GroqService]:
    """
    GroqService ka singleton instance return karta hai.
    Agar GROQ_API_KEY nahi hai config mein → None return karta hai.
    """
    global _groq_service_instance

    if _groq_service_instance is None:
        try:
            from config import get_settings
            settings = get_settings()

            if not settings.groq_api_key:
                logger.warning(
                    "GROQ_API_KEY not set in config. "
                    "Add GROQ_API_KEY to your .env file."
                )
                return None

            _groq_service_instance = GroqService(
                api_key         = settings.groq_api_key,
                models          = settings.groq_models,
                timeout_seconds = settings.groq_timeout_seconds,
            )
            logger.info(
                "GroqService initialized with models: %s",
                _groq_service_instance._models,
            )

        except Exception as exc:
            logger.error("Failed to initialize GroqService: %s", exc)
            return None

    return _groq_service_instance
