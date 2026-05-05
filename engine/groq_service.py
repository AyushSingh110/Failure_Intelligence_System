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
    Single model response.
    """
    model_name:        str
    output_text:       str   = ""
    success:           bool  = False
    latency_ms:        float = 0.0
    error:             str   = ""
    # additions — confidence signal from the model
    model_confidence:  str   = "MEDIUM"   # "HIGH" | "MEDIUM" | "LOW"
    confidence_weight: float = 2.0        # HIGH=3, MEDIUM=2, LOW=1


# Diverse model families to reduce correlated failure.
# Fallback options provided for any model that fails 
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
]
_MODEL_ALIASES = {
    "mixtral-8x7b-32768":  "llama-3.3-70b-versatile",
    "gemma2-9b-it":        "llama-3.3-70b-versatile",
    "llama3-8b-8192":      "llama-3.1-8b-instant",
    "llama-3.2-3b-preview": "llama-3.1-8b-instant",
}

#Suffix appended to each prompt in fan_out_with_confidence().
# ask the model to rate its own certainty so we can weight its vote.
_CONFIDENCE_SUFFIX = (
    "\n\n---\nAfter your answer add exactly one line in this format:\n"
    "CONFIDENCE: HIGH\n"
    "or CONFIDENCE: MEDIUM\n"
    "or CONFIDENCE: LOW\n"
    "Rate HIGH if you are very sure, MEDIUM if somewhat sure, LOW if uncertain."
)

_CONFIDENCE_WEIGHTS = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqService:
    """
    Groq API se multiple shadow models call karta hai parallel mein.
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
        model_name:     str,
        prompt:         str,
        *,
        system_message: Optional[str] = None,
        max_tokens:     int           = 500,
        temperature:    float         = 0.1,
    ) -> GroqModelResponse:
        """
        single model is called with the prompt, and response is returned as GroqModelResponse.
        Errors are caught and returned in the response object.
        Optional system_message is injected as the system role — used by the
        canary tracker to detect prompt exfiltration across shadow models.
        """
        start = time.time()

        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._session.post(
                GROQ_API_URL,
                json={
                    "model":       model_name,
                    "messages":    messages,
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

    def fan_out(
        self,
        prompt:         str,
        system_message: Optional[str] = None,
    ) -> list[GroqModelResponse]:
        """
        Returns list of GroqModelResponse sorted by model name.
        """
        results: list[GroqModelResponse] = []

        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            futures = {
                executor.submit(
                    self._call_single_model, model, prompt,
                    system_message=system_message,
                ): model
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

    # Confidence-signal helpers 

    @staticmethod
    def _parse_confidence(raw_text: str) -> tuple[str, str]:
        """
        Strips the CONFIDENCE line from model output and returns
        (cleaned_output, confidence_level).
        Model is asked to append "CONFIDENCE: HIGH/MEDIUM/LOW" on
        a new line. This method finds that line, extracts the level,
        and removes it from the output text so the caller gets a
        clean answer.
        Returns ("MEDIUM", cleaned_text) if no CONFIDENCE line found.
        """
        lines = raw_text.strip().split("\n")
        confidence = "MEDIUM"
        clean_lines = []

        for line in reversed(lines):
            stripped = line.strip().upper()
            if stripped.startswith("CONFIDENCE:"):
                level = stripped.replace("CONFIDENCE:", "").strip()
                if level in ("HIGH", "MEDIUM", "LOW"):
                    confidence = level
                # Do not add this line to clean output
                continue
            clean_lines.insert(0, line)

        cleaned = "\n".join(clean_lines).strip()
        return confidence, cleaned

    def fan_out_with_confidence(
        self,
        prompt:         str,
        system_message: Optional[str] = None,
    ) -> list[GroqModelResponse]:
        """
        Appends a confidence-request suffix to the prompt so each shadow model
        rates its own certainty. Pass system_message to inject a canary token
        into the shadow models' system prompt for exfiltration detection.
        """
        confidenced_prompt = prompt + _CONFIDENCE_SUFFIX
        raw_results = self.fan_out(confidenced_prompt, system_message=system_message)

        enriched: list[GroqModelResponse] = []
        for r in raw_results:
            if r.success and r.output_text:
                level, cleaned = self._parse_confidence(r.output_text)
                weight = _CONFIDENCE_WEIGHTS.get(level, 2.0)
                enriched.append(GroqModelResponse(
                    model_name        = r.model_name,
                    output_text       = cleaned,
                    success           = r.success,
                    latency_ms        = r.latency_ms,
                    error             = r.error,
                    model_confidence  = level,
                    confidence_weight = weight,
                ))
            else:
                enriched.append(r)  # failed response — keep as-is

        logger.info(
            "fan_out_with_confidence: %d responses | confidences=%s",
            len([r for r in enriched if r.success]),
            [r.model_confidence for r in enriched if r.success],
        )
        return enriched

    def is_available(self) -> bool:
        """
        Groq API reachable? Simple health check.
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


# Singleton instance 
# Created once when module loads, reused for all requests
_groq_service_instance: Optional[GroqService] = None

def get_groq_service() -> Optional[GroqService]:
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
