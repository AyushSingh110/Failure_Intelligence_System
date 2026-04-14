from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    model_name:  str
    output_text: str
    latency_ms:  float
    success:     bool
    error:       str = ""


class OllamaService:
    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def base_url(self) -> str:
        return self._settings.ollama_base_url.rstrip("/")

    @property
    def models(self) -> list[str]:
        return self._settings.ollama_models

    @property
    def timeout(self) -> int:
        return self._settings.ollama_timeout_seconds

    # Public API

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> list[str]:
        """Returns list of models currently pulled in Ollama."""
        try:
            r    = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"].split(":")[0] for m in data.get("models", [])]
        except Exception:
            return []

    def fan_out(self, prompt: str) -> list[ModelResponse]:
        if not self.is_available():
            logger.warning(
                "Ollama is not running. Start with: ollama serve\n"
                "Returning empty responses — FIE will use only the primary output."
            )
            return []

        available = self.get_available_models()
        models_to_call = [m for m in self.models if m in available]

        if not models_to_call:
            logger.warning(
                "None of the configured models %s are available in Ollama. "
                "Pull them with: ollama pull mistral && ollama pull llama3.2 && ollama pull phi3",
                self.models,
            )
            return []

        logger.debug("Fanning out to models: %s", models_to_call)

        # Call all models in parallel using ThreadPoolExecutor.
        # All models run simultaneously — total time = slowest model.
        results: list[ModelResponse] = []
        with ThreadPoolExecutor(max_workers=len(models_to_call)) as executor:
            futures = {
                executor.submit(self._call_model, model, prompt): model
                for model in models_to_call
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.success:
                    logger.debug(
                        "Model %s responded in %.0fms: %s...",
                        result.model_name, result.latency_ms,
                        result.output_text[:60],
                    )
                else:
                    logger.warning(
                        "Model %s failed: %s",
                        result.model_name, result.error,
                    )

        results.sort(key=lambda r: r.model_name)
        return results

    def fan_out_outputs_only(self, prompt: str) -> list[str]:
        """
        Convenience method — returns only the successful output strings.
        Used directly by the /monitor endpoint.
        """
        results = self.fan_out(prompt)
        return [r.output_text for r in results if r.success]

    # Internal

    def _call_model(self, model_name: str, prompt: str) -> ModelResponse:
        """Calls a single Ollama model and returns its response."""
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  model_name,
                    "prompt": prompt,
                    "stream": False,       # get full response at once
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 256,  # max tokens per response
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data       = response.json()
            output     = data.get("response", "").strip()
            latency_ms = (time.time() - start) * 1000

            return ModelResponse(
                model_name=model_name,
                output_text=output,
                latency_ms=round(latency_ms, 1),
                success=bool(output),
                error="" if output else "Empty response",
            )

        except requests.exceptions.Timeout:
            return ModelResponse(
                model_name=model_name,
                output_text="",
                latency_ms=(time.time() - start) * 1000,
                success=False,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as exc:
            return ModelResponse(
                model_name=model_name,
                output_text="",
                latency_ms=(time.time() - start) * 1000,
                success=False,
                error=str(exc),
            )


#Module-level singleton 
ollama_service = OllamaService()