from __future__ import annotations

import logging
from typing import Optional

import requests

from fie.config import FIEConfig

logger = logging.getLogger("fie")


class FIEClient:
    """
    HTTP client for the FIE server.
    """

    def __init__(self, config: FIEConfig) -> None:
        self._config  = config
        self._session = requests.Session()
        if config.api_key:
            self._session.headers.update({
                "X-API-Key":    config.api_key,
                "Content-Type": "application/json",
            })

    #Core: monitor endpoint 

    def monitor(
        self,
        prompt:             str,
        primary_output:     str,
        primary_model_name: str            = "primary",
        latency_ms:         Optional[float] = None,
        run_full_jury:      bool            = True,
    ) -> dict:
        payload = {
            "prompt":             prompt,
            "primary_output":     primary_output,
            "primary_model_name": primary_model_name,
            "run_full_jury":      run_full_jury,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        try:
            response = self._session.post(
                f"{self._config.fie_url}/api/v1/monitor",
                json    = payload,
                timeout = 300,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            logger.warning(
                "[FIE] Cannot connect to FIE server at %s. "
                "Start it with: uvicorn app.main:app --reload",
                self._config.fie_url,
            )
            return {}
        except requests.exceptions.Timeout:
            logger.warning("[FIE] FIE server timed out after 300s.")
            return {}
        except Exception as exc:
            logger.warning("[FIE] monitor() error: %s", exc)
            return {}

    #Feedback / Ground Truth 

    def submit_feedback(
        self,
        request_id:     str,
        is_correct:     bool,
        correct_answer: Optional[str] = None,
        notes:          Optional[str] = None,
    ) -> dict:
        """Submit ground truth feedback for a stored inference.
        Returns dict with keys: status, cache_updated, message
        """
        payload: dict = {"is_correct": is_correct}
        if correct_answer:
            payload["correct_answer"] = correct_answer
        if notes:
            payload["notes"] = notes

        try:
            response = self._session.post(
                f"{self._config.fie_url}/api/v1/feedback/{request_id}",
                json    = payload,
                timeout = 30,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("cache_updated"):
                logger.info(
                    "[FIE] Ground truth cache updated for request %s", request_id
                )
            return result
        except Exception as exc:
            logger.warning("[FIE] submit_feedback() error: %s", exc)
            return {}

    #Inference lookup 

    def get_inference(self, request_id: str) -> dict:
        """
        Fetches a single stored inference by request_id.
        """
        try:
            r = self._session.get(
                f"{self._config.fie_url}/api/v1/inferences/{request_id}",
                timeout = 15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("[FIE] get_inference(%s) error: %s", request_id, exc)
            return {}

    def list_inferences(self) -> list:
        """Returns your most recent stored inferences (newest first)."""
        try:
            r = self._session.get(
                f"{self._config.fie_url}/api/v1/inferences",
                timeout = 15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("[FIE] list_inferences() error: %s", exc)
            return []

    def get_trend(self) -> dict:
        """Returns EMA-based model degradation trend for the tenant."""
        try:
            r = self._session.get(
                f"{self._config.fie_url}/api/v1/trend",
                timeout = 10,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("[FIE] get_trend() error: %s", exc)
            return {}

    # Health check 

    def health_check(self) -> bool:
        """
        Returns True if FIE server is reachable and healthy.
        """
        try:
            r = self._session.get(
                f"{self._config.fie_url}/health",
                timeout = 5,
            )
            return r.status_code == 200
        except Exception:
            return False
