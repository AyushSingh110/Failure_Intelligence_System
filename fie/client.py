"""
fie/client.py

Makes HTTP calls to your FIE FastAPI server.

This is the only file that actually talks to the network.
Everything else in the SDK calls this file.

The key design principle: NEVER crash the user's app.
If your FIE server is down or slow, the user's LLM call
must still succeed. Every network call is wrapped in
try/except and silently logs errors instead of raising them.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from fie.config import FIEConfig

logger = logging.getLogger("fie")


class FIEClient:
    """
    HTTP client for the FIE server.

    Usage:
        client = FIEClient(config)
        result = client.monitor(
            prompt="What is the capital of France?",
            primary_output="Paris",
            primary_model_name="gpt-4",
        )
    """

    def __init__(self, config: FIEConfig) -> None:
        self._config = config
        self._session = requests.Session()

        # Add API key to every request header if provided
        if config.api_key:
            self._session.headers.update({
                "X-API-Key": config.api_key,
                "Content-Type": "application/json",
            })

    def monitor(
        self,
        prompt:             str,
        primary_output:     str,
        primary_model_name: str           = "primary",
        latency_ms:         Optional[float] = None,
        run_full_jury:      bool           = True,
    ) -> dict:
        """
        Sends one inference to the FIE /monitor endpoint.

        Returns the full FIE analysis result as a dict.
        Returns an empty dict if the server is unreachable —
        this means the user's app never crashes.

        Parameters
        ----------
        prompt             : the original user prompt sent to the LLM
        primary_output     : what the user's LLM returned
        primary_model_name : name of the user's model (for logging)
        latency_ms         : how long the LLM call took (optional)
        run_full_jury      : whether to run the DiagnosticJury (Phase 3)
                             Set False for faster analysis without root cause
        """
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
                json=payload,
                timeout=300,  # 300 seconds 
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            logger.warning(
                "[FIE] Could not connect to FIE server at %s. "
                "Is the server running? Start with: uvicorn app.main:app",
                self._config.fie_url,
            )
            return {}

        except requests.exceptions.Timeout:
            logger.warning(
                "[FIE] Request to FIE server timed out after 30s. "
                "Server may be overloaded."
            )
            return {}

        except Exception as exc:
            logger.warning("[FIE] Unexpected error sending to FIE server: %s", exc)
            return {}

    def health_check(self) -> bool:
        """
        Checks if the FIE server is reachable.
        Returns True if healthy, False otherwise.

        Usage:
            if not client.health_check():
                print("FIE server is not running")
        """
        try:
            r = self._session.get(
                f"{self._config.fie_url}/health",
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False