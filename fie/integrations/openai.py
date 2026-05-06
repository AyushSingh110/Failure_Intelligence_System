"""
fie.integrations.openai — FIE-wrapped OpenAI client.

Drop-in replacement for openai.OpenAI. Every chat completion is:
  1. Scanned for adversarial attacks BEFORE the API call (blocks malicious prompts)
  2. Monitored by FIE AFTER the response (background thread, never adds latency)
  3. Optionally corrected if FIE detects a hallucination (mode="correct")

Usage:
    from fie.integrations import openai

    client = openai.Client(
        api_key     = "sk-...",           # your OpenAI key
        fie_url     = "https://...",      # FIE server URL
        fie_api_key = "fie-...",          # FIE API key
        mode        = "monitor",          # "monitor" | "correct" | "local"
        alert_slack = "https://hooks...", # optional Slack webhook
    )

    # Works exactly like openai.OpenAI — no other changes needed
    response = client.chat.completions.create(
        model    = "gpt-4o",
        messages = [{"role": "user", "content": "Who invented the telephone?"}],
    )
    print(response.choices[0].message.content)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("fie.integrations.openai")


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the last user message as the prompt string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle vision-style content blocks
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""


def _extract_response_text(response) -> str:
    """Pull text from an openai ChatCompletion response object."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


class _CompletionsProxy:
    """Proxies client.chat.completions so attribute access still works."""

    def __init__(self, owner: "Client"):
        self._owner = owner

    def create(self, *, messages: list[dict], model: str, **kwargs):
        return self._owner._monitored_create(
            messages=messages, model=model, **kwargs
        )


class _ChatProxy:
    def __init__(self, owner: "Client"):
        self.completions = _CompletionsProxy(owner)


class Client:
    """
    FIE-wrapped OpenAI client. Accepts all the same kwargs as openai.OpenAI.

    Extra kwargs (all optional):
        fie_url     : FIE server base URL
        fie_api_key : FIE API key
        mode        : "monitor" (background, default) | "correct" | "local"
        alert_slack : Slack webhook URL for high-risk alerts
        run_full_jury: pass to FIE server (default True)
        block_attacks: if True, raise ValueError when adversarial attack detected
                       instead of letting it through (default False)
    """

    def __init__(
        self,
        api_key:       str,
        fie_url:       Optional[str] = None,
        fie_api_key:   Optional[str] = None,
        mode:          str           = "monitor",
        alert_slack:   Optional[str] = None,
        run_full_jury: bool          = True,
        block_attacks: bool          = False,
        **openai_kwargs,
    ):
        try:
            import openai as _openai
            self._openai = _openai.OpenAI(api_key=api_key, **openai_kwargs)
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        self._fie_url       = fie_url
        self._fie_api_key   = fie_api_key
        self._mode          = mode
        self._alert_slack   = alert_slack
        self._run_full_jury = run_full_jury
        self._block_attacks = block_attacks
        self._fie_client    = None

        if fie_url and fie_api_key:
            try:
                from fie.config import FIEConfig
                from fie.client import FIEClient
                self._fie_client = FIEClient(FIEConfig(fie_url=fie_url, api_key=fie_api_key))
            except Exception as exc:
                logger.warning("FIE client init failed (monitoring disabled): %s", exc)

        self.chat = _ChatProxy(self)

    def _monitored_create(self, *, messages: list[dict], model: str, **kwargs):
        prompt = _extract_prompt(messages)

        # Step 1: Adversarial scan BEFORE the API call
        if prompt:
            try:
                from fie.adversarial import scan_prompt
                attack = scan_prompt(prompt)
                if attack.is_attack:
                    logger.warning(
                        "[FIE:openai] ADVERSARIAL ATTACK detected | type=%s | confidence=%.2f | "
                        "layers=%s | matched=%r",
                        attack.attack_type, attack.confidence,
                        ",".join(attack.layers_fired),
                        (attack.matched_text or "")[:80],
                    )
                    if self._block_attacks:
                        raise ValueError(
                            f"FIE blocked adversarial prompt: {attack.attack_type} "
                            f"(confidence={attack.confidence:.2f})"
                        )
            except ImportError:
                pass
            except ValueError:
                raise
            except Exception as exc:
                logger.debug("FIE adversarial scan error (non-fatal): %s", exc)

        # Step 2: Call OpenAI
        start    = time.time()
        response = self._openai.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        latency_ms     = round((time.time() - start) * 1000, 1)
        primary_output = _extract_response_text(response)

        # Step 3: Send to FIE server
        if self._fie_client and primary_output and prompt:
            if self._mode == "correct":
                self._run_fie_correct(prompt, primary_output, model, latency_ms, response)
            else:
                t = threading.Thread(
                    target=self._run_fie_background,
                    args=(prompt, primary_output, model, latency_ms),
                    daemon=True,
                )
                t.start()

        return response

    def _run_fie_background(
        self, prompt: str, primary_output: str, model: str, latency_ms: float
    ) -> None:
        try:
            result = self._fie_client.monitor(
                prompt             = prompt,
                primary_output     = primary_output,
                primary_model_name = model,
                latency_ms         = latency_ms,
                run_full_jury      = self._run_full_jury,
            )
            if result:
                _log_fie_result(result, model)
                if self._alert_slack and result.get("high_failure_risk"):
                    from fie.monitor import _fire_slack_alert
                    _fire_slack_alert(self._alert_slack, result, prompt, primary_output, model)
        except Exception as exc:
            logger.debug("FIE background monitor error (non-fatal): %s", exc)

    def _run_fie_correct(
        self, prompt: str, primary_output: str, model: str, latency_ms: float, response
    ) -> None:
        try:
            result = self._fie_client.monitor(
                prompt             = prompt,
                primary_output     = primary_output,
                primary_model_name = model,
                latency_ms         = latency_ms,
                run_full_jury      = self._run_full_jury,
            )
            if result:
                _log_fie_result(result, model)
                fix = result.get("fix_result") or {}
                if fix.get("fix_applied") and fix.get("fixed_output"):
                    try:
                        response.choices[0].message.content = fix["fixed_output"]
                        logger.info(
                            "[FIE:openai] CORRECTED | strategy=%s | model=%s",
                            fix.get("fix_strategy", ""), model,
                        )
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("FIE correct mode error (non-fatal): %s", exc)


def _log_fie_result(result: dict, model: str) -> None:
    high_risk = result.get("high_failure_risk", False)
    summary   = result.get("failure_summary", "")
    fix       = result.get("fix_result") or {}
    if fix.get("fix_applied"):
        logger.info("[FIE:openai] FIXED | model=%s | strategy=%s | %s",
                    model, fix.get("fix_strategy", ""), summary[:100])
    elif high_risk:
        logger.warning("[FIE:openai] HIGH RISK | model=%s | %s", model, summary[:100])
