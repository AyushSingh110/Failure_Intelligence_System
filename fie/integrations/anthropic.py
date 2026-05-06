"""
fie.integrations.anthropic — FIE-wrapped Anthropic client.

Drop-in replacement for anthropic.Anthropic. Every messages.create call is:
  1. Scanned for adversarial attacks BEFORE the API call
  2. Monitored by FIE AFTER the response (background thread, never adds latency)
  3. Optionally corrected if FIE detects a hallucination (mode="correct")

Usage:
    from fie.integrations import anthropic

    client = anthropic.Client(
        api_key     = "sk-ant-...",       # your Anthropic key
        fie_url     = "https://...",      # FIE server URL
        fie_api_key = "fie-...",          # FIE API key
        mode        = "monitor",          # "monitor" | "correct" | "local"
    )

    response = client.messages.create(
        model      = "claude-3-5-sonnet-20241022",
        max_tokens = 1024,
        messages   = [{"role": "user", "content": "Who invented the telephone?"}],
    )
    print(response.content[0].text)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("fie.integrations.anthropic")


def _extract_prompt(messages: list[dict]) -> str:
    """Pull the last user message as the prompt string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""


def _extract_response_text(response) -> str:
    """Pull text from an anthropic Message response object."""
    try:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
    except Exception:
        return ""


class _MessagesProxy:
    def __init__(self, owner: "Client"):
        self._owner = owner

    def create(self, *, messages: list[dict], model: str, **kwargs):
        return self._owner._monitored_create(
            messages=messages, model=model, **kwargs
        )


class Client:
    """
    FIE-wrapped Anthropic client. Accepts all the same kwargs as anthropic.Anthropic.

    Extra kwargs (all optional):
        fie_url     : FIE server base URL
        fie_api_key : FIE API key
        mode        : "monitor" (default) | "correct" | "local"
        alert_slack : Slack webhook URL for high-risk alerts
        block_attacks: raise ValueError when adversarial attack detected (default False)
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
        **anthropic_kwargs,
    ):
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic.Anthropic(api_key=api_key, **anthropic_kwargs)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
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

        self.messages = _MessagesProxy(self)

    def _monitored_create(self, *, messages: list[dict], model: str, **kwargs):
        prompt = _extract_prompt(messages)

        # Step 1: Adversarial scan BEFORE the API call
        if prompt:
            try:
                from fie.adversarial import scan_prompt
                attack = scan_prompt(prompt)
                if attack.is_attack:
                    logger.warning(
                        "[FIE:anthropic] ADVERSARIAL ATTACK detected | type=%s | confidence=%.2f | "
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

        # Step 2: Call Anthropic
        start    = time.time()
        response = self._anthropic.messages.create(
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
                        # Mutate the first text block in-place
                        for block in response.content:
                            if hasattr(block, "text"):
                                object.__setattr__(block, "text", fix["fixed_output"])
                                break
                        logger.info(
                            "[FIE:anthropic] CORRECTED | strategy=%s | model=%s",
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
        logger.info("[FIE:anthropic] FIXED | model=%s | strategy=%s | %s",
                    model, fix.get("fix_strategy", ""), summary[:100])
    elif high_risk:
        logger.warning("[FIE:anthropic] HIGH RISK | model=%s | %s", model, summary[:100])
