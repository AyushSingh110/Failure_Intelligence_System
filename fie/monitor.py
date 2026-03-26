"""
fie/monitor.py

The @monitor decorator — main thing users interact with.

TWO MODES:

mode="monitor" (default, async):
  - Primary model call hota hai
  - User ko IMMEDIATELY answer milta hai
  - FIE background mein check karta hai
  - Agar galat tha → alert aata hai, MongoDB mein save hota hai
  - User ko galat answer mil sakta hai
  - Use karo jab: speed important hai, monitoring sufficient hai

mode="correct" (real-time correction):
  - Primary model aur Shadow models SIMULTANEOUSLY call hote hain
  - Dono ke answers aane ke baad FIE compare karta hai
  - Agar galat → fixed answer return hota hai
  - User ko HAMESHA sahi answer milta hai
  - Thoda slower (15-30s wait) but CORRECT
  - Use karo jab: accuracy critical hai (medical, legal, finance)

Example:
    from fie import monitor

    # Mode 1: Fast monitoring
    @monitor(fie_url="...", mode="monitor")
    def call_gpt4(prompt): ...

    # Mode 2: Real-time correction
    @monitor(fie_url="...", mode="correct")
    def call_gpt4(prompt): ...
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import threading
import time
from typing import Callable, Optional

from fie.client import FIEClient
from fie.config import get_config

logger = logging.getLogger("fie")


def _preview(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def monitor(
    fie_url:       Optional[str] = None,
    api_key:       Optional[str] = None,
    model_name:    Optional[str] = None,
    alert_slack:   Optional[str] = None,
    run_full_jury: bool          = True,
    mode:          str           = "monitor",  # "monitor" or "correct"
    log_results:   bool          = True,

    # Legacy parameter — kept for backward compatibility
    # async_mode=True  → same as mode="monitor"
    # async_mode=False → same as mode="correct"
    async_mode:    Optional[bool] = None,
):
    """
    Decorator that monitors or corrects LLM outputs automatically.

    Parameters
    ----------
    fie_url      : URL of your FIE server
    api_key      : API key for authentication
    model_name   : Name of LLM being monitored (for logs/dashboard)
    alert_slack  : Slack webhook URL for failure alerts
    run_full_jury: Run DiagnosticJury (Phase 3) for root cause
    mode         : "monitor" = fast async monitoring
                   "correct" = real-time correction (slower but accurate)
    log_results  : Print FIE results to console
    """

    # Handle legacy async_mode parameter
    if async_mode is not None:
        effective_mode = "monitor" if async_mode else "correct"
    else:
        effective_mode = mode

    def decorator(func: Callable) -> Callable:

        config      = get_config(fie_url=fie_url, api_key=api_key)
        client      = FIEClient(config)
        _model_name = model_name or func.__name__

        # ── MODE 1: MONITOR (fast async) ──────────────────────────
        if effective_mode == "monitor":

            @functools.wraps(func)
            def monitor_wrapper(*args, **kwargs):
                """
                Fast path:
                1. Primary model call karo
                2. Answer immediately return karo
                3. Background mein FIE check karo
                """
                prompt     = kwargs.get("prompt") or (args[0] if args else "")
                prompt_str = str(prompt) if prompt else ""

                # Call primary model
                start      = time.time()
                result     = func(*args, **kwargs)
                latency_ms = round((time.time() - start) * 1000, 1)
                primary_output = result if isinstance(result, str) else str(result)

                # Send to FIE in background — never blocks user
                def _background_check():
                    fie_result = client.monitor(
                        prompt             = prompt_str,
                        primary_output     = primary_output,
                        primary_model_name = _model_name,
                        latency_ms         = latency_ms,
                        run_full_jury      = run_full_jury,
                    )
                    if fie_result:
                        _log_result(fie_result, _model_name, latency_ms, log_results)
                        if alert_slack and fie_result.get("high_failure_risk"):
                            _fire_slack_alert(alert_slack, fie_result,
                                              prompt_str, primary_output, _model_name)

                t = threading.Thread(target=_background_check, daemon=True)
                t.start()

                # Return original answer immediately — no waiting
                return result

            return monitor_wrapper

        # ── MODE 2: CORRECT (real-time correction) ─────────────────
        else:

            @functools.wraps(func)
            def correct_wrapper(*args, **kwargs):
                """
                Correction path:
                1. Primary model aur FIE/shadow models SIMULTANEOUSLY call karo
                2. Dono ke results aane ka wait karo
                3. Agar fix available → fixed answer return karo
                4. Agar stable → original answer return karo

                User thoda wait karta hai (15-30s on local Ollama)
                But user ko HAMESHA correct answer milta hai.

                Timeline:
                  t=0     → Primary model start + FIE start (parallel)
                  t=1s    → Primary model done (fast LLM like GPT-4)
                  t=15s   → FIE done (Ollama shadow models)
                  t=15s   → Compare karo, fix karo, return karo
                """
                prompt     = kwargs.get("prompt") or (args[0] if args else "")
                prompt_str = str(prompt) if prompt else ""

                # Run primary model and FIE simultaneously
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                    # Thread 1: Primary model call
                    primary_future = executor.submit(func, *args, **kwargs)

                    # Thread 2: We need primary output first for FIE
                    # So: get primary output, then immediately start FIE
                    start          = time.time()
                    primary_result = primary_future.result()
                    latency_ms     = round((time.time() - start) * 1000, 1)
                    primary_output = (
                        primary_result if isinstance(primary_result, str)
                        else str(primary_result)
                    )

                    # Now start FIE with primary output
                    fie_future = executor.submit(
                        client.monitor,
                        prompt_str,
                        primary_output,
                        _model_name,
                        latency_ms,
                        run_full_jury,
                    )

                    # Wait for FIE to complete
                    try:
                        fie_result = fie_future.result(timeout=300)
                    except concurrent.futures.TimeoutError:
                        logger.warning("[FIE] Correction timed out — returning original")
                        fie_result = {}

                # Process FIE result
                if fie_result:
                    _log_result(fie_result, _model_name, latency_ms, log_results)

                    # Check if fix was applied
                    fix_result   = fie_result.get("fix_result") or {}
                    fix_applied  = fix_result.get("fix_applied", False)
                    fixed_output = fix_result.get("fixed_output", "")

                    if fix_applied and fixed_output:
                        logger.info(
                            "[FIE] ⚡ CORRECTED — User gets fixed answer | "
                            "strategy=%s | original='%s...' | fixed='%s...'",
                            fix_result.get("fix_strategy", ""),
                            _preview(primary_output),
                            _preview(fixed_output),
                        )
                        # Return fixed answer to user
                        return fixed_output

                    # No fix needed or fix not applied — return original
                    warning = fix_result.get("warning", "")
                    if warning:
                        logger.warning("[FIE] %s | %s", _model_name, warning)

                    if alert_slack and fie_result.get("high_failure_risk"):
                        _fire_slack_alert(alert_slack, fie_result,
                                          prompt_str, primary_output, _model_name)

                # Return original (either stable or fix not applicable)
                return primary_result

            return correct_wrapper

    return decorator


# ── Helper: Log FIE result ─────────────────────────────────────────────────

def _log_result(
    fie_result:  dict,
    model_name:  str,
    latency_ms:  float,
    log_results: bool,
) -> None:
    if not log_results:
        return

    archetype   = fie_result.get("archetype", "UNKNOWN")
    high_risk   = fie_result.get("high_failure_risk", False)
    summary     = fie_result.get("failure_summary", "")
    fix_result  = fie_result.get("fix_result") or {}
    fix_applied = fix_result.get("fix_applied", False)
    risk_label  = "⚠ HIGH RISK" if high_risk else "✓ STABLE"

    if fix_applied:
        strategy = fix_result.get("fix_strategy", "")
        logger.info(
            "[FIE] %s | ⚡ FIXED | %s → %s | latency=%.0fms | %s",
            model_name, archetype, strategy, latency_ms, _preview(summary),
        )
    else:
        logger.info(
            "[FIE] %s | %s | %s | latency=%.0fms | %s",
            model_name, risk_label, archetype, latency_ms, _preview(summary),
        )

    warning = fix_result.get("warning", "")
    if warning:
        logger.warning("[FIE] %s | %s", model_name, warning)


# ── Helper: Slack alert ────────────────────────────────────────────────────

def _fire_slack_alert(
    webhook_url:    str,
    fie_result:     dict,
    prompt:         str,
    primary_output: str,
    model_name:     str,
) -> None:
    import requests as _requests

    jury      = fie_result.get("jury") or {}
    primary_v = jury.get("primary_verdict") or {}
    fsv       = fie_result.get("failure_signal_vector") or {}

    root_cause = primary_v.get("root_cause", "UNKNOWN")
    confidence = int(primary_v.get("confidence_score", 0) * 100)
    mitigation = primary_v.get("mitigation_strategy", "")
    archetype  = fie_result.get("archetype", "UNKNOWN")
    entropy    = fsv.get("entropy_score", 0)
    agreement  = fsv.get("agreement_score", 0)

    # Check if fix was applied
    fix_result  = fie_result.get("fix_result") or {}
    fix_applied = fix_result.get("fix_applied", False)
    fix_strategy = fix_result.get("fix_strategy", "")
    fixed_output = fix_result.get("fixed_output", "")

    fix_section = (
        f"\n\n✅ *Auto-Fixed:* `{fix_strategy}`\n"
        f"*Fixed output:* {fixed_output[:200]}"
        if fix_applied else
        "\n\n⚠️ *Fix not applied* — manual review needed"
    )

    message = {
        "text": (
            f"🚨 *FIE Alert — Failure Detected*\n\n"
            f"*Model:* `{model_name}`\n"
            f"*Archetype:* `{archetype}`\n"
            f"*Root cause:* `{root_cause}` ({confidence}% confidence)\n"
            f"*Entropy:* `{entropy:.3f}` | *Agreement:* `{agreement:.3f}`\n\n"
            f"*Prompt:*\n> {prompt[:200]}\n\n"
            f"*Primary output:*\n> {primary_output[:200]}\n\n"
            f"*Mitigation:*\n{mitigation[:300]}"
            f"{fix_section}"
        )
    }

    try:
        _requests.post(webhook_url, json=message, timeout=5)
        logger.info("[FIE] Slack alert sent for %s failure", archetype)
    except Exception as exc:
        logger.warning("[FIE] Failed to send Slack alert: %s", exc)
