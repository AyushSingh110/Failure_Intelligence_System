from __future__ import annotations

import concurrent.futures
import functools
import logging
import threading
import time
from typing import Callable, Optional

from fie.client import FIEClient
from fie.config import get_config
from fie.local_predictor import predict_local

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
    mode:          str           = "monitor", 
    log_results:   bool          = True,

    async_mode:    Optional[bool] = None,
):
    """
    Decorator that monitors or corrects LLM outputs automatically."""

    # Handle legacy async_mode parameter
    if async_mode is not None:
        effective_mode = "monitor" if async_mode else "correct"
    else:
        effective_mode = mode

    def decorator(func: Callable) -> Callable:

        _model_name = model_name or func.__name__

        # MODE 0: LOCAL (no server, rule-based POET predictor)
        if effective_mode == "local":

            @functools.wraps(func)
            def local_wrapper(*args, **kwargs):
                prompt     = kwargs.get("prompt") or (args[0] if args else "")
                prompt_str = str(prompt) if prompt else ""

                result         = func(*args, **kwargs)
                primary_output = result if isinstance(result, str) else str(result)

                prediction = predict_local(prompt_str, primary_output)

                if log_results:
                    risk_label = "⚠ SUSPICIOUS" if prediction.is_suspicious else "✓ STABLE"
                    logger.info(
                        "[FIE:local] %s | %s | qt=%s | confidence=%.2f | signals=%s",
                        _model_name, risk_label,
                        prediction.question_type,
                        prediction.confidence,
                        prediction.signals,
                    )

                return result  # local mode never modifies the output

            return local_wrapper

        config      = get_config(fie_url=fie_url, api_key=api_key)
        client      = FIEClient(config)

        # MODE 1: MONITOR (fast async)
        if effective_mode == "monitor":

            @functools.wraps(func)
            def monitor_wrapper(*args, **kwargs):
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
                        # Opt-in telemetry — anonymized, fire-and-forget
                        fsv = fie_result.get("failure_signal_vector") or {}
                        client._send_telemetry("monitor_call", {
                            "high_failure_risk": fie_result.get("high_failure_risk", False),
                            "fix_applied":       (fie_result.get("fix_result") or {}).get("fix_applied", False),
                            "question_type":     fsv.get("question_type", "UNKNOWN"),
                            "model_version":     fie_result.get("model_version", ""),
                            "mode":              "monitor",
                        })

                t = threading.Thread(target=_background_check, daemon=True)
                t.start()

                # Return original answer immediately — no waiting
                return result

            return monitor_wrapper

        # MODE 2: CORRECT (real-time correction)
        else:

            @functools.wraps(func)
            def correct_wrapper(*args, **kwargs):

                prompt     = kwargs.get("prompt") or (args[0] if args else "")
                prompt_str = str(prompt) if prompt else ""

                # Run primary model and FIE simultaneously
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                    # Thread 1:Primary model call
                    primary_future = executor.submit(func, *args, **kwargs)

                    # Thread 2:We need primary output first for FIE
                    # So: get primary output, then immediately start FIE
                    start          = time.time()
                    primary_result = primary_future.result()
                    latency_ms     = round((time.time() - start) * 1000, 1)
                    primary_output = (
                        primary_result if isinstance(primary_result, str)
                        else str(primary_result)
                    )
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

                    # Opt-in telemetry — anonymized, fire-and-forget
                    fsv = fie_result.get("failure_signal_vector") or {}
                    fix_r = fie_result.get("fix_result") or {}
                    client._send_telemetry("monitor_call", {
                        "high_failure_risk": fie_result.get("high_failure_risk", False),
                        "fix_applied":       fix_r.get("fix_applied", False),
                        "question_type":     fsv.get("question_type", "UNKNOWN"),
                        "model_version":     fie_result.get("model_version", ""),
                        "mode":              "correct",
                    })

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


#Helper: Log FIE result

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

    # Log ground truth verification result
    gt = fie_result.get("ground_truth") or {}
    if gt:
        gt_source     = gt.get("source", "")
        gt_confidence = gt.get("confidence", 0.0)
        from_cache    = gt.get("from_cache", False)
        if from_cache:
            logger.info(
                "[FIE] %s | GT Cache HIT | confidence=100%% | source=%s",
                model_name, gt_source,
            )
        elif gt_source and gt_source != "none":
            logger.info(
                "[FIE] %s | GT verified | source=%s | confidence=%.0f%%",
                model_name, gt_source, gt_confidence * 100,
            )

    #Log escalation
    requires_review = fie_result.get("requires_human_review", False)
    escalation_reason = fie_result.get("escalation_reason", "")
    if requires_review:
        logger.warning(
            "[FIE] %s | ⚠ HUMAN REVIEW REQUIRED | %s",
            model_name, escalation_reason[:120] if escalation_reason else "Confidence too low to auto-correct",
        )


#Helper: Slack alert
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
        f"\n\n *Auto-Fixed:* `{fix_strategy}`\n"
        f"*Fixed output:* {fixed_output[:200]}"
        if fix_applied else
        "\n\n *Fix not applied* — manual review needed"
    )

    message = {
        "text": (
            f" *FIE Alert — Failure Detected*\n\n"
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
