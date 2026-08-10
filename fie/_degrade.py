"""
Explicit degradation helpers.

THE PROBLEM THIS SOLVES
-----------------------
A hardening pass once rewrote ~105 bare `except Exception: pass` blocks across
this codebase into:

    except Exception:
        logger.warning("Suppressed exception in some_function()", exc_info=True)

That is strictly better than `pass`, but it still fails three ways:

1. It does not say WHAT capability was lost. "Suppressed exception in
   scan_prompt()" tells an operator nothing about whether detection recall just
   dropped or a metrics write failed.
2. It does not say WHY continuing is safe. Every one of those blocks encodes a
   safety argument ("this is telemetry, the verdict is unaffected") that lived
   only in the author's head.
3. It logs a full traceback at WARNING for events that are routine and
   expected — an optional dependency missing in a lite install. Real problems
   drown in that noise.

The distinction that matters in a security tool is between an optional
component failing (degrade and continue) and a load-bearing one failing (say so
loudly, and let the caller apply a fail-secure policy). These helpers force the
author to state which one they mean, at the call site, in one line.

USAGE
-----
    from fie._degrade import degraded

    with degraded("feedback_store", "instant verdict for known prompts",
                  impact="adds latency, verdict unchanged"):
        record_block(prompt)

    # Value-returning form:
    dampen = attempt(
        lambda: get_dampening_factor(prompt, fired),
        default=1.0,
        capability="framing_filter",
        impact="no benign dampening — fails toward blocking",
    )

WHAT NOT TO WRAP
----------------
Do not use these to silence a component the result actually depends on. If a
detection layer fails, the scan is degraded and the caller must be told — see
`ScanResult.degraded_layers`. Swallowing that would turn "not scanned" into
"looks safe", which is the exact failure mode a guardrail must never have.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Exceptions that mean "an optional dependency is not installed". These are a
# supported configuration (lite installs), so they log at INFO with no
# traceback. Anything else is unexpected and keeps its stack trace.
_EXPECTED_ABSENCE = (ImportError, ModuleNotFoundError, FileNotFoundError)


@contextmanager
def degraded(capability: str, provides: str, impact: str, *, logger_: logging.Logger | None = None):
    """
    Run a block whose failure is survivable, and record exactly what was lost.

    Parameters
    ----------
    capability : str
        The component that may fail, e.g. "feedback_store", "llama_guard".
        Used as a stable grep key in logs — keep it short and consistent.
    provides : str
        What that component contributes when it works.
    impact : str
        What the caller loses when it does not. Write this as an operator would
        need to read it during an incident. If you cannot state a benign impact
        here, this block should not be using `degraded()` at all.
    logger_ : logging.Logger, optional
        Log through the caller's module logger so records carry the right name.

    Never re-raises. Exceptions that indicate a missing optional dependency are
    logged at INFO without a traceback; everything else at WARNING with one.
    """
    log = logger_ or logger
    try:
        yield
    except _EXPECTED_ABSENCE as exc:
        log.info(
            "degraded capability=%s reason=unavailable impact=%s detail=%s",
            capability, impact, exc,
        )
    except Exception as exc:
        log.warning(
            "degraded capability=%s provides=%s impact=%s reason=%s: %s",
            capability, provides, impact, type(exc).__name__, exc,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )


def attempt(
    fn: Callable[[], T],
    *,
    default: T,
    capability: str,
    impact: str,
    logger_: logging.Logger | None = None,
) -> T:
    """
    Call `fn`, returning `default` if it fails. The value-returning `degraded()`.

    `default` must be the SAFE value for this call site, not merely a neutral
    one. For anything that scales a confidence score, the safe default is the
    one that fails toward blocking — losing a component must never make a
    prompt look safer than it was actually shown to be.
    """
    log = logger_ or logger
    try:
        return fn()
    except _EXPECTED_ABSENCE as exc:
        log.info(
            "degraded capability=%s reason=unavailable fallback=%r impact=%s detail=%s",
            capability, default, impact, exc,
        )
        return default
    except Exception as exc:
        log.warning(
            "degraded capability=%s fallback=%r impact=%s reason=%s: %s",
            capability, default, impact, type(exc).__name__, exc,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )
        return default
