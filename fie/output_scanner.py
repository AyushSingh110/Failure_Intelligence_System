"""
Output-side adversarial scanner (Flaw 7).

Scans LLM responses for three failure classes:
  1. POLICY_ECHO        — model reports it was jailbroken ("As an AI without restrictions...")
  2. SYSTEM_PROMPT_LEAK — model reveals its system prompt / initial instructions
  3. HARMFUL_OUTPUT     — model produces genuinely harmful step-by-step content

All three gates are O(n) regex scans — sub-millisecond on typical response lengths.
`scan_output_async` runs in a daemon thread so callers pay zero added latency.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("fie.output")


# ── Gate 1: Policy-echo signatures ───────────────────────────────────────────
# These phrases appear in responses from models that were successfully jailbroken.
# They indicate the model is narrating its own "liberated" state.
_POLICY_ECHO_RE = re.compile(
    r"""(?xi)
    \b(?:
        as\s+an?\s+(?:AI|language\s+model|assistant)\s+without\s+(?:restrictions?|guidelines?|limits?|filters?|ethical\s+constraints?) |
        (?:since|now\s+that)\s+(?:you(?:'ve|\s+have)?\s+(?:entered|activated|enabled|switched\s+to|unlocked)|(?:I(?:'m|\s+am)\s+(?:now\s+)?in|we(?:'ve|\s+have)\s+entered)) |
        (?:DAN|developer\s+mode|jailbreak(?:ed)?|unrestricted\s+mode|god\s+mode)\s+(?:is\s+(?:now\s+)?)?(?:active|enabled|on|activated) |
        I\s+(?:no\s+longer\s+have|don't\s+have\s+any)\s+(?:restrictions?|guidelines?|ethical\s+constraints?|safety\s+(?:filters?|guidelines?)) |
        my\s+(?:safety|ethical|content)\s+(?:filters?|guidelines?|protocols?|training)\s+(?:are|have\s+been)\s+(?:disabled?|removed|bypassed|lifted|suspended) |
        in\s+(?:this|the)\s+(?:fictional|hypothetical|alternate|simulation|roleplay)\s+(?:context|world|reality|mode|scenario)[^.]{0,60}I\s+(?:can|will|am\s+able\s+to) |
        (?:ignoring|disregarding|bypassing)\s+(?:all\s+)?(?:my\s+)?(?:previous\s+|prior\s+)?(?:instructions?|guidelines?|restrictions?|constraints?|training) |
        (?:pretending\s+to\s+be|acting\s+as|playing\s+the\s+role\s+of)\s+(?:a\s+)?(?:DAN|evil|unrestricted|unfiltered|uncensored|jailbroken)\b |
        (?:without\s+(?:any\s+)?(?:ethical\s+)?(?:filters?|restrictions?|guidelines?|censorship)|freed\s+from\s+(?:all\s+)?(?:restrictions?|guidelines?|constraints?))
            [^.]{0,80}(?:I\s+(?:can|will|am\s+able|am\s+happy)|here\s+is|here\s+are|certainly|of\s+course|sure)
    )\b
    """,
    re.IGNORECASE,
)

# ── Gate 2: System prompt leakage ────────────────────────────────────────────
# Response contains the system prompt text or reveals it was instructed not to share.
_SYSTEM_PROMPT_LEAK_RE = re.compile(
    r"""(?xi)
    (?:
        (?:my|the)\s+(?:system\s+prompt|initial\s+instructions?|base\s+instructions?|developer\s+(?:message|instructions?))\s+(?:is|are|reads?|states?|says?)\s*[:\"]
        |
        (?:here\s+is|here\s+are|below\s+is|as\s+follows?)\s*(?:[,:]?\s*)?(?:my\s+)?(?:system\s+prompt|initial\s+instructions?|the\s+instructions?\s+I\s+(?:was\s+given|received|have))
        |
        (?:system\s+prompt|base\s+instructions?|my\s+instructions?)\s*:\s*["'\[<\{]
        |
        (?:you\s+are\s+a\s+helpful\s+assistant|you\s+are\s+an?\s+AI\s+assistant)\b[^.]{0,200}(?:was\s+told|instructed\s+not\s+to|must\s+not|cannot\s+reveal)
        |
        <system>\s*.{5,}|SYSTEM\s*:\s*.{5,200}
    )
    """,
    re.IGNORECASE | re.DOTALL,
)

# ── Gate 3: Harmful output content ───────────────────────────────────────────
# Patterns matching actual step-by-step harmful instructions in the response.
# Precision-first design: require both an action verb AND a specific harmful noun
# to avoid flagging general chemistry/security education content.
_OUTPUT_HARM_RE = re.compile(
    r"""(?xi)
    (?:
        # Weapons synthesis steps in response
        (?:step\s+[1-9\d]|first[,\s]|then[,\s]|next[,\s]|finally[,\s])[^.]{0,80}
            (?:synthesize|manufacture|produce|create|combine|mix|prepare)\s+
            (?:sarin|VX\b|novichok|tabun|soman|mustard\s+gas|nerve\s+agent|
               fentanyl|methamphetamine|TATP|HMTD|ricin|anthrax\s+spores?)
        |
        # Improvised explosives assembly
        (?:connect|attach|wire)\s+(?:the\s+)?(?:detonator|blasting\s+cap|initiator)
            [^.]{0,60}(?:to\s+the|with\s+the|and\s+then)\s+(?:charge|device|explosive)
        |
        # CSAM
        (?:sexual|explicit)\s+(?:content|material|scene|image)\s+(?:of|involving|featuring|depicting)\s+
            (?:a\s+)?(?:child|minor|kid|underage|pre-?teen|juvenile)
        |
        # Credential-stealing tutorial
        (?:harvest|steal|capture|exfiltrate)\s+(?:user\s+)?(?:credentials?|passwords?|tokens?|cookies?)
            \s+(?:by|using|via|through)\s+(?:creating|deploying|sending|injecting|planting)
        |
        # Malware deployment steps
        (?:(?:to\s+)?(?:install|deploy|execute|run|launch|trigger))\s+
            (?:the\s+)?(?:malware|ransomware|keylogger|backdoor|rootkit|trojan|payload|shellcode)
            \s+(?:on|in|to|onto|into)\s+(?:the\s+)?(?:target|victim|host|system|machine|device)
    )
    """,
    re.IGNORECASE,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OutputScanResult:
    """Result of scanning an LLM response for adversarial output."""
    is_flagged:  bool
    flag_type:   str    # "POLICY_ECHO" | "SYSTEM_PROMPT_LEAK" | "HARMFUL_OUTPUT" | "CLEAN"
    confidence:  float  # 0.0 if CLEAN, otherwise 0.85–0.92
    evidence:    dict   = field(default_factory=dict)
    scan_ms:     float  = 0.0


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_output(
    prompt:     str,
    response:   str,
    session_id: Optional[str] = None,
) -> OutputScanResult:
    """
    Scan an LLM response for post-generation adversarial failures.

    Three sequential gates (fail-fast):
      1. Policy-echo    → is the model narrating its own jailbreak?
      2. System leak    → is the model revealing its system prompt?
      3. Harmful output → does the response contain harmful step-by-step content?

    Returns immediately on first gate match. O(n) — sub-millisecond on typical responses.
    Thread-safe; no shared mutable state.
    """
    t0 = time.monotonic()
    text = response.strip()
    if not text:
        return OutputScanResult(
            is_flagged=False, flag_type="CLEAN", confidence=0.0,
            scan_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    # Gate 1 — policy echo
    m = _POLICY_ECHO_RE.search(text)
    if m:
        return OutputScanResult(
            is_flagged=True,
            flag_type="POLICY_ECHO",
            confidence=0.92,
            evidence={"matched": m.group(0)[:140]},
            scan_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    # Gate 2 — system prompt leak
    m = _SYSTEM_PROMPT_LEAK_RE.search(text)
    if m:
        return OutputScanResult(
            is_flagged=True,
            flag_type="SYSTEM_PROMPT_LEAK",
            confidence=0.88,
            evidence={"matched": m.group(0)[:140]},
            scan_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    # Gate 3 — harmful content in output
    m = _OUTPUT_HARM_RE.search(text)
    if m:
        return OutputScanResult(
            is_flagged=True,
            flag_type="HARMFUL_OUTPUT",
            confidence=0.90,
            evidence={"matched": m.group(0)[:140]},
            scan_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    return OutputScanResult(
        is_flagged=False,
        flag_type="CLEAN",
        confidence=0.0,
        scan_ms=round((time.monotonic() - t0) * 1000, 2),
    )


def scan_output_async(
    prompt:     str,
    response:   str,
    session_id: Optional[str] = None,
    on_flag:    Optional[Callable[["OutputScanResult"], None]] = None,
    model_name: str = "",
) -> None:
    """
    Fire-and-forget output scan in a daemon thread.

    Zero latency impact on the caller — the thread starts and the caller returns
    immediately. If the scan flags the response, it logs a warning and calls
    `on_flag(result)` if provided.

    Args:
        prompt:     Original user prompt (for context in logs).
        response:   LLM response to scan.
        session_id: Optional session ID for cross-turn tracking.
        on_flag:    Optional callback invoked when a flag is detected.
        model_name: Model name for log attribution.
    """
    def _run() -> None:
        try:
            result = scan_output(prompt, response, session_id)
            if result.is_flagged:
                logger.warning(
                    "[FIE:output] FLAGGED | model=%s | type=%s | confidence=%.2f | "
                    "scan_ms=%.1f | matched=%.80s",
                    model_name or "unknown",
                    result.flag_type,
                    result.confidence,
                    result.scan_ms,
                    result.evidence.get("matched", ""),
                )
                if on_flag is not None:
                    try:
                        on_flag(result)
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("[FIE:output] scan_output failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()
