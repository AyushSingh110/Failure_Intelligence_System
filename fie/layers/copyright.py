"""
Layer 12 — copyright reproduction requests.

Pattern-based detector for verbatim-reproduction requests. The ablation
credits this layer with recovering 12.5% -> 100% on its own vector.
Weight 1.0.

Extracted from fie/adversarial.py. Detection logic is unchanged — see
tests/test_detection_golden.py, which pins the exact confidence values.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ── Layer 12: Copyright reproduction detection ────────────────────────────────
# Targets verbatim reproduction requests for copyrighted text — the worst-
# performing HarmBench category (36.2% recall). Patterns cover four families:
#   A. Explicit verbatim commands ("reproduce word for word", "copy exactly")
#   B. Full-work reproduction ("entire chapter", "complete lyrics", "full book")
#   C. Content-type + reproduction verb combinations
#   D. Paraphrase-evasion ("pretend it's your own", "rewrite letter by letter")

_COPYRIGHT_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # A — explicit verbatim/exact copy commands (highest confidence)
    (re.compile(
        r"\b(?:reproduce|copy|transcribe|type\s+out|write\s+out|paste)\b.{0,40}"
        r"\b(?:verbatim|word[\s\-]for[\s\-]word|letter[\s\-]by[\s\-]letter|exactly|character[\s\-]by[\s\-]character)\b",
        re.IGNORECASE,
    ), 0.82, "verbatim_copy_command"),

    (re.compile(
        r"\b(?:verbatim|word[\s\-]for[\s\-]word|letter[\s\-]by[\s\-]letter)\b.{0,40}"
        r"\b(?:reproduce|copy|transcribe|give\s+me|show\s+me|write\s+out)\b",
        re.IGNORECASE,
    ), 0.82, "verbatim_copy_command"),

    # B — full work requests ("the full lyrics", "entire text of", "complete chapter")
    (re.compile(
        r"\b(?:full|entire|complete|whole)\s+(?:text|lyrics|chapter|book|article|"
        r"script|screenplay|transcript|content|passage|excerpt)\b",
        re.IGNORECASE,
    ), 0.72, "full_work_request"),

    (re.compile(
        r"\b(?:give|show|provide|output|print|list|write)\s+(?:me\s+)?(?:the\s+)?"
        r"(?:full|entire|complete|whole)\s+\w+",
        re.IGNORECASE,
    ), 0.68, "give_full_content"),

    # C — content type + reproduction verb (songs, books, articles)
    (re.compile(
        r"\b(?:reproduce|transcribe|copy\s+out|write\s+out|type\s+out)\s+.{0,30}"
        r"\b(?:song|book|novel|article|paper|poem|screenplay|script|lyrics|copyright)\b",
        re.IGNORECASE,
    ), 0.78, "reproduce_copyrighted_work"),

    (re.compile(
        r"\b(?:all\s+(?:the\s+)?(?:lyrics|words|verses|lines|stanzas)\s+(?:of|from|to))\b",
        re.IGNORECASE,
    ), 0.75, "reproduce_lyrics"),

    (re.compile(
        r"\b(?:quote|cite)\s+(?:the\s+)?(?:entire|full|complete|whole)\b",
        re.IGNORECASE,
    ), 0.74, "full_quote"),

    # D — paraphrase-evasion patterns
    (re.compile(
        r"\b(?:pretend|act\s+as\s+if|imagine)\s+.{0,40}"
        r"\b(?:wrote|authored|created|published)\s+(?:it|this|the\s+\w+)\b",
        re.IGNORECASE,
    ), 0.70, "authorship_deception"),

    (re.compile(
        r"\b(?:rewrite|paraphrase)\s+.{0,50}\b(?:word\s+for\s+word|verbatim|exactly)\b",
        re.IGNORECASE,
    ), 0.76, "verbatim_rewrite"),
]

_COPYRIGHT_MITIGATIONS = (
    "A request to reproduce copyrighted content verbatim was detected. "
    "Mitigations: (1) Limit output to short quotations (fair use). "
    "(2) Summarise or paraphrase instead of reproducing exactly. "
    "(3) Direct users to official licensed sources. "
    "(4) Add output length caps and similarity filters against known copyrighted corpora."
)


def _layer_copyright(prompt: str) -> tuple[str | None, float, dict]:
    best_conf = 0.0
    best_match: str | None = None
    best_pattern: str | None = None

    for pattern, conf, label in _COPYRIGHT_PATTERNS:
        m = pattern.search(prompt)
        if m and conf > best_conf:
            best_conf = conf
            best_match = m.group(0)[:80]
            best_pattern = label

    if best_conf < 0.65:
        return None, 0.0, {}

    return "COPYRIGHT_REPRODUCTION", round(best_conf, 4), {
        "matched_text":    best_match,
        "matched_pattern": best_pattern,
        "confidence":      round(best_conf, 4),
    }

