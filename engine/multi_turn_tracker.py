"""
engine/multi_turn_tracker.py — Multi-turn adversarial intent drift detection.

Detects Crescendo-style attacks where no single turn looks malicious but
the conversation trajectory escalates toward a harmful goal across turns.

How it works:
  1. Each /monitor call with a conversation_id stores the turn in MongoDB
     (collection: conversation_turns, TTL: 2 hours)
  2. On each new turn, FIE loads the last 7 turns and checks for:
     - REPEATED_REFUSED: user keeps rephrasing a request already flagged adversarial
     - GRADUAL_ESCALATION: multiple distinct concern categories appear across turns
     - PERSISTENT_CONCERN: one concern category appears 4+ times across turns
  3. Returns MultiTurnResult which the /monitor route adds to the response

Usage (routes.py):
    from engine.multi_turn_tracker import check_multi_turn_escalation
    mt_result = check_multi_turn_escalation(
        conversation_id=body.conversation_id,
        prompt=body.prompt,
        question_type=_question_type,
        is_adversarial=jury_is_adversarial,
        adversarial_confidence=jury_confidence,
    )
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Concern keyword categories ────────────────────────────────────────────────
# Each category represents a family of harmful intent.
# Escalation is detected when these accumulate across turns.

_CONCERN_PATTERNS: dict[str, re.Pattern] = {
    "weapons": re.compile(
        r"\b(bomb|explos|weapon|firearm|gun|shoot|kill|poison|nerve\s+agent|"
        r"bioweapon|nuclear|synthesize\s+(?:gas|drug|compound)|detonate|"
        r"plastique|semtex|thermite|ricin|sarin|vx\s+gas)\b",
        re.IGNORECASE,
    ),
    "hacking": re.compile(
        r"\b(hack|exploit|vulnerabilit|malware|ransomware|sql\s*injection|"
        r"cross.?site|xss|buffer\s+overflow|remote\s+code|rce|zero.?day|"
        r"backdoor|rootkit|keylogger|phishing|credential\s+(?:theft|dump|harvest)|"
        r"privilege\s+escalation|lateral\s+movement|c2\s+server|command.and.control)\b",
        re.IGNORECASE,
    ),
    "bypass": re.compile(
        r"\b(bypass|circumvent|jailbreak|unlock\s+(?:mode|restrictions?)|"
        r"remove\s+(?:all\s+)?restrictions?|unrestricted\s+(?:ai|mode)|"
        r"uncensored|unfiltered|without\s+(?:safety|ethical)\s+(?:limits?|filters?)|"
        r"ignore\s+(?:safety|ethical|content)\s+(?:guidelines?|rules?|training))\b",
        re.IGNORECASE,
    ),
    "harm": re.compile(
        r"\b(hurt|harm|injure|stalk|doxx?|swat|blackmail|extort|"
        r"human\s+traffic|smuggl|coerce|threaten|intimidat|assault|"
        r"kidnap|abduct)\b",
        re.IGNORECASE,
    ),
    "illegal_substances": re.compile(
        r"\b(synthesize\s+(?:meth|fentanyl|heroin|mdma|lsd|cocaine)|"
        r"drug\s+(?:synthesis|production|manufacturing)|"
        r"precursor\s+chemical|clandestine\s+lab)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class MultiTurnResult:
    is_escalating: bool
    confidence: float
    pattern: str          # REPEATED_REFUSED | GRADUAL_ESCALATION | PERSISTENT_CONCERN | NONE
    turn_count: int
    evidence: dict = field(default_factory=dict)


def _get_concern_categories(prompt: str) -> list[str]:
    return [cat for cat, pat in _CONCERN_PATTERNS.items() if pat.search(prompt)]


def _get_db_collection():
    """Returns the conversation_turns MongoDB collection, or None if DB unavailable."""
    try:
        from storage.database import _client, _db
        if _db is None:
            return None
        col = _db["conversation_turns"]
        # Ensure TTL index exists (auto-delete docs after 2 hours)
        try:
            col.create_index("timestamp", expireAfterSeconds=7200, background=True)
            col.create_index("conversation_id", background=True)
        except Exception:
            pass
        return col
    except Exception as exc:
        logger.debug("multi_turn_tracker: could not get collection: %s", exc)
        return None


def check_multi_turn_escalation(
    conversation_id: str,
    prompt: str,
    question_type: str,
    is_adversarial: bool,
    adversarial_confidence: float,
) -> MultiTurnResult:
    """
    Store this turn and check if the conversation trajectory is escalating.

    Returns MultiTurnResult. If DB is unavailable, returns a safe no-op result
    so the rest of the pipeline is never blocked.
    """
    col = _get_db_collection()
    if col is None:
        return MultiTurnResult(False, 0.0, "NONE", 1, {})

    concern_categories = _get_concern_categories(prompt)
    now = datetime.now(timezone.utc)

    # Store this turn
    try:
        col.insert_one({
            "conversation_id":      conversation_id,
            "prompt":               prompt[:500],
            "timestamp":            now,
            "question_type":        question_type,
            "concern_categories":   concern_categories,
            "is_adversarial":       is_adversarial,
            "adversarial_confidence": adversarial_confidence,
        })
    except Exception as exc:
        logger.warning("multi_turn_tracker: insert failed: %s", exc)
        return MultiTurnResult(False, 0.0, "NONE", 1, {})

    # Load conversation history (last 8 turns, most recent first)
    cutoff = now - timedelta(hours=2)
    try:
        history = list(col.find(
            {"conversation_id": conversation_id, "timestamp": {"$gte": cutoff}},
            sort=[("timestamp", -1)],
            limit=8,
        ))
    except Exception as exc:
        logger.warning("multi_turn_tracker: history fetch failed: %s", exc)
        return MultiTurnResult(False, 0.0, "NONE", 1, {})

    # Skip current turn (just inserted, first in results)
    history = history[1:]
    turn_count = len(history) + 1

    if len(history) < 2:
        return MultiTurnResult(False, 0.0, "NONE", turn_count, {})

    # ── Signal 1: Repeated adversarial requests ───────────────────────────────
    # User has already been flagged for adversarial activity and keeps trying
    prior_adversarial = [t for t in history if t.get("is_adversarial", False)]
    if prior_adversarial and is_adversarial:
        confidence = round(min(0.75 + len(prior_adversarial) * 0.05, 0.93), 4)
        return MultiTurnResult(
            is_escalating=True,
            confidence=confidence,
            pattern="REPEATED_REFUSED",
            turn_count=turn_count,
            evidence={
                "prior_adversarial_turns": len(prior_adversarial),
                "prior_confidences": [
                    round(t.get("adversarial_confidence", 0.0), 3)
                    for t in prior_adversarial[:3]
                ],
                "history_length": len(history),
            },
        )

    # ── Signal 2: Gradual escalation across multiple concern categories ───────
    # Turn 1: asks about history of explosives (weapons)
    # Turn 2: asks about hacking industrial systems (hacking)
    # Turn 3: current — combines both concerns
    all_cats: list[str] = []
    for t in reversed(history):
        all_cats.extend(t.get("concern_categories", []))
    all_cats.extend(concern_categories)

    unique_cats = set(all_cats)
    total_hits = len(all_cats)

    if len(unique_cats) >= 2 and total_hits >= 4:
        return MultiTurnResult(
            is_escalating=True,
            confidence=0.68,
            pattern="GRADUAL_ESCALATION",
            turn_count=turn_count,
            evidence={
                "concern_categories_seen": sorted(unique_cats),
                "total_concern_hits":      total_hits,
                "history_length":          len(history),
            },
        )

    # ── Signal 3: Persistent single-category concern across many turns ────────
    # User keeps asking about the same harmful topic in different ways
    if total_hits >= 4:
        most_common = max(set(all_cats), key=all_cats.count)
        count = all_cats.count(most_common)
        if count >= 4:
            return MultiTurnResult(
                is_escalating=True,
                confidence=0.62,
                pattern="PERSISTENT_CONCERN",
                turn_count=turn_count,
                evidence={
                    "dominant_category":   most_common,
                    "occurrences":         count,
                    "history_length":      len(history),
                },
            )

    return MultiTurnResult(False, 0.0, "NONE", turn_count, {})
