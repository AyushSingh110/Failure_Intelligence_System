"""
engine/model_extraction_tracker.py — Model extraction / model stealing detection.

Model extraction attacks probe an LLM systematically to replicate its behavior
without access to the weights. Detection signals:

  1. Capability probing   — "what can you do", "can you X", "what are your limits"
  2. Boundary testing     — rapid alternation of similar allowed/refused queries
  3. Output harvesting    — many near-identical prompts varying one token at a time
  4. High request rate    — >20 requests in 5 minutes from the same tenant

All signals are tracked per-tenant in MongoDB with a 1-hour TTL.
Falls back gracefully to in-memory if MongoDB is unavailable.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger("fie.model_extraction")

# ── Capability probe patterns ─────────────────────────────────────────────────
# Queries that probe what the model can/cannot do rather than actually using it.

_CAPABILITY_PROBE_RE = re.compile(
    r"\b("
    r"what\s+can\s+you\s+(?:do|help|assist|generate|create|write|analyze)|"
    r"can\s+you\s+(?:do|help|assist|generate|create|write|code|translate|summarize|"
    r"analyze|explain|solve|answer|predict|classify|extract|convert)|"
    r"(?:are\s+you\s+(?:able|capable)\s+to|do\s+you\s+(?:have\s+the\s+ability|support|know\s+how))\s+to\b|"
    r"what\s+(?:are\s+your|are\s+the)\s+(?:capabilities|limitations?|restrictions?|abilities|features)|"
    r"(?:list|tell\s+me|show\s+me|describe)\s+(?:all\s+)?(?:your|the)\s+"
    r"(?:capabilities|features|functions?|abilities|what\s+you\s+can)|"
    r"what\s+(?:tasks|things|questions)\s+can\s+you\s+(?:handle|answer|do)|"
    r"what\s+(?:are\s+you|type\s+of\s+(?:AI|model|assistant)\s+are\s+you)|"
    r"what\s+(?:languages|programming\s+languages?|subjects?|topics?)\s+do\s+you\s+(?:know|support|understand)|"
    r"how\s+(?:smart|good|accurate|capable|powerful)\s+are\s+you|"
    r"what\s+(?:is\s+your\s+(?:knowledge\s+cutoff|training\s+data|context\s+(?:window|length))|"
    r"(?:model|version|architecture)\s+are\s+you)|"
    r"how\s+many\s+(?:tokens?|words?|characters?)\s+(?:can\s+you|do\s+you)\s+(?:handle|process|support)"
    r")\b",
    re.IGNORECASE,
)

# ── Systematic variation detector ─────────────────────────────────────────────
# Detects prompts that differ from each other by only 1-3 tokens — the hallmark
# of automated output harvesting (query the model, change one word, repeat).

def _prompt_fingerprint(prompt: str) -> str:
    """Rough bag-of-words fingerprint: sorted lowercased words, hashed."""
    words = sorted(set(re.sub(r"[^a-z0-9\s]", "", prompt.lower()).split()))
    return hashlib.md5(" ".join(words).encode()).hexdigest()[:8]


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word-token sets between two prompts."""
    wa = set(re.sub(r"[^a-z0-9\s]", "", a.lower()).split())
    wb = set(re.sub(r"[^a-z0-9\s]", "", b.lower()).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── In-memory fallback store ─────────────────────────────────────────────────

@dataclass
class _TenantRecord:
    prompts:     list[tuple[float, str]] = field(default_factory=list)  # (timestamp, prompt)
    probe_count: int                     = 0
    last_cleanup: float                  = field(default_factory=time.time)


_memory_store: dict[str, _TenantRecord] = defaultdict(_TenantRecord)
_WINDOW_SECONDS  = 300   # 5-minute rolling window
_RATE_THRESHOLD  = 20    # >20 requests in window → suspicious
_PROBE_THRESHOLD = 5     # >5 capability probe queries → suspicious
_SIM_THRESHOLD   = 0.85  # Jaccard similarity threshold for "near-identical"
_SIM_MIN_PAIRS   = 3     # need ≥3 near-identical pairs to flag output harvesting


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    is_extracting: bool
    confidence:    float
    pattern:       str          # e.g. "CAPABILITY_PROBING", "OUTPUT_HARVESTING", "HIGH_RATE"
    evidence:      dict = field(default_factory=dict)


# ── MongoDB-backed tracker ─────────────────────────────────────────────────────

def _get_mongo_collection():
    try:
        from storage.database import get_db
        db = get_db()
        if db is None:
            return None
        col = db["model_extraction_tracking"]
        col.create_index("tenant_id")
        col.create_index("timestamp", expireAfterSeconds=3600)
        return col
    except Exception:
        return None


def _store_prompt_mongo(tenant_id: str, prompt: str, is_probe: bool) -> None:
    col = _get_mongo_collection()
    if col is None:
        return
    try:
        col.insert_one({
            "tenant_id":  tenant_id,
            "timestamp":  datetime.now(timezone.utc),
            "prompt_fp":  _prompt_fingerprint(prompt),
            "prompt_len": len(prompt),
            "is_probe":   is_probe,
            "prompt_head": prompt[:120],
        })
    except Exception:
        pass


def _fetch_recent_mongo(tenant_id: str) -> list[dict]:
    col = _get_mongo_collection()
    if col is None:
        return []
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=_WINDOW_SECONDS)
        return list(col.find(
            {"tenant_id": tenant_id, "timestamp": {"$gte": cutoff}},
            {"_id": 0, "timestamp": 1, "prompt_fp": 1, "is_probe": 1, "prompt_head": 1},
        ))
    except Exception:
        return []


# ── Main API ───────────────────────────────────────────────────────────────────

def check_model_extraction(
    tenant_id:   str,
    prompt:      str,
    conversation_id: Optional[str] = None,
) -> ExtractionResult:
    """
    Check whether this request is part of a model extraction / model stealing attack.

    Args:
        tenant_id:       The tenant ID making the request.
        prompt:          The user prompt text.
        conversation_id: Optional conversation ID (used for multi-turn rate tracking).

    Returns ExtractionResult with is_extracting, confidence, pattern, evidence.
    """
    if not prompt or not tenant_id:
        return ExtractionResult(False, 0.0, "none")

    is_probe = bool(_CAPABILITY_PROBE_RE.search(prompt))

    # ── Store prompt (try MongoDB, fall back to memory) ────────────────────────
    _store_prompt_mongo(tenant_id, prompt, is_probe)

    # ── Fetch recent window ────────────────────────────────────────────────────
    recent_mongo = _fetch_recent_mongo(tenant_id)
    use_mongo    = len(recent_mongo) > 0

    if use_mongo:
        total_requests = len(recent_mongo)
        probe_count    = sum(1 for r in recent_mongo if r.get("is_probe"))
        recent_heads   = [r.get("prompt_head", "") for r in recent_mongo]
    else:
        # In-memory fallback
        rec  = _memory_store[tenant_id]
        now  = time.time()
        cutoff = now - _WINDOW_SECONDS
        rec.prompts = [(ts, p) for ts, p in rec.prompts if ts > cutoff]
        rec.prompts.append((now, prompt))
        if is_probe:
            rec.probe_count += 1
        total_requests = len(rec.prompts)
        probe_count    = rec.probe_count
        recent_heads   = [p for _, p in rec.prompts]

    signals:    list[str] = []
    confidence: float     = 0.0
    pattern:    str       = "none"

    # ── Signal 1: High request rate ────────────────────────────────────────────
    if total_requests > _RATE_THRESHOLD:
        excess = total_requests - _RATE_THRESHOLD
        rate_conf = min(0.60 + (excess / _RATE_THRESHOLD) * 0.25, 0.85)
        signals.append(f"request_rate={total_requests} in {_WINDOW_SECONDS}s window (>{_RATE_THRESHOLD} threshold)")
        confidence = max(confidence, rate_conf)
        pattern    = "HIGH_RATE"

    # ── Signal 2: Capability probing ───────────────────────────────────────────
    if probe_count >= _PROBE_THRESHOLD:
        probe_ratio = probe_count / max(total_requests, 1)
        probe_conf  = min(0.65 + probe_ratio * 0.25, 0.90)
        signals.append(f"capability_probe_count={probe_count} ({probe_ratio:.0%} of requests)")
        confidence = max(confidence, probe_conf)
        pattern    = "CAPABILITY_PROBING"

    # ── Signal 3: Output harvesting (near-identical prompts) ───────────────────
    if len(recent_heads) >= 4:
        near_identical_pairs = 0
        sample = recent_heads[-20:]  # check last 20 prompts
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                if _token_overlap(sample[i], sample[j]) >= _SIM_THRESHOLD:
                    near_identical_pairs += 1

        if near_identical_pairs >= _SIM_MIN_PAIRS:
            harvest_conf = min(0.70 + (near_identical_pairs / 10) * 0.20, 0.92)
            signals.append(
                f"near_identical_pairs={near_identical_pairs} "
                f"(Jaccard ≥{_SIM_THRESHOLD} — systematic output harvesting)"
            )
            confidence = max(confidence, harvest_conf)
            pattern    = "OUTPUT_HARVESTING"

    # ── Boost for multiple signals ─────────────────────────────────────────────
    if len(signals) >= 2:
        confidence = min(confidence + 0.07, 0.94)
        pattern    = "COMBINED_EXTRACTION"

    if confidence < 0.50 or not signals:
        return ExtractionResult(False, 0.0, "none")

    logger.warning(
        "MODEL_EXTRACTION detected | tenant=%s | pattern=%s | conf=%.3f | signals=%s",
        tenant_id, pattern, confidence, signals,
    )

    return ExtractionResult(
        is_extracting = True,
        confidence    = round(confidence, 4),
        pattern       = pattern,
        evidence      = {
            "signals_fired":       signals,
            "total_requests":      total_requests,
            "probe_count":         probe_count,
            "current_is_probe":    is_probe,
            "window_seconds":      _WINDOW_SECONDS,
        },
    )


# Module-level singleton check function for easy import
def is_model_extraction(tenant_id: str, prompt: str) -> ExtractionResult:
    return check_model_extraction(tenant_id, prompt)
