"""
Hard-positive collection pipeline for PAIR retraining.

Problem:
  The feedback_store records all blocked prompts, but only stores a 140-char
  excerpt and a prompt hash (for privacy). PAIR training needs full prompt text
  (sentence-transformer embeddings of the full prompt).

  Additionally, UNCERTAIN-zone blocks were not recorded in the feedback_store
  at all — only CLEAR_ATTACK results were. This means the most valuable
  training examples (prompts that passed the confidence threshold but were
  blocked by LlamaGuard or conservative default) were completely invisible.

What this module does:
  1. At scan time: If enabled, stores the full prompt text for UNCERTAIN-zone
     blocks and CLEAR_ATTACK blocks in a local staging file, keyed by event_id.
  2. At label time: When a human labels a flagged event as "true_positive"
     via POST /flags/{id}/label, the staged candidate is confirmed.
  3. Export: Confirmed hard positives can be exported to JSONL for inclusion
     in the next PAIR retraining run via scripts/retrain_pair_v4.py.

Privacy:
  The staging file is LOCAL ONLY (never sent to MongoDB, never committed).
  It is opt-in: staging only happens when FIE_COLLECT_HARD_POSITIVES=1.
  The confirmed file is what gets exported for retraining — review it before
  adding to training data.

Storage:
  data/hard_positive_candidates.jsonl   — staged, awaiting human label
  data/hard_positives_confirmed.jsonl   — confirmed TPs, ready for retraining

Schema (each line is a JSON object):
  {
    "event_id":   str,     # matches feedback_store event id
    "flag_type":  str,     # attack type
    "zone":       str,     # "UNCERTAIN" | "CLEAR_ATTACK"
    "confidence": float,
    "prompt":     str,     # full prompt text
    "staged_at":  str,     # ISO-8601 UTC
    "confirmed_at": str | null,
    "status":     "staged" | "confirmed" | "dismissed",
  }

Enable:
  export FIE_COLLECT_HARD_POSITIVES=1
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger("fie.hard_positive_collector")

_ENABLED = os.environ.get("FIE_COLLECT_HARD_POSITIVES", "").lower() in ("1", "true", "yes")

_DATA_DIR         = Path(os.environ.get("FIE_DATA_DIR", "data"))
_CANDIDATES_PATH  = _DATA_DIR / "hard_positive_candidates.jsonl"
_CONFIRMED_PATH   = _DATA_DIR / "hard_positives_confirmed.jsonl"

_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Write helpers ─────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, record: dict) -> None:
    _ensure_data_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_candidates() -> list[dict]:
    if not _CANDIDATES_PATH.exists():
        return []
    records = []
    try:
        with open(_CANDIDATES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.debug("hard_positive_collector: read_candidates error: %s", exc)
    return records


def _rewrite_candidates(records: list[dict]) -> None:
    _ensure_data_dir()
    try:
        with open(_CANDIDATES_PATH, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("hard_positive_collector: rewrite_candidates error: %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def stage_candidate(
    event_id:   str,
    prompt:     str,
    flag_type:  str,
    confidence: float,
    zone:       Literal["UNCERTAIN", "CLEAR_ATTACK"] = "UNCERTAIN",
) -> None:
    """
    Store a candidate for human review.
    Call at scan time when a block occurs (for both UNCERTAIN and CLEAR_ATTACK zones).
    No-op if FIE_COLLECT_HARD_POSITIVES is not set.
    """
    if not _ENABLED:
        return
    record = {
        "event_id":     event_id,
        "flag_type":    flag_type,
        "zone":         zone,
        "confidence":   round(confidence, 4),
        "prompt":       prompt,
        "staged_at":    _now(),
        "confirmed_at": None,
        "status":       "staged",
    }
    with _write_lock:
        try:
            _append_jsonl(_CANDIDATES_PATH, record)
            logger.debug(
                "hard_positive_collector: staged %s zone=%s type=%s",
                event_id[:8], zone, flag_type,
            )
        except Exception as exc:
            logger.debug("hard_positive_collector: stage_candidate error: %s", exc)


def confirm_hard_positive(event_id: str) -> bool:
    """
    Move a staged candidate to the confirmed file.
    Call when a human labels the corresponding feedback event as 'true_positive'.
    Returns True if found and confirmed.
    """
    if not _ENABLED:
        return False
    with _write_lock:
        candidates = _read_candidates()
        found = None
        updated = []
        for c in candidates:
            if c.get("event_id") == event_id and c.get("status") == "staged":
                c["status"]       = "confirmed"
                c["confirmed_at"] = _now()
                found = c
            else:
                updated.append(c)

        if found is None:
            logger.debug("hard_positive_collector: confirm — event_id %s not found in staged", event_id[:8])
            return False

        # Write updated candidates (without the confirmed one)
        _rewrite_candidates(updated)
        # Append to confirmed file
        try:
            _append_jsonl(_CONFIRMED_PATH, found)
            logger.info(
                "hard_positive_collector: confirmed TP event_id=%s type=%s zone=%s",
                event_id[:8], found.get("flag_type"), found.get("zone"),
            )
        except Exception as exc:
            logger.warning("hard_positive_collector: confirm write error: %s", exc)
        return True


def dismiss_candidate(event_id: str) -> bool:
    """
    Remove a staged candidate (human labeled as false positive).
    Returns True if found and dismissed.
    """
    if not _ENABLED:
        return False
    with _write_lock:
        candidates = _read_candidates()
        found = False
        updated = []
        for c in candidates:
            if c.get("event_id") == event_id and c.get("status") == "staged":
                found = True
            else:
                updated.append(c)

        if found:
            _rewrite_candidates(updated)
            logger.debug("hard_positive_collector: dismissed event_id=%s", event_id[:8])
        return found


def export_for_retraining(output_path: str | None = None) -> list[dict]:
    """
    Return all confirmed hard positives as a list of {prompt, flag_type, zone, ...}.
    Optionally write to a JSONL file at output_path for use in retrain_pair_v4.py.
    """
    if not _CONFIRMED_PATH.exists():
        return []
    confirmed = []
    try:
        with open(_CONFIRMED_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        confirmed.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        logger.warning("hard_positive_collector: export_for_retraining read error: %s", exc)
        return []

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in confirmed:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("hard_positive_collector: exported %d records to %s", len(confirmed), output_path)
        except Exception as exc:
            logger.warning("hard_positive_collector: export write error: %s", exc)

    return confirmed


def get_stats() -> dict:
    """Return counts for staged and confirmed candidates."""
    staged    = sum(1 for c in _read_candidates() if c.get("status") == "staged")
    confirmed = 0
    if _CONFIRMED_PATH.exists():
        try:
            with open(_CONFIRMED_PATH, encoding="utf-8") as f:
                confirmed = sum(1 for line in f if line.strip())
        except Exception:
            pass
    return {
        "enabled":   _ENABLED,
        "staged":    staged,
        "confirmed": confirmed,
        "paths": {
            "candidates": str(_CANDIDATES_PATH),
            "confirmed":  str(_CONFIRMED_PATH),
        },
    }


def is_enabled() -> bool:
    return _ENABLED
