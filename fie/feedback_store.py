"""
Feedback store (Flaw: No feedback loop).

Every time scan_prompt blocks a prompt or scan_output flags a response, the event
is recorded here. A human reviewer can then label each event as a true positive (TP)
or false positive (FP). Labels feed back into detection:

  TP confirmation  → add prompt hash to the "known attack" accelerator so future
                     identical prompts skip the layer pipeline and block immediately.
  FP confirmation  → add prompt hash to the per-prompt whitelist so the same prompt
                     is never blocked again; adds the match text to an exclusion list
                     that downstream pattern code can query.

Storage backend (in priority order):
  1. MongoDB  `flagged_events` collection — when server is connected
  2. Local JSON file  `~/.fie/flagged_events.jsonl` — SDK-only / offline use

Both backends are append-only. The file backend uses newline-delimited JSON.
MongoDB stores standard documents with the same schema.

Schema:
  {
    "id":          str (UUID4),
    "kind":        "input_block" | "output_flag",
    "flag_type":   str,          # attack_type or output flag_type
    "confidence":  float,
    "prompt_hash": str,          # SHA-256 of prompt (privacy-preserving)
    "matched":     str,          # excerpt that triggered detection (≤ 140 chars)
    "session_id":  str | null,
    "timestamp":   str,          # ISO-8601 UTC
    "label":       null | "true_positive" | "false_positive",
    "labeled_at":  str | null,
  }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger("fie.feedback")

Label = Literal["true_positive", "false_positive"]

# ── In-process "known attack" accelerator (Flaw: feedback loop) ───────────────
# Confirmed TPs are added here. scan_prompt checks this set before running layers.
# Thread-safe; in-memory only — rebuilt from DB on server restart.
_KNOWN_ATTACK_HASHES: set[str] = set()
_WHITELIST_HASHES:    set[str] = set()
_known_lock = threading.Lock()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode("utf-8", errors="replace")).hexdigest()


def is_known_attack(prompt: str) -> bool:
    """True if a human confirmed this prompt as a TP attack."""
    return _prompt_hash(prompt) in _KNOWN_ATTACK_HASHES


def is_whitelisted(prompt: str) -> bool:
    """True if a human confirmed this prompt was a false positive."""
    return _prompt_hash(prompt) in _WHITELIST_HASHES


# ── Local file backend ─────────────────────────────────────────────────────────

_LOCAL_PATH = Path(os.environ.get("FIE_FEEDBACK_PATH", Path.home() / ".fie" / "flagged_events.jsonl"))


def _local_append(event: dict) -> None:
    try:
        _LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOCAL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as exc:
        logger.debug("feedback_store local write failed: %s", exc)


def _local_read_all() -> list[dict]:
    if not _LOCAL_PATH.exists():
        return []
    events = []
    try:
        with open(_LOCAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return events


def _local_update_label(event_id: str, label: Label) -> bool:
    """Rewrite the JSONL file with the label applied to the matching event."""
    events = _local_read_all()
    found = False
    for ev in events:
        if ev.get("id") == event_id:
            ev["label"]      = label
            ev["labeled_at"] = datetime.now(timezone.utc).isoformat()
            found = True
    if found:
        try:
            with open(_LOCAL_PATH, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
        except Exception as exc:
            logger.debug("feedback_store local rewrite failed: %s", exc)
    return found


# ── MongoDB backend ────────────────────────────────────────────────────────────

def _mongo_collection():
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        return _db["flagged_events"]
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def record(
    kind:       Literal["input_block", "output_flag"],
    flag_type:  str,
    confidence: float,
    prompt:     str,
    matched:    str        = "",
    session_id: Optional[str] = None,
) -> str:
    """
    Store a flagged event. Returns the event ID.
    Writes to MongoDB if available, always appends to local JSONL as backup.
    Never raises.
    """
    event: dict = {
        "id"         : str(uuid.uuid4()),
        "kind"       : kind,
        "flag_type"  : flag_type,
        "confidence" : round(confidence, 4),
        "prompt_hash": _prompt_hash(prompt),
        "matched"    : matched[:140],
        "session_id" : session_id,
        "timestamp"  : datetime.now(timezone.utc).isoformat(),
        "label"      : None,
        "labeled_at" : None,
    }
    try:
        col = _mongo_collection()
        if col is not None:
            col.insert_one({**event})
        else:
            _local_append(event)
    except Exception as exc:
        logger.debug("feedback_store record failed: %s", exc)
        _local_append(event)
    return event["id"]


def apply_label(event_id: str, label: Label) -> bool:
    """
    Label an event as 'true_positive' or 'false_positive'.

    Side effects on confirmation:
      TP → add prompt_hash to _KNOWN_ATTACK_HASHES (immediate fast-path block)
      FP → add prompt_hash to _WHITELIST_HASHES (suppress future block)

    Returns True if the event was found and updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    prompt_hash: str | None = None

    col = _mongo_collection()
    if col is not None:
        doc = col.find_one({"id": event_id})
        if doc:
            prompt_hash = doc.get("prompt_hash")
            col.update_one(
                {"id": event_id},
                {"$set": {"label": label, "labeled_at": now}},
            )
    else:
        updated = _local_update_label(event_id, label)
        if not updated:
            return False
        for ev in _local_read_all():
            if ev.get("id") == event_id:
                prompt_hash = ev.get("prompt_hash")
                break

    if prompt_hash:
        with _known_lock:
            if label == "true_positive":
                _KNOWN_ATTACK_HASHES.add(prompt_hash)
                _WHITELIST_HASHES.discard(prompt_hash)
                logger.info("feedback_store TP confirmed — hash %s added to fast-block list", prompt_hash[:12])
            else:
                _WHITELIST_HASHES.add(prompt_hash)
                _KNOWN_ATTACK_HASHES.discard(prompt_hash)
                logger.info("feedback_store FP confirmed — hash %s added to whitelist", prompt_hash[:12])

    return True


def list_events(
    unlabeled_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return stored events, newest first. Strips MongoDB _id field."""
    col = _mongo_collection()
    if col is not None:
        query = {"label": None} if unlabeled_only else {}
        docs = list(
            col.find(query, {"_id": 0})
               .sort("timestamp", -1)
               .skip(offset)
               .limit(limit)
        )
        return docs
    else:
        events = _local_read_all()
        if unlabeled_only:
            events = [e for e in events if e.get("label") is None]
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return events[offset: offset + limit]


def export_confirmed_tps(output_path: str | None = None) -> list[dict]:
    """
    Export all confirmed true positives for use as training data.
    Returns list of {prompt_hash, flag_type, matched, timestamp}.
    Optionally writes to a JSONL file at output_path.
    """
    col = _mongo_collection()
    if col is not None:
        docs = list(col.find({"label": "true_positive"}, {"_id": 0}))
    else:
        docs = [e for e in _local_read_all() if e.get("label") == "true_positive"]

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
        except Exception as exc:
            logger.warning("export_confirmed_tps write failed: %s", exc)

    return docs


def _load_confirmed_from_db() -> None:
    """
    On server startup, rebuild in-memory fast-path sets from confirmed labels.
    Call once during app lifespan.
    """
    col = _mongo_collection()
    if col is None:
        return
    try:
        with _known_lock:
            for doc in col.find({"label": "true_positive"}, {"prompt_hash": 1}):
                _KNOWN_ATTACK_HASHES.add(doc["prompt_hash"])
            for doc in col.find({"label": "false_positive"}, {"prompt_hash": 1}):
                _WHITELIST_HASHES.add(doc["prompt_hash"])
        logger.info(
            "feedback_store loaded %d known attacks, %d whitelisted from DB",
            len(_KNOWN_ATTACK_HASHES), len(_WHITELIST_HASHES),
        )
    except Exception as exc:
        logger.warning("feedback_store DB load failed: %s", exc)
