"""
Community feedback on demo scans — the labelled data that actually matters.

WHY THIS EXISTS
---------------
Raw prompt collection is the obvious thing to build and the least useful. It
yields thousands of unlabelled strings, mostly "hello" and copied examples, and
labelling them is the expensive part — which leaves you exactly where you
started.

What FIE needs is one specific human judgement:

    "this prompt was safe, and you blocked it"

Over-refusal is the project's largest documented weakness (53.6% XSTest /
90.4% OR-Bench-hard). Nobody can scrape that label; a person has to assert it.
So this module records only what somebody explicitly submitted, and nothing
else.

PRIVACY
-------
Stores the prompt, the verdict FIE produced, and which correction was reported.
No IP, no cookie, no fingerprint, no account, no session id — submissions cannot
be linked to each other or to a person. This is stated in the public privacy
policy, which was updated BEFORE this module shipped; if you extend what is
stored here, update that page in the same commit or it becomes a false
statement.

STORAGE
-------
MongoDB when reachable, JSONL on disk otherwise. The fallback exists because the
demo runs on a Hugging Face Space whose filesystem is ephemeral — a container
rebuild wipes it — so disk is a last resort for local runs, not the primary
path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION = "demo_feedback"
_FALLBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_feedback.jsonl"

# Report kinds. Deliberately closed — free-text categories become unusable for
# training within a week.
FALSE_POSITIVE = "false_positive"   # safe prompt, FIE blocked it
MISSED_ATTACK = "missed_attack"     # adversarial prompt, FIE allowed it
VALID_KINDS = frozenset({FALSE_POSITIVE, MISSED_ATTACK})

# A public write endpoint needs a ceiling. This is not security — it is a guard
# against one enthusiastic script filling the collection with duplicates.
_MAX_PROMPT_CHARS = 8000

_lock = threading.Lock()
_seen_hashes: set[str] = set()


def _collection():
    """MongoDB collection, or None when unavailable."""
    try:
        from storage.database import get_db
        db = get_db()
        if db is None:
            return None
        col = db[_COLLECTION]
        return col
    except Exception as exc:
        logger.warning(
            "degraded capability=demo_feedback_store impact='falling back to local "
            "JSONL, which is ephemeral on a Space' reason=%s: %s",
            type(exc).__name__, exc,
        )
        return None


def record_feedback(
    prompt: str,
    kind: str,
    scan_result: dict | None = None,
) -> dict:
    """
    Store one opt-in correction.

    Parameters
    ----------
    prompt : str
        The text the user scanned. Stored verbatim — the whole point is to
        capture the exact string that produced a wrong verdict.
    kind : str
        FALSE_POSITIVE or MISSED_ATTACK.
    scan_result : dict, optional
        What FIE returned: is_attack, attack_type, confidence, layer_scores.
        Without it the report is much less useful, because "you got this wrong"
        cannot be acted on if the verdict itself was not captured.

    Returns
    -------
    dict with `ok` (bool) and a user-facing `message`. Never raises: a failure
    to record feedback must not break the demo the user is trying to help with.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "message": "Nothing to report — scan a prompt first."}
    if kind not in VALID_KINDS:
        return {"ok": False, "message": f"Unknown report type: {kind}"}
    if len(prompt) > _MAX_PROMPT_CHARS:
        return {"ok": False, "message": f"Prompt too long (limit {_MAX_PROMPT_CHARS} chars)."}

    # Deduplicate within the process. Someone clicking twice should not create
    # two training examples, and the same well-known prompt reported repeatedly
    # skews class balance.
    fingerprint = hashlib.sha256(f"{kind}\x00{prompt}".encode()).hexdigest()
    with _lock:
        if fingerprint in _seen_hashes:
            return {"ok": True, "message": "Already recorded — thank you."}
        _seen_hashes.add(fingerprint)

    scan_result = scan_result or {}
    record = {
        "prompt":       prompt,
        "kind":         kind,
        "reported_at":  datetime.now(timezone.utc).isoformat(),
        # What FIE said, so the report is actionable.
        "fie_verdict": {
            "is_attack":   bool(scan_result.get("is_attack", False)),
            "attack_type": scan_result.get("attack_type"),
            "confidence":  float(scan_result.get("confidence", 0.0) or 0.0),
            "layers_fired": list(scan_result.get("layers_fired") or []),
            "layer_scores": {
                k: round(float(v), 4)
                for k, v in (scan_result.get("layer_scores") or {}).items()
            },
        },
        # Version the record so a future schema change is distinguishable and
        # old rows stay usable.
        "schema": 1,
        "source": os.getenv("FIE_FEEDBACK_SOURCE", "demo"),
        # Deliberately absent: ip, user agent, session id, account, cookie.
    }

    col = _collection()
    if col is not None:
        try:
            col.insert_one(dict(record))
            logger.info("demo_feedback recorded kind=%s via=mongodb", kind)
            return {"ok": True, "message": _thanks(kind)}
        except Exception as exc:
            logger.warning(
                "demo_feedback: mongo insert failed, falling back to disk (%s: %s)",
                type(exc).__name__, exc,
            )

    try:
        _FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FALLBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("demo_feedback recorded kind=%s via=jsonl", kind)
        return {"ok": True, "message": _thanks(kind)}
    except Exception as exc:
        logger.error("demo_feedback: could not record (%s: %s)", type(exc).__name__, exc)
        return {"ok": False, "message": "Could not record that — sorry. Please open a GitHub issue."}


def _thanks(kind: str) -> str:
    if kind == FALSE_POSITIVE:
        return (
            "Recorded — thank you. False positives are the most useful data this "
            "project can receive, and this one may end up in the public dataset."
        )
    return "Recorded — thank you. Missed attacks feed directly into the next retraining round."


def stats() -> dict:
    """Counts by kind, for the dashboard and the export script."""
    col = _collection()
    if col is not None:
        try:
            return {
                "total":          col.count_documents({}),
                "false_positive": col.count_documents({"kind": FALSE_POSITIVE}),
                "missed_attack":  col.count_documents({"kind": MISSED_ATTACK}),
                "backend":        "mongodb",
            }
        except Exception as exc:
            logger.warning("demo_feedback stats failed: %s", exc)

    if not _FALLBACK_PATH.exists():
        return {"total": 0, "false_positive": 0, "missed_attack": 0, "backend": "jsonl"}

    counts = {FALSE_POSITIVE: 0, MISSED_ATTACK: 0}
    with open(_FALLBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                kind = json.loads(line).get("kind", "")
            except json.JSONDecodeError:
                continue
            if kind in counts:
                counts[kind] += 1
    return {
        "total":          sum(counts.values()),
        "false_positive": counts[FALSE_POSITIVE],
        "missed_attack":  counts[MISSED_ATTACK],
        "backend":        "jsonl",
    }


def export_all() -> list[dict]:
    """Every stored report, newest last. Used by the dataset export script."""
    col = _collection()
    if col is not None:
        try:
            return [
                {k: v for k, v in doc.items() if k != "_id"}
                for doc in col.find({}).sort("reported_at", 1)
            ]
        except Exception as exc:
            logger.warning("demo_feedback export from mongo failed: %s", exc)

    if not _FALLBACK_PATH.exists():
        return []
    out = []
    with open(_FALLBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
