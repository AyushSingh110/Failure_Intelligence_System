"""
Session context store — auto-threads conversation history per session_id.

When a client sends requests with the same session_id, FIE automatically
stores each turn and injects prior history as context[] for shadow models.
This eliminates the CONTEXT_DEPENDENT misclassification that occurs when
single-turn fragments arrive without their conversation history.

Storage: MongoDB with 24-hour TTL index on `expires_at`.
Fallback: in-memory dict (no persistence across restarts).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_TURNS  = 10   # store at most last 10 turns per session
_TTL_HOURS  = 24   # sessions expire after 24 hours of inactivity

# In-memory fallback when MongoDB is unavailable
_fallback: dict[str, list[dict]] = {}
_fallback_lock = threading.Lock()

# MongoDB collection (lazy init)
_collection = None
_mongo_ok   = False


def _get_collection():
    global _collection, _mongo_ok
    if _collection is not None:
        return _collection

    try:
        from config import get_settings
        from pymongo import MongoClient, ASCENDING
        settings = get_settings()
        if not settings.mongodb_uri:
            return None

        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        db     = client[settings.mongodb_db_name]
        col    = db["session_context"]

        # TTL index — MongoDB auto-deletes docs after expires_at
        col.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
        col.create_index("session_id")

        _collection = col
        _mongo_ok   = True
        logger.info("SessionStore: MongoDB collection ready.")
        return _collection

    except Exception as exc:
        logger.warning("SessionStore: MongoDB unavailable (%s) — using in-memory fallback.", exc)
        _mongo_ok = False
        return None


def store_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn to a session. Keeps only last _MAX_TURNS turns."""
    if not session_id:
        return

    col = _get_collection()
    turn = {"role": role, "content": content[:4000]}  # cap content size

    if col is not None:
        try:
            expires = datetime.now(timezone.utc) + timedelta(hours=_TTL_HOURS)
            # Upsert: push turn, trim to last _MAX_TURNS, reset TTL
            col.update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "turns": {
                            "$each":  [turn],
                            "$slice": -_MAX_TURNS,
                        }
                    },
                    "$set": {"expires_at": expires},
                },
                upsert=True,
            )
        except Exception as exc:
            logger.debug("SessionStore.store_turn MongoDB error: %s", exc)
    else:
        with _fallback_lock:
            if session_id not in _fallback:
                _fallback[session_id] = []
            _fallback[session_id].append(turn)
            _fallback[session_id] = _fallback[session_id][-_MAX_TURNS:]


def get_context(session_id: str, max_turns: int = 5) -> list[dict]:
    """
    Returns the last max_turns turns for session_id.
    Returns empty list if session not found or store unavailable.
    """
    if not session_id:
        return []

    col = _get_collection()

    if col is not None:
        try:
            doc = col.find_one({"session_id": session_id}, {"turns": 1})
            if doc and "turns" in doc:
                return doc["turns"][-max_turns:]
        except Exception as exc:
            logger.debug("SessionStore.get_context MongoDB error: %s", exc)
    else:
        with _fallback_lock:
            turns = _fallback.get(session_id, [])
            return turns[-max_turns:]

    return []


def clear_session(session_id: str) -> None:
    """Remove a session (used in tests or explicit reset)."""
    col = _get_collection()
    if col is not None:
        try:
            col.delete_one({"session_id": session_id})
        except Exception:
            pass
    else:
        with _fallback_lock:
            _fallback.pop(session_id, None)
