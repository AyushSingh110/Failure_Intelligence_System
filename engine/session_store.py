from __future__ import annotations
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_TURNS      = 10   # store at most last 10 turns per session
_TTL_HOURS      = 24   # sessions expire after 24 hours of inactivity
_COMPRESS_AFTER = 8    # trigger summarization when turns reach this count
_KEEP_RAW       = 4    # keep this many recent turns as raw after compression

# In-memory fallback when MongoDB is unavailable
_fallback: dict[str, list[dict]]  = {}
_fallback_summaries: dict[str, str] = {}  # session_id → rolling summary text
_fallback_lock = threading.Lock()

# MongoDB collection (lazy init)
_collection = None
_mongo_ok   = False


# ── Context summarization ─────────────────────────────────────────────────────

def _summarize_turns(turns: list[dict], existing_summary: str = "") -> Optional[str]:
    """
    Compress `turns` into a short text summary using Groq llama-3.1-8b-instant.
    If `existing_summary` is non-empty it is prepended so each call is additive.
    Returns the summary string, or None if the call fails.
    """
    try:
        import os
        import requests as _req

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            try:
                from config import get_settings
                api_key = get_settings().groq_api_key or ""
            except Exception:
                pass
        if not api_key:
            return None

        turns_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:300]}" for t in turns
        )
        prior = f"[Existing summary]: {existing_summary}\n\n" if existing_summary else ""
        prompt = (
            f"{prior}Summarize the following conversation turns in 2-3 concise sentences "
            f"for an AI safety monitoring system. Preserve key intents, topics, and any "
            f"suspicious or adversarial patterns mentioned.\n\n{turns_text}"
        )

        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except Exception as exc:
        logger.debug("SessionStore._summarize_turns failed: %s", exc)
        return None


def _maybe_compress(session_id: str, turns: list[dict], existing_summary: str) -> tuple[list[dict], str]:
    """
    If turns >= _COMPRESS_AFTER, compress oldest turns into summary and return
    (trimmed_turns, updated_summary). Otherwise returns inputs unchanged.
    Compression is fire-and-forget — failure keeps original state.
    """
    if len(turns) < _COMPRESS_AFTER:
        return turns, existing_summary

    to_compress = turns[:-_KEEP_RAW]
    keep_raw    = turns[-_KEEP_RAW:]

    new_summary = _summarize_turns(to_compress, existing_summary)
    if new_summary:
        logger.debug("SessionStore: compressed %d turns for session %s", len(to_compress), session_id)
        return keep_raw, new_summary

    # Summarization failed — trim to _MAX_TURNS and keep going
    return turns[-_MAX_TURNS:], existing_summary


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
    """Append one turn to a session. Compresses old turns into summary when needed."""
    if not session_id:
        return

    col  = _get_collection()
    turn = {"role": role, "content": content[:4000]}  # cap content size

    if col is not None:
        try:
            expires = datetime.now(timezone.utc) + timedelta(hours=_TTL_HOURS)

            # Fetch current state so we can decide whether to compress
            doc = col.find_one({"session_id": session_id}, {"turns": 1, "summary": 1}) or {}
            current_turns   = doc.get("turns", [])
            current_summary = doc.get("summary", "")

            current_turns.append(turn)
            new_turns, new_summary = _maybe_compress(session_id, current_turns, current_summary)

            col.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "turns":      new_turns,
                        "summary":    new_summary,
                        "expires_at": expires,
                    }
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

            new_turns, new_summary = _maybe_compress(
                session_id,
                _fallback[session_id],
                _fallback_summaries.get(session_id, ""),
            )
            _fallback[session_id] = new_turns
            if new_summary:
                _fallback_summaries[session_id] = new_summary


def get_context(session_id: str, max_turns: int = 5) -> list[dict]:
    """
    Returns context for session_id as a list of role/content dicts.

    If a rolling summary exists it is prepended as a system turn so shadow
    models receive full history without raw token blowup:
        [{"role": "system", "content": "[Context summary]: ..."},
         <last max_turns raw turns>]

    Returns empty list if session not found or store unavailable.
    """
    if not session_id:
        return []

    col = _get_collection()

    if col is not None:
        try:
            doc = col.find_one({"session_id": session_id}, {"turns": 1, "summary": 1})
            if doc:
                raw     = doc.get("turns", [])[-max_turns:]
                summary = doc.get("summary", "")
                return _build_context(raw, summary)
        except Exception as exc:
            logger.debug("SessionStore.get_context MongoDB error: %s", exc)
    else:
        with _fallback_lock:
            raw     = _fallback.get(session_id, [])[-max_turns:]
            summary = _fallback_summaries.get(session_id, "")
            return _build_context(raw, summary)

    return []


def _build_context(raw_turns: list[dict], summary: str) -> list[dict]:
    """Prepend summary as system turn if present."""
    if not summary:
        return raw_turns
    system_turn = {"role": "system", "content": f"[Context summary]: {summary}"}
    return [system_turn] + raw_turns


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
            _fallback_summaries.pop(session_id, None)
