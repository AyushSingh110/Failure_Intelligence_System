from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    """A verified answer found in the ground truth cache."""
    question_text:   str
    verified_answer: str
    confidence:      float
    source:          str
    verified_by:     str
    verified_at:     str
    use_count:       int


# ── Internal helpers ────────────────────────────────────────────────────────

def _get_collection():
    """Returns the ground_truth_cache MongoDB collection, or None on error."""
    try:
        from storage.database import _db, _fallback_mode
        if _fallback_mode or _db is None:
            return None
        col = _db["ground_truth_cache"]
        return col
    except Exception as exc:
        logger.debug("Could not get ground_truth_cache collection: %s", exc)
        return None


def _embed_question(question: str) -> Optional[list[float]]:
    """
    Encodes the question into a 384-dim vector for similarity matching.
    Returns None if encoder is unavailable.
    """
    try:
        from engine.encoder import get_encoder
        encoder = get_encoder()
        if not encoder.available:
            return None
        vec = encoder.encode(question)
        if vec is not None:
            return vec.tolist()
        return None
    except Exception as exc:
        logger.debug("Cache embedding failed: %s", exc)
        return None


def _question_id(question: str) -> str:
    """Deterministic ID for a question (SHA-256 of normalized text)."""
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _get_similarity_threshold() -> float:
    try:
        from config import get_settings
        return get_settings().ground_truth_similarity_threshold
    except Exception:
        return 0.92


#Public API 
def lookup_cache(question: str) -> Optional[CacheHit]:
    """
    Step 7 — Check the verified answer cache for a question.
    """
    if not question or len(question.strip()) < 5:
        return None

    col = _get_collection()
    if col is None:
        return None  # MongoDB not available — skip cache

    try:
        # try exact-ish match via SHA-256 (fastest path)
        exact_id  = _question_id(question)
        exact_doc = col.find_one({"_id": exact_id})
        if exact_doc:
            _increment_use_count(col, exact_id)
            return _doc_to_hit(exact_doc)

        # semantic similarity search across all cached questions
        query_vec = _embed_question(question)
        if query_vec is None:
            return None  

        threshold = _get_similarity_threshold()
        query_arr = np.array(query_vec, dtype=np.float32)

        # Load all cached entries (reasonable for cache sizes < 10,000)
        docs = list(col.find({}, {"question_vector": 1, "question_text": 1,
                                   "verified_answer": 1, "source": 1,
                                   "confidence": 1, "verified_by": 1,
                                   "verified_at": 1, "use_count": 1}))

        best_sim  = -1.0
        best_doc  = None

        for doc in docs:
            stored_vec = doc.get("question_vector")
            if not stored_vec:
                continue
            stored_arr = np.array(stored_vec, dtype=np.float32)
            # Cosine similarity (vectors are L2-normalized by sentence-transformer)
            sim = float(np.dot(query_arr, stored_arr))
            if sim > best_sim:
                best_sim = sim
                best_doc = doc

        if best_sim >= threshold and best_doc is not None:
            logger.info(
                "Ground truth cache HIT | similarity=%.4f | answer=%s",
                best_sim, best_doc.get("verified_answer", "")[:80],
            )
            _increment_use_count(col, best_doc["_id"])
            return _doc_to_hit(best_doc)

        logger.debug("Ground truth cache MISS | best_similarity=%.4f", best_sim)
        return None

    except Exception as exc:
        logger.warning("Cache lookup error: %s", exc)
        return None


def save_to_cache(
    question:        str,
    verified_answer: str,
    source:          str  = "user_feedback",
    confidence:      float = 1.0,
    verified_by:     str  = "user",
) -> bool:
    """
    Saves a verified answer to the ground truth cache.
    """
    col = _get_collection()
    if col is None:
        logger.debug("Cache unavailable — MongoDB not connected")
        return False

    try:
        question_vec = _embed_question(question)
        now          = datetime.now(timezone.utc).isoformat()
        doc_id       = _question_id(question)

        doc = {
            "_id":            doc_id,
            "question_text":  question.strip(),
            "question_vector": question_vec,  # may be None if encoder unavailable
            "verified_answer": verified_answer.strip(),
            "source":         source,
            "confidence":     confidence,
            "verified_by":    verified_by,
            "verified_at":    now,
            "use_count":      0,
            "last_used_at":   now,
        }

        col.update_one(
            {"_id": doc_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info(
            "Saved to GT cache | source=%s | answer=%s...",
            source, verified_answer[:60],
        )
        return True

    except Exception as exc:
        logger.error("Cache save error: %s", exc)
        return False


def _increment_use_count(col, doc_id: str) -> None:
    """Increments use_count and updates last_used_at for analytics."""
    try:
        col.update_one(
            {"_id": doc_id},
            {"$inc": {"use_count": 1},
             "$set": {"last_used_at": datetime.now(timezone.utc).isoformat()}},
        )
    except Exception:
        pass  # Non-critical — never crash on analytics update


def _doc_to_hit(doc: dict) -> CacheHit:
    return CacheHit(
        question_text   = doc.get("question_text", ""),
        verified_answer = doc.get("verified_answer", ""),
        confidence      = doc.get("confidence", 1.0),
        source          = doc.get("source", "cache"),
        verified_by     = doc.get("verified_by", "unknown"),
        verified_at     = doc.get("verified_at", ""),
        use_count       = doc.get("use_count", 0),
    )
