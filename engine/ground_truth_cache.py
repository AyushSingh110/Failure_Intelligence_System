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


CACHE_SCHEMA = 2   # entries written before tenant scoping have no `schema` and are never served


def _question_id(question: str, tenant_id: str) -> str:
    """
    Deterministic id for one tenant's question: SHA-256 over the schema tag, the
    tenant and the normalized text. Two tenants asking the same question get two
    different ids, and the raw tenant id is not recoverable from the key.
    """
    normalized = question.strip().lower()
    material = f"gtc{CACHE_SCHEMA}\x00{tenant_id}\x00{normalized}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _scope_tenant(scope, operation: str) -> Optional[str]:
    """
    The tenant of `scope`, or None. A cache call without a scope is served
    uncached — never from a shared entry — and is reported, because it means a
    code path lost the tenant.
    """
    from app.tenancy import tenant_of

    tenant_id = tenant_of(scope)
    if tenant_id is None:
        try:
            from app.security_events import emit
            emit("cache.scope_missing", severity="error", outcome="denied",
                 reason=operation, store="ground_truth_cache")
        except Exception as exc:   # pragma: no cover - defensive
            logger.error("cache.scope_missing could not be recorded: %s", type(exc).__name__)
    return tenant_id


def _get_similarity_threshold() -> float:
    try:
        from config import get_settings
        return get_settings().ground_truth_similarity_threshold
    except Exception as exc:
        logger.warning(
            "degraded capability=gt_cache_config impact='default similarity threshold used' "
            "reason=%s: %s", type(exc).__name__, exc,
        )
        return 0.92


#Public API
def lookup_cache(question: str, *, scope=None) -> Optional[CacheHit]:
    """
    Step 7 — Check the tenant's verified-answer cache for a question.

    `scope` is the caller's TenantScope. Only entries written under the same
    tenant are considered, for the exact match and for the similarity search.
    Without a scope the result is a miss.
    """
    tenant_id = _scope_tenant(scope, "lookup")
    if tenant_id is None:
        return None

    if not question or len(question.strip()) < 5:
        return None

    col = _get_collection()
    if col is None:
        return None  # MongoDB not available — skip cache

    try:
        owned = {"tenant_id": tenant_id, "schema": CACHE_SCHEMA}

        # try exact-ish match via SHA-256 (fastest path)
        exact_id  = _question_id(question, tenant_id)
        exact_doc = col.find_one({"_id": exact_id, **owned})
        if exact_doc:
            _increment_use_count(col, exact_id, tenant_id)
            return _doc_to_hit(exact_doc)

        # semantic similarity search across this tenant's cached questions
        query_vec = _embed_question(question)
        if query_vec is None:
            return None

        threshold = _get_similarity_threshold()
        query_arr = np.array(query_vec, dtype=np.float32)

        docs = list(col.find(owned, {"question_vector": 1, "question_text": 1,
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
            logger.info("Ground truth cache HIT | similarity=%.4f", best_sim)
            _increment_use_count(col, best_doc["_id"], tenant_id)
            return _doc_to_hit(best_doc)

        logger.debug("Ground truth cache MISS | best_similarity=%.4f", best_sim)
        return None

    except Exception as exc:
        logger.warning("Cache lookup error: %s", type(exc).__name__)
        return None


def save_to_cache(
    question:        str,
    verified_answer: str,
    source:          str  = "user_feedback",
    confidence:      float = 1.0,
    verified_by:     str  = "tenant",
    *,
    scope=None,
    source_class:    str  = "system",
) -> bool:
    """
    Saves a verified answer to the tenant's ground truth cache.

    scope        : the caller's TenantScope. Without one nothing is written.
    source_class : "tenant_feedback" for a correction the tenant submitted,
                   "system" for an answer the pipeline verified itself. A system
                   write never replaces the tenant's own correction.
    verified_by  : kept for call compatibility and ignored. The entry records a
                   class of verifier ("tenant" / "system"), never a person.
    """
    tenant_id = _scope_tenant(scope, "save")
    if tenant_id is None:
        return False

    col = _get_collection()
    if col is None:
        logger.debug("Cache unavailable — MongoDB not connected")
        return False

    try:
        doc_id = _question_id(question, tenant_id)
        owned  = {"_id": doc_id, "tenant_id": tenant_id}
        from_tenant = source_class == "tenant_feedback"

        if not from_tenant:
            existing = col.find_one(owned, {"source_class": 1})
            if existing and existing.get("source_class") == "tenant_feedback":
                return False

        question_vec = _embed_question(question)
        now          = datetime.now(timezone.utc).isoformat()

        doc = {
            "tenant_id":      tenant_id,
            "schema":         CACHE_SCHEMA,
            "source_class":   "tenant_feedback" if from_tenant else "system",
            "question_text":  question.strip(),
            "question_vector": question_vec,  # may be None if encoder unavailable
            "verified_answer": verified_answer.strip(),
            "source":         source,
            "confidence":     confidence,
            "verified_by":    "tenant" if from_tenant else "system",
            "verified_at":    now,
            "use_count":      0,
            "last_used_at":   now,
        }

        col.update_one(owned, {"$set": doc}, upsert=True)
        try:
            col.create_index("tenant_id", background=True)
        except Exception as exc:
            logger.warning("ground_truth_cache index request failed: %s", type(exc).__name__)
        logger.info("Saved to GT cache | source=%s", source)
        return True

    except Exception as exc:
        logger.error("Cache save error: %s", type(exc).__name__)
        return False


def _increment_use_count(col, doc_id: str, tenant_id: str) -> None:
    """Increments use_count and updates last_used_at for analytics."""
    try:
        col.update_one(
            {"_id": doc_id, "tenant_id": tenant_id},
            {"$inc": {"use_count": 1},
             "$set": {"last_used_at": datetime.now(timezone.utc).isoformat()}},
        )
    except Exception as exc:
        logger.warning(
            "degraded capability=gt_cache_stats impact='cache hit not counted; eviction ordering may be less accurate' "
            "reason=%s: %s", type(exc).__name__, exc,
        )


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
