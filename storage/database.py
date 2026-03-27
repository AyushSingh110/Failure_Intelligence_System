"""
storage/database.py — MongoDB Atlas Backend

Replaces the flat JSON vault with MongoDB Atlas.

Collections
-----------
  inferences   : one document per InferenceRequest
                 same schema as before — drop-in replacement

Public API — identical to the original flat-file version
---------------------------------------------------------
  initialize_vault()              → connects to MongoDB
  save_inference(data)            → inserts one document
  get_all_inferences()            → returns all as InferenceRequest list
  get_inference_by_id(request_id) → returns one or None
  delete_inference(request_id)    → deletes one, returns bool
  flush_vault()                   → no-op (MongoDB writes are immediate)

No changes needed anywhere else in the codebase.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import InferenceRequest
from config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level MongoDB client and collection ─────────────────────────────
_client     = None
_db         = None
_collection = None
_fallback_records: dict[str, InferenceRequest] = {}
_fallback_mode = False


# ── Internal helpers ───────────────────────────────────────────────────────

def _get_collection():
    """Returns the inferences collection, initializing if needed."""
    global _collection
    if _collection is None:
        initialize_vault()
    return _collection


def _to_doc(record: InferenceRequest) -> dict[str, Any]:
    """Converts InferenceRequest to a MongoDB document."""
    doc = record.model_dump()
    # Use request_id as the MongoDB _id for fast lookups
    doc["_id"] = doc["request_id"]
    return doc


def _from_doc(doc: dict[str, Any]) -> InferenceRequest | None:
    """Converts a MongoDB document back to InferenceRequest."""
    try:
        doc.pop("_id", None)   # remove MongoDB _id before passing to Pydantic
        return InferenceRequest(**doc)
    except Exception as exc:
        logger.warning("Failed to parse document: %s", exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────

def initialize_vault() -> None:
    """
    Connects to MongoDB Atlas and initializes the collection.
    Called once at application startup from main.py lifespan.
    """
    global _client, _db, _collection, _fallback_mode

    settings = get_settings()

    if not settings.mongodb_uri:
        logger.error(
            "MONGODB_URI is not set in .env file. "
            "Add MONGODB_URI=mongodb+srv://... to your .env"
        )
        _fallback_mode = True
        _client = None
        _db = None
        _collection = None
        print("[database] Falling back to in-memory storage because MongoDB is not configured.")
        return

    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        print(f"[database] Connecting to MongoDB Atlas...")

        _client = MongoClient(
            settings.mongodb_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=10000,  # 10 seconds
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            tls=True,
            tlsAllowInvalidCertificates=True,  # fixes Windows TLS issues
        )

        # Ping to confirm connection works
        _client.admin.command("ping")
        print("[database] Connected to MongoDB Atlas successfully.")

        _db         = _client[settings.mongodb_db_name]
        _collection = _db["inferences"]

        # Create index on request_id for fast lookups
        _collection.create_index("request_id", unique=True, background=True)
        # Create index on timestamp for chronological queries
        _collection.create_index("timestamp", background=True)
        # Create index on model_name for filtering by model
        _collection.create_index("model_name", background=True)

        count = _collection.count_documents({})
        _fallback_mode = False
        print(f"[database] Collection ready — {count} existing records.")

    except ImportError:
        print("[database] ERROR: pymongo not installed. Run: pip install pymongo")
        _fallback_mode = True
        _client = None
        _db = None
        _collection = None
        print("[database] Falling back to in-memory storage because pymongo is unavailable.")
    except Exception as exc:
        print(f"[database] ERROR connecting to MongoDB: {exc}")
        _fallback_mode = True
        _client = None
        _db = None
        _collection = None
        print("[database] Falling back to in-memory storage because MongoDB is unavailable.")


def flush_vault() -> None:
    """
    No-op for MongoDB — writes are immediate and persistent.
    Kept for API compatibility with the original flat-file version.
    """
    pass


def save_inference(data: InferenceRequest) -> bool:
    """
    Inserts one inference record into MongoDB.
    Returns True on success, False on failure.
    Uses upsert so duplicate request_ids don't crash.
    """
    try:
        if _fallback_mode:
            _fallback_records[data.request_id] = data
            return True
        col = _get_collection()
        if col is None:
            _fallback_records[data.request_id] = data
            return True
        doc = _to_doc(data)
        col.update_one(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True,
        )
        return True
    except Exception as exc:
        logger.error("Failed to save inference %s: %s", data.request_id, exc)
        return False


def get_all_inferences() -> list[InferenceRequest]:
    """
    Returns all stored inference records sorted by timestamp descending.
    Most recent first — matches the original vault behaviour.
    """
    try:
        if _fallback_mode:
            return sorted(
                _fallback_records.values(),
                key=lambda record: record.timestamp,
                reverse=True,
            )
        col  = _get_collection()
        if col is None:
            return sorted(
                _fallback_records.values(),
                key=lambda record: record.timestamp,
                reverse=True,
            )
        docs = col.find({}, sort=[("timestamp", -1)])
        records = []
        for doc in docs:
            record = _from_doc(doc)
            if record:
                records.append(record)
        return records
    except Exception as exc:
        logger.error("Failed to fetch inferences: %s", exc)
        return []


def get_inferences_for_tenant(tenant_id: str) -> list[InferenceRequest]:
    """Returns only the inference records belonging to a single tenant."""
    try:
        if _fallback_mode:
            return sorted(
                (record for record in _fallback_records.values() if record.tenant_id == tenant_id),
                key=lambda record: record.timestamp,
                reverse=True,
            )
        col = _get_collection()
        if col is None:
            return sorted(
                (record for record in _fallback_records.values() if record.tenant_id == tenant_id),
                key=lambda record: record.timestamp,
                reverse=True,
            )
        docs = col.find({"tenant_id": tenant_id}, sort=[("timestamp", -1)])
        records = []
        for doc in docs:
            record = _from_doc(doc)
            if record:
                records.append(record)
        return records
    except Exception as exc:
        logger.error("Failed to fetch inferences for tenant %s: %s", tenant_id, exc)
        return []


def get_inference_by_id(request_id: str) -> InferenceRequest | None:
    """Returns a single inference record by request_id, or None."""
    try:
        if _fallback_mode:
            return _fallback_records.get(request_id)
        col = _get_collection()
        if col is None:
            return _fallback_records.get(request_id)
        doc = col.find_one({"request_id": request_id})
        if doc is None:
            return None
        return _from_doc(doc)
    except Exception as exc:
        logger.error("Failed to fetch inference %s: %s", request_id, exc)
        return None


def get_inference_by_id_for_tenant(request_id: str, tenant_id: str) -> InferenceRequest | None:
    """Returns a single inference record if it belongs to the given tenant."""
    try:
        if _fallback_mode:
            record = _fallback_records.get(request_id)
            if record and record.tenant_id == tenant_id:
                return record
            return None
        col = _get_collection()
        if col is None:
            record = _fallback_records.get(request_id)
            if record and record.tenant_id == tenant_id:
                return record
            return None
        doc = col.find_one({"request_id": request_id, "tenant_id": tenant_id})
        if doc is None:
            return None
        return _from_doc(doc)
    except Exception as exc:
        logger.error(
            "Failed to fetch inference %s for tenant %s: %s",
            request_id,
            tenant_id,
            exc,
        )
        return None


def delete_inference(request_id: str) -> bool:
    """Deletes a single inference record. Returns True if deleted."""
    try:
        if _fallback_mode:
            return _fallback_records.pop(request_id, None) is not None
        col    = _get_collection()
        if col is None:
            return _fallback_records.pop(request_id, None) is not None
        result = col.delete_one({"request_id": request_id})
        return result.deleted_count > 0
    except Exception as exc:
        logger.error("Failed to delete inference %s: %s", request_id, exc)
        return False


def delete_inference_for_tenant(request_id: str, tenant_id: str) -> bool:
    """Deletes one inference record if it belongs to the given tenant."""
    try:
        if _fallback_mode:
            record = _fallback_records.get(request_id)
            if record and record.tenant_id == tenant_id:
                _fallback_records.pop(request_id, None)
                return True
            return False
        col = _get_collection()
        if col is None:
            record = _fallback_records.get(request_id)
            if record and record.tenant_id == tenant_id:
                _fallback_records.pop(request_id, None)
                return True
            return False
        result = col.delete_one({"request_id": request_id, "tenant_id": tenant_id})
        return result.deleted_count > 0
    except Exception as exc:
        logger.error(
            "Failed to delete inference %s for tenant %s: %s",
            request_id,
            tenant_id,
            exc,
        )
        return False


def clear_inferences_for_tenant(tenant_id: str) -> int:
    """Deletes all inference records for a single tenant and returns the number removed."""
    try:
        if _fallback_mode:
            to_delete = [key for key, record in _fallback_records.items() if record.tenant_id == tenant_id]
            for key in to_delete:
                _fallback_records.pop(key, None)
            return len(to_delete)
        col = _get_collection()
        if col is None:
            to_delete = [key for key, record in _fallback_records.items() if record.tenant_id == tenant_id]
            for key in to_delete:
                _fallback_records.pop(key, None)
            return len(to_delete)
        result = col.delete_many({"tenant_id": tenant_id})
        return int(result.deleted_count)
    except Exception as exc:
        logger.error("Failed to clear inferences for tenant %s: %s", tenant_id, exc)
        return 0
