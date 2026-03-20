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
    global _client, _db, _collection

    settings = get_settings()

    if not settings.mongodb_uri:
        logger.error(
            "MONGODB_URI is not set in .env file. "
            "Add MONGODB_URI=mongodb+srv://... to your .env"
        )
        raise RuntimeError("MONGODB_URI not configured. Check your .env file.")

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
        print(f"[database] Collection ready — {count} existing records.")

    except ImportError:
        print("[database] ERROR: pymongo not installed. Run: pip install pymongo")
        raise
    except Exception as exc:
        print(f"[database] ERROR connecting to MongoDB: {exc}")
        raise


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
        col = _get_collection()
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
        col  = _get_collection()
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


def get_inference_by_id(request_id: str) -> InferenceRequest | None:
    """Returns a single inference record by request_id, or None."""
    try:
        col = _get_collection()
        doc = col.find_one({"request_id": request_id})
        if doc is None:
            return None
        return _from_doc(doc)
    except Exception as exc:
        logger.error("Failed to fetch inference %s: %s", request_id, exc)
        return None


def delete_inference(request_id: str) -> bool:
    """Deletes a single inference record. Returns True if deleted."""
    try:
        col    = _get_collection()
        result = col.delete_one({"request_id": request_id})
        return result.deleted_count > 0
    except Exception as exc:
        logger.error("Failed to delete inference %s: %s", request_id, exc)
        return False