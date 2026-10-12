from __future__ import annotations

import logging
from typing import Any

from app.schemas import InferenceRequest
from config import get_settings

logger = logging.getLogger(__name__)

#Module-level MongoDB client and collection
_client     = None
_db         = None
_collection = None
# In-memory store used when MongoDB is unavailable. Keyed by "<tenant>:<request_id>",
# so two tenants can never address the same entry.
_fallback_records: dict[str, InferenceRequest] = {}
_fallback_mode = False

_RESERVED_TENANTS = frozenset({"", "anonymous"})


class InferenceIdConflict(Exception):
    """
    The request_id is already held by a different tenant and the database still
    has the pre-WP-002 unique index on `request_id` alone. Nothing was written.
    """


#Internal helpers

def _get_collection():
    """Returns the inferences collection, initializing if needed."""
    global _collection
    if _collection is None:
        initialize_vault()
    return _collection


def _doc_id(tenant_id: str, request_id: str) -> str:
    """Document id namespaced by tenant: one tenant cannot name another tenant's document."""
    return f"{tenant_id}:{request_id}"


def _valid_tenant(tenant_id: Any) -> bool:
    return isinstance(tenant_id, str) and tenant_id.strip() not in _RESERVED_TENANTS


def _from_doc(doc: dict[str, Any]) -> InferenceRequest | None:
    """Converts a MongoDB document back to InferenceRequest."""
    try:
        doc.pop("_id", None)   # remove MongoDB _id before passing to Pydantic
        return InferenceRequest(**doc)
    except Exception as exc:
        logger.warning("Failed to parse document: %s", exc)
        return None


def _fallback_sorted(tenant_id: str | None) -> list[InferenceRequest]:
    records = (
        r for r in _fallback_records.values()
        if tenant_id is None or r.tenant_id == tenant_id
    )
    return sorted(records, key=lambda record: record.timestamp, reverse=True)


#Public API

def request_indexes(col) -> None:
    """
    Ask for the indexes the tenant-scoped queries use. Each request is independent
    and a failure is logged, not raised: a missing index is slow, not unsafe, and
    must not push the whole store into in-memory fallback.

    The unique index on `request_id` alone is deliberately no longer requested:
    it made request ids one global namespace. Its replacement is the unique
    index on (tenant_id, request_id). Dropping the old index on an existing
    database is an owner-run migration step.
    """
    wanted = (
        ([("tenant_id", 1), ("request_id", 1)], {"unique": True}),
        ([("tenant_id", 1), ("timestamp", -1)], {}),
        ("timestamp", {}),
        ("model_name", {}),
    )
    for keys, options in wanted:
        try:
            col.create_index(keys, background=True, **options)
        except Exception as exc:
            logger.warning("inferences index request failed (%s): %s", keys, type(exc).__name__)


def initialize_vault() -> None:
    """
    Connects to MongoDB Atlas and initializes the collection.
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

        print("[database] Connecting to MongoDB Atlas...")

        _client = MongoClient(
            settings.mongodb_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=10000,
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

        request_indexes(_collection)

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


def get_db():
    """
    Public accessor for the MongoDB database handle. Returns None if unavailable.

    Deliberately a PURE accessor — it never connects. Connection is the
    application's startup job (`initialize_vault()` from the FastAPI lifespan).

    An earlier version of this function lazily called `initialize_vault()` when
    `_db` was None. That looked convenient and was a trap: any incidental caller
    on a request path — here, the model-extraction tracker — would trigger a
    blocking MongoDB SRV lookup with a multi-second DNS timeout, inside a
    request, from a code path whose failure is supposed to be non-fatal. In a
    test run with no reachable cluster it hung the whole process.

    Prefer this over importing the module-level `_db` global: several modules
    reach for the private name directly, which couples them to this module's
    internals and breaks silently if it is renamed. That is exactly what
    happened to engine/model_extraction_tracker.py, which imported a `get_db`
    that did not exist — the ImportError was swallowed by a broad handler and
    the extraction tracker degraded to a permanent no-op in production.
    """
    return _db


def flush_vault() -> None:
    """
    No-op for MongoDB — writes are immediate and persistent.
    """
    pass


def save_inference(data: InferenceRequest) -> bool:
    """
    Stores one inference record under the tenant named on the record.

    The record's `tenant_id` is the scope of the write. Routes never set it from
    a request: storage.tenant_store.TenantStore overwrites it with the
    authenticated tenant before calling this. A record with no real tenant is
    refused — there is no shared or anonymous bucket.

    The write can only create or update a document of that same tenant. Raises
    InferenceIdConflict when the id belongs to another tenant and the old
    single-field unique index is still in place.
    """
    tenant_id = data.tenant_id
    if not _valid_tenant(tenant_id):
        logger.error("Refused to save an inference without a tenant")
        return False
    doc_id = _doc_id(tenant_id, data.request_id)
    try:
        if _fallback_mode:
            _fallback_records[doc_id] = data
            return True
        col = _get_collection()
        if col is None:
            _fallback_records[doc_id] = data
            return True
        doc = data.model_dump()
        col.update_one(
            {"tenant_id": tenant_id, "request_id": data.request_id},
            {"$set": doc, "$setOnInsert": {"_id": doc_id}},
            upsert=True,
        )
        return True
    except Exception as exc:
        if type(exc).__name__ == "DuplicateKeyError":
            raise InferenceIdConflict(data.request_id) from None
        logger.error("Failed to save inference %s: %s", data.request_id, type(exc).__name__)
        return False


def list_inferences_all_tenants(limit: int = 200, offset: int = 0) -> list[InferenceRequest]:
    """
    Stored inference records of EVERY tenant, newest first.
    Platform-admin use only; callers must go through authorize_cross_tenant_read.
    """
    try:
        if _fallback_mode:
            return _fallback_sorted(None)[offset: offset + limit]
        col  = _get_collection()
        if col is None:
            return _fallback_sorted(None)[offset: offset + limit]
        docs = col.find({}, sort=[("timestamp", -1)]).skip(offset).limit(limit)
        records = []
        for doc in docs:
            record = _from_doc(doc)
            if record:
                records.append(record)
        return records
    except Exception as exc:
        logger.error("Failed to fetch inferences: %s", type(exc).__name__)
        return []


def get_inferences_for_tenant(tenant_id: str, limit: int = 200, offset: int = 0) -> list[InferenceRequest]:
    """Returns inference records for a single tenant, newest first."""
    if not _valid_tenant(tenant_id):
        return []
    try:
        if _fallback_mode:
            return _fallback_sorted(tenant_id)[offset: offset + limit]
        col = _get_collection()
        if col is None:
            return _fallback_sorted(tenant_id)[offset: offset + limit]
        docs = col.find({"tenant_id": tenant_id}, sort=[("timestamp", -1)]).skip(offset).limit(limit)
        records = []
        for doc in docs:
            record = _from_doc(doc)
            if record:
                records.append(record)
        return records
    except Exception as exc:
        logger.error("Failed to fetch inferences for a tenant: %s", type(exc).__name__)
        return []


def get_inference_all_tenants(request_id: str) -> InferenceRequest | None:
    """
    One inference record by request_id, whichever tenant holds it.
    Platform-admin use only; callers must go through authorize_cross_tenant_read.
    """
    try:
        if _fallback_mode:
            return next((r for r in _fallback_records.values() if r.request_id == request_id), None)
        col = _get_collection()
        if col is None:
            return next((r for r in _fallback_records.values() if r.request_id == request_id), None)
        doc = col.find_one({"request_id": request_id})
        if doc is None:
            return None
        return _from_doc(doc)
    except Exception as exc:
        logger.error("Failed to fetch inference %s: %s", request_id, type(exc).__name__)
        return None


def get_inference_by_id_for_tenant(request_id: str, tenant_id: str) -> InferenceRequest | None:
    """Returns a single inference record if it belongs to the given tenant."""
    if not _valid_tenant(tenant_id):
        return None
    try:
        if _fallback_mode:
            return _fallback_records.get(_doc_id(tenant_id, request_id))
        col = _get_collection()
        if col is None:
            return _fallback_records.get(_doc_id(tenant_id, request_id))
        doc = col.find_one({"tenant_id": tenant_id, "request_id": request_id})
        if doc is None:
            return None
        return _from_doc(doc)
    except Exception as exc:
        logger.error("Failed to fetch inference %s for a tenant: %s", request_id, type(exc).__name__)
        return None


def delete_inference_for_tenant(request_id: str, tenant_id: str) -> bool:
    """Deletes one inference record if it belongs to the given tenant."""
    if not _valid_tenant(tenant_id):
        return False
    try:
        if _fallback_mode:
            return _fallback_records.pop(_doc_id(tenant_id, request_id), None) is not None
        col = _get_collection()
        if col is None:
            return _fallback_records.pop(_doc_id(tenant_id, request_id), None) is not None
        result = col.delete_one({"tenant_id": tenant_id, "request_id": request_id})
        return result.deleted_count > 0
    except Exception as exc:
        logger.error("Failed to delete inference %s for a tenant: %s", request_id, type(exc).__name__)
        return False


# User feedback storage

def save_feedback(feedback_doc: dict) -> bool:
    """
    Saves a user feedback record to the 'feedback' collection.
    The document must name its tenant; storage.tenant_store sets it.
    """
    if not _valid_tenant(feedback_doc.get("tenant_id")):
        logger.error("Refused to save feedback without a tenant")
        return False
    try:
        if _fallback_mode or _db is None:
            logger.warning("Feedback not saved — MongoDB unavailable")
            return False
        col = _db["feedback"]
        col.create_index([("tenant_id", 1), ("request_id", 1)], background=True)
        col.insert_one(feedback_doc)
        return True
    except Exception as exc:
        logger.error("Failed to save feedback: %s", type(exc).__name__)
        return False


def get_feedback_for_request(request_id: str, tenant_id: str) -> list[dict]:
    """Returns the tenant's feedback records for a given request_id."""
    if not _valid_tenant(tenant_id):
        return []
    try:
        if _fallback_mode or _db is None:
            return []
        col  = _db["feedback"]
        docs = list(col.find({"tenant_id": tenant_id, "request_id": request_id}, {"_id": 0}))
        return docs
    except Exception as exc:
        logger.error("Failed to fetch feedback for %s: %s", request_id, type(exc).__name__)
        return []


def clear_inferences_for_tenant(tenant_id: str) -> int:
    """Deletes all inference records for a single tenant and returns the number removed."""
    if not _valid_tenant(tenant_id):
        return 0
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
        logger.error("Failed to clear inferences for a tenant: %s", type(exc).__name__)
        return 0
