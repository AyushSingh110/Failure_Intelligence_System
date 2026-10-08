"""
Tenant-scoped data access — the only path a route uses to reach stored tenant data.

    TenantStore(scope)        every read filters on the scope's tenant,
                              every write is stamped with it

A store cannot be built without a TenantScope, and a scope cannot be built from
request data, so a handler has no way to address another tenant's rows. If the
scope is missing the constructor raises; nothing falls back to an unscoped query.

Functions whose name ends in `_all_tenants` read across tenants. They are for
platform-admin routes only (tests/security/test_static_boundary.py checks that
each caller holds the admin policy or went through authorize_cross_tenant_read).
There is no cross-tenant write or delete.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.schemas import InferenceRequest
from app.tenancy import require_tenant_id
from storage import database, signal_logger

logger = logging.getLogger(__name__)

InferenceIdConflict = database.InferenceIdConflict


class TenantStore:
    def __init__(self, scope: Any) -> None:
        self._tenant_id = require_tenant_id(scope)   # raises TenantScopeError

    # ── Inferences ────────────────────────────────────────────────────────────

    def save_inference(self, record: InferenceRequest) -> bool:
        """Store a record under this tenant, whatever tenant the record itself named."""
        owned = record.model_copy(update={"tenant_id": self._tenant_id})
        return database.save_inference(owned)

    def list_inferences(self, limit: int = 200, offset: int = 0) -> list[InferenceRequest]:
        return database.get_inferences_for_tenant(self._tenant_id, limit=limit, offset=offset)

    def get_inference(self, request_id: str) -> Optional[InferenceRequest]:
        return database.get_inference_by_id_for_tenant(request_id, self._tenant_id)

    def delete_inference(self, request_id: str) -> bool:
        return database.delete_inference_for_tenant(request_id, self._tenant_id)

    def clear_inferences(self) -> int:
        return database.clear_inferences_for_tenant(self._tenant_id)

    # ── Feedback ──────────────────────────────────────────────────────────────

    def save_feedback(self, feedback_doc: dict) -> bool:
        return database.save_feedback({**feedback_doc, "tenant_id": self._tenant_id})

    # ── Signal logs ───────────────────────────────────────────────────────────

    def log_signal(self, **fields: Any) -> str:
        fields.pop("tenant_id", None)
        return signal_logger.log_signal(tenant_id=self._tenant_id, **fields)

    def attach_request_id(self, log_id: str, request_id: str) -> bool:
        return signal_logger.attach_request_id(log_id, request_id, self._tenant_id)

    def find_signal_log(self, request_id: str) -> Optional[dict]:
        return signal_logger.find_log_by_request_id(request_id, self._tenant_id)

    def label_signal(self, log_id: str, fie_was_correct: bool, correct_answer: str = "") -> bool:
        return signal_logger.update_signal_feedback(
            log_id, fie_was_correct, correct_answer, tenant_id=self._tenant_id,
        )


# ── Platform-admin reads across tenants ───────────────────────────────────────

def list_inferences_all_tenants(limit: int = 200, offset: int = 0) -> list[InferenceRequest]:
    return database.list_inferences_all_tenants(limit=limit, offset=offset)


def get_inference_all_tenants(request_id: str) -> Optional[InferenceRequest]:
    return database.get_inference_all_tenants(request_id)


def signal_logs_collection_all_tenants():
    return signal_logger.signal_logs_collection_all_tenants()


def recent_signal_logs_all_tenants(limit: int = 100) -> list[dict]:
    return signal_logger.get_recent_logs_all_tenants(limit=limit)


def calibration_stats_all_tenants() -> dict:
    return signal_logger.get_calibration_stats_all_tenants()


# ── Global by design: anonymous SDK telemetry (no tenant data) ────────────────

def insert_sdk_telemetry(doc: dict) -> bool:
    db = database.get_db()
    if db is None:
        return False
    db["sdk_telemetry"].insert_one(doc)
    return True


def sdk_telemetry_since_all_tenants(cutoff_iso: str) -> Optional[list[dict]]:
    """Anonymous pings received since `cutoff_iso`, or None when the database is unavailable."""
    db = database.get_db()
    if db is None:
        return None
    return list(db["sdk_telemetry"].find({"received_at": {"$gte": cutoff_iso}}, {"_id": 0}))
