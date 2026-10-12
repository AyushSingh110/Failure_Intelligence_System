"""
Security events: one JSON object per line on the `fie.security` logger.

Only what is needed to verify the tenant boundary. An event never carries a
credential, a prompt, an answer, a correction or an e-mail address, and a tenant
appears only as a short one-way reference (the raw tenant id contains part of
the owner's e-mail).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("fie.security")

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

# Extra fields an event may carry. A closed list: anything else is dropped, so a
# caller cannot put user text or a secret into the log by passing it here.
_ALLOWED_EXTRA = frozenset({
    "label", "event_id", "block_enabled", "scan_threshold",
    "old_version", "new_version", "n_labeled", "store", "resource",
})


def tenant_ref(tenant_id: Optional[str]) -> str:
    """Short one-way reference to a tenant, safe to log."""
    if not tenant_id:
        return "none"
    return hashlib.sha256(str(tenant_id).encode("utf-8")).hexdigest()[:12]


def emit(
    event:            str,
    *,
    severity:         str = "info",
    outcome:          str,
    reason:           str = "",
    principal:        Any = None,
    request:          Any = None,
    target_tenant_id: Optional[str] = None,
    **extra:          Any,
) -> None:
    """Record one security event. Never raises: audit logging must not break a request."""
    try:
        record: dict[str, Any] = {
            "event":           event,
            "severity":        severity if severity in _LEVELS else "info",
            "ts":              datetime.now(timezone.utc).isoformat(),
            "outcome":         outcome,
            "reason":          reason,
            "tenant_ref":      tenant_ref(getattr(principal, "tenant_id", None)),
            "credential_kind": getattr(principal, "credential_kind", None) or "none",
        }
        if target_tenant_id is not None:
            record["target_tenant_ref"] = tenant_ref(target_tenant_id)
        if request is not None:
            state = getattr(request, "state", None)
            record["rid"] = getattr(state, "request_id", None)
            scope = getattr(request, "scope", None) or {}
            route = scope.get("route") if isinstance(scope, dict) else None
            record["route"]  = getattr(route, "path", None)
            record["method"] = getattr(request, "method", None)
        for key, value in extra.items():
            if key in _ALLOWED_EXTRA and isinstance(value, (str, int, float, bool, type(None))):
                record[key] = value
        logger.log(_LEVELS.get(severity, logging.INFO), json.dumps(record, sort_keys=True))
    except Exception:   # pragma: no cover - defensive
        logger.error('{"event": "security_event_failed", "severity": "error"}')
