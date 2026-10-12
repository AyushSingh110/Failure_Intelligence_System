"""
Tenant scope: the value every tenant-sensitive data access needs.

A TenantScope can be built only from a Principal, which only authentication
builds. A string taken from a request cannot become one. Functions that hold
tenant state take a scope as an explicit argument; a function that receives
none uses no tenant state at all. There is no "current tenant" in a context
variable or a thread-local: this server runs handlers and shadow-model calls in
thread pools, and an implicit tenant is exactly what a pool loses.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional


class TenantScopeError(Exception):
    """Raised when tenant-sensitive state is reached without a valid scope."""


class TenantScope:
    """One tenant, derived from an authenticated Principal. Immutable."""

    __slots__ = ("_tenant_id",)

    def __init__(self, principal: Any) -> None:
        from app.auth_guard import Principal

        if not isinstance(principal, Principal):
            raise TenantScopeError("a tenant scope can only be built from an authenticated principal")
        object.__setattr__(self, "_tenant_id", principal.tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("TenantScope is immutable")

    def __repr__(self) -> str:   # never print the raw tenant id into a log by accident
        return f"TenantScope(ref={hashlib.sha256(self._tenant_id.encode()).hexdigest()[:12]})"


def tenant_of(scope: Any) -> Optional[str]:
    """The tenant of a scope, or None for anything that is not a TenantScope."""
    return scope.tenant_id if isinstance(scope, TenantScope) else None


def require_tenant_id(scope: Any) -> str:
    tenant_id = tenant_of(scope)
    if not tenant_id:
        raise TenantScopeError("tenant scope required")
    return tenant_id


def scope_key(scope: Any, *parts: str) -> str:
    """A cache key bound to one tenant. Raises without a scope; never falls back to a global key."""
    digest = hashlib.sha256(require_tenant_id(scope).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8", errors="replace"))
    return digest.hexdigest()


# ── Per-tenant clusters and trend ─────────────────────────────────────────────

class TenantAnalytics:
    """One tenant's failure clusters and trend. All access goes through one lock."""

    def __init__(self) -> None:
        from engine.archetypes.clustering import ArchetypeClusterRegistry
        from engine.evolution.tracker import SignalEvolutionTracker

        self._lock     = threading.RLock()
        self._registry = ArchetypeClusterRegistry()
        self._tracker  = SignalEvolutionTracker()

    def assign(self, signal: Any) -> Any:
        with self._lock:
            return self._registry.assign(signal)

    def record(self, signal: Any) -> None:
        with self._lock:
            self._tracker.record(signal)

    def trend_summary(self) -> dict:
        with self._lock:
            return self._tracker.trend_summary()

    def summarize(self) -> list[dict]:
        with self._lock:
            return self._registry.summarize()

    def reset_clusters(self) -> None:
        from engine.archetypes.clustering import ArchetypeClusterRegistry

        with self._lock:
            self._registry = ArchetypeClusterRegistry()


class TenantRegistry:
    """
    tenant → TenantAnalytics, bounded. When full, the tenant used longest ago is
    dropped, so creating many tenants cannot grow memory without limit.
    """

    def __init__(self, max_tenants: int = 512) -> None:
        self._max   = max(1, int(max_tenants))
        self._items: OrderedDict[str, TenantAnalytics] = OrderedDict()
        self._lock  = threading.Lock()

    def for_scope(self, scope: Any) -> TenantAnalytics:
        tenant_id = require_tenant_id(scope)
        with self._lock:
            holder = self._items.get(tenant_id)
            if holder is None:
                holder = TenantAnalytics()
                self._items[tenant_id] = holder
                while len(self._items) > self._max:
                    self._items.popitem(last=False)
            else:
                self._items.move_to_end(tenant_id)
            return holder

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


tenant_analytics = TenantRegistry()
