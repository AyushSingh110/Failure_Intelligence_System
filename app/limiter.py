from __future__ import annotations
import inspect
import logging
logger = logging.getLogger(__name__)

def _remote_address(request) -> str:
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "127.0.0.1"


def rate_key(request) -> str:
    """
    Whose budget a request spends.

    An authenticated request is counted against its tenant, so two tenants behind
    one address (a proxy, a NAT) have independent limits and one cannot exhaust
    the other's. The tenant is taken from the principal the auth dependency put
    on the request — never from a header, which a caller could set. A request
    with no principal is counted against its socket address, as before.
    Forwarded-address headers are not trusted: the proxy in front is unknown.
    """
    principal = getattr(getattr(request, "state", None), "principal", None)
    tenant_id = getattr(principal, "tenant_id", None)
    if isinstance(tenant_id, str) and tenant_id:
        from app.security_events import tenant_ref
        return f"tenant:{tenant_ref(tenant_id)}"
    return _remote_address(request)


try:
    from slowapi import Limiter

    limiter: Limiter | None = Limiter(key_func=rate_key)
    available: bool = True
except ImportError:
    # slowapi is optional, but its absence is a real exposure rather than a
    # cosmetic degradation: every endpoint then serves UNLIMITED requests per
    # IP. Logged at WARNING with the consequence spelled out, because a public
    # deployment that silently lost rate limiting looks identical to a healthy
    # one until it is being abused.
    logger.warning(
        "degraded capability=rate_limiting impact='NO per-IP request limits — "
        "all endpoints are unthrottled' action='pip install slowapi'"
    )
    limiter = None
    available = False


def rate_limit(rate: str):
    def decorator(func):
        if available and limiter is not None:
            wrapped = limiter.limit(rate)(func)
            # slowapi's wrapper drops the original signature, so FastAPI misreads
            # Pydantic body params as required query params (every request → 422).
            # Restore the signature with annotations resolved (routes use
            # `from __future__ import annotations`, so they'd otherwise be strings
            # FastAPI can't evaluate in slowapi's module globals).
            try:
                wrapped.__signature__ = inspect.signature(func, eval_str=True)
            except (TypeError, ValueError):
                pass
            return wrapped
        return func
    return decorator
