"""
Centralized rate-limiter singleton for the FIE API server.

Importing from here (instead of app.main) breaks the circular-import
that existed when routes.py tried to import _limiter from app.main.

Usage
-----
In route files:

    from app.limiter import limiter, rate_limit

    @router.post("/endpoint")
    @rate_limit("60/minute")
    def my_endpoint(request: Request, ...):
        ...

In app/main.py:

    from app.limiter import limiter, available as _rate_limiting_available
    if _rate_limiting_available:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
"""
from __future__ import annotations

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter: Limiter | None = Limiter(key_func=get_remote_address)
    available: bool = True
except ImportError:
    limiter = None
    available = False


def rate_limit(rate: str):
    """
    Decorator factory — applies a slowapi rate limit when available, no-op otherwise.

    Example::

        @router.post("/monitor")
        @rate_limit("60/minute")
        def monitor(request: Request, ...): ...
    """
    def decorator(func):
        if available and limiter is not None:
            return limiter.limit(rate)(func)
        return func
    return decorator
