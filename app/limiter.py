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
    def decorator(func):
        if available and limiter is not None:
            return limiter.limit(rate)(func)
        return func
    return decorator
