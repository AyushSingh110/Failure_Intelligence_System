from __future__ import annotations
import inspect
import logging
logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter: Limiter | None = Limiter(key_func=get_remote_address)
    available: bool = True
except ImportError:
    logger.warning("Suppressed exception in limiter", exc_info=True)
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
