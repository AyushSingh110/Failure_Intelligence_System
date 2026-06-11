from __future__ import annotations
import contextlib
import contextvars
import json
import logging
import time
from typing import Any, Generator

# ── Correlation-ID context variable ───────────────────────────────────────────
# ContextVar is async-safe and thread-safe.  Each request sets its own value;
# background threads see the default "-".
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "fie_request_id", default="-"
)


def get_request_id() -> str:
    """Return the correlation ID bound to the current execution context."""
    return _request_id_var.get()


def set_request_id(rid: str) -> contextvars.Token:
    """Bind *rid* to the current context; returns a token for restoration."""
    return _request_id_var.set(rid)


@contextlib.contextmanager
def bind_request_id(rid: str) -> Generator[None, None, None]:
    """Context manager — sets the request ID for the duration of the block."""
    token = _request_id_var.set(rid)
    try:
        yield
    finally:
        _request_id_var.reset(token)


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """
    Emits each log record as a single JSON line.

    Standard fields
    ---------------
    ts      ISO-8601 UTC timestamp
    level   DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger  Dotted module name (e.g. "engine.pipeline.langgraph_pipeline")
    rid     Request correlation ID from the current context
    msg     Formatted log message
    exc     Formatted traceback (only when exc_info is set)
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":  record.levelname,
            "logger": record.name,
            "rid":    _request_id_var.get(),
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


# ── Public setup ──────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """
    Replace the root logger's handlers with a single JSON stdout handler.
    Idempotent — safe to call multiple times (subsequent calls are no-ops).

    Parameters
    ----------
    level : str
        Logging level name ("DEBUG", "INFO", "WARNING", "ERROR").
        Resolved case-insensitively; defaults to INFO on unrecognised values.
    """
    root = logging.getLogger()

    # Idempotency guard
    if any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JSONFormatter)
        for h in root.handlers
    ):
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    handler.setLevel(numeric_level)

    # Replace all existing handlers (avoids duplicate output)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence overly chatty third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
