from __future__ import annotations

from typing import Iterable


_SENSITIVE_TOKENS = (
    "regex",
    "pattern",
    "prompt_guard",
    "faiss",
    "system prompt",
    "developer message",
    "hidden instructions",
    "internal rules",
    "chain of thought",
)


def sanitize_text_for_external(text: str) -> str:
    """Return a softer explanation string that avoids leaking detector internals."""
    safe = (text or "").strip()
    for token in _SENSITIVE_TOKENS:
        safe = safe.replace(token, "internal safety signal")
        safe = safe.replace(token.title(), "internal safety signal")
        safe = safe.replace(token.upper(), "internal safety signal")
    return safe


def filter_safe_evidence(items: Iterable[dict]) -> list[dict]:
    """Drop evidence that should not be shown in an external explanation."""
    safe_items: list[dict] = []
    for item in items:
        if item.get("safe_to_expose", False):
            safe_items.append(item)
    return safe_items

