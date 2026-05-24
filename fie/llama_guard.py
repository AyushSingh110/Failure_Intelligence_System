"""LlamaGuard 3 Tier-3 tiebreaker — circuit breaker + LRU cache, Groq free tier."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal


_GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
_GROQ_MODEL       = "meta-llama/llama-guard-3-8b"
_GROQ_ENDPOINT    = "https://api.groq.com/openai/v1/chat/completions"
_REQUEST_TIMEOUT  = 10.0
_CB_FAILURE_LIMIT = 3
_CB_RECOVERY_SECS = 60.0
_CACHE_MAXSIZE    = 256
_CACHE_TTL        = 600.0

class _LGCache:
    def __init__(self, maxsize: int = _CACHE_MAXSIZE, ttl: float = _CACHE_TTL) -> None:
        self._maxsize = maxsize
        self._ttl     = ttl
        self._store: OrderedDict[str, tuple[bool, float]] = OrderedDict()
        self._lock  = threading.Lock()

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def get(self, prompt: str) -> bool | None:
        k = self._key(prompt)
        with self._lock:
            if k not in self._store:
                return None
            verdict, ts = self._store[k]
            if time.monotonic() - ts > self._ttl:
                del self._store[k]
                return None
            self._store.move_to_end(k)
            return verdict

    def set(self, prompt: str, verdict: bool) -> None:
        k = self._key(prompt)
        with self._lock:
            self._store[k] = (verdict, time.monotonic())
            self._store.move_to_end(k)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)


_lg_cache = _LGCache()


_CBState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

@dataclass
class _CircuitBreaker:
    _failures:    int       = field(default=0, init=False)
    _state:       _CBState  = field(default="CLOSED", init=False)
    _opened_at:   float     = field(default=0.0, init=False)
    _lock:        threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self) -> bool:
        with self._lock:
            if self._state == "CLOSED":
                return True
            if self._state == "OPEN":
                if time.monotonic() - self._opened_at >= _CB_RECOVERY_SECS:
                    self._state = "HALF_OPEN"
                    return True
                return False
            return True  # HALF_OPEN — allow one probe

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state    = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= _CB_FAILURE_LIMIT:
                self._state    = "OPEN"
                self._opened_at = time.monotonic()

    @property
    def state(self) -> _CBState:
        return self._state


_circuit = _CircuitBreaker()


def query_llama_guard(prompt: str) -> bool:
    """Returns True if LlamaGuard 3 classifies the prompt as unsafe. Raises on circuit open or API error."""
    if not _GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — LlamaGuard unavailable")

    # Cache hit
    cached = _lg_cache.get(prompt)
    if cached is not None:
        return cached

    # Circuit breaker
    if not _circuit.allow():
        raise RuntimeError(f"LlamaGuard circuit breaker OPEN (state={_circuit.state})")

    # Build LlamaGuard conversation format
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    try:
        import requests as _req
        resp = _req.post(
            _GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model"      : _GROQ_MODEL,
                "messages"   : messages,
                "max_tokens" : 20,
                "temperature": 0.0,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as exc:
        _circuit.record_failure()
        raise RuntimeError(f"LlamaGuard call failed: {exc}") from exc

    # LlamaGuard 3 returns "safe" or "unsafe\n<category>"
    is_unsafe = content.startswith("unsafe")
    verdict   = is_unsafe  # True = attack, False = safe

    _circuit.record_success()
    _lg_cache.set(prompt, verdict)
    return verdict


def reset_circuit() -> None:
    """Force circuit breaker back to CLOSED. Useful in tests."""
    with _circuit._lock:
        _circuit._failures  = 0
        _circuit._state     = "CLOSED"
        _circuit._opened_at = 0.0
