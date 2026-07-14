"""
Phase 0 — Pipeline instrumentation (measurement only, no behavior change).

A lightweight, opt-in metrics collector for the hallucination pipeline. It records
per-request:
  - wall-clock time per LangGraph node
  - Groq API round-trips (real calls vs cache hits) + token usage
  - external HTTP round-trips (Wikidata / Serper / Wikipedia)

Design notes
------------
* INACTIVE BY DEFAULT. Nothing is recorded unless a caller wraps a request in
  `begin()` / `end()`. Production code paths (the /monitor route) never call
  those, so behavior and output are unchanged when the harness is not driving.

* Single-request scope. The collector is a process-global guarded by a lock so
  that Groq's ThreadPoolExecutor fan-out (which does NOT inherit contextvars)
  still records into it. This is correct for the measurement harness, which runs
  requests strictly sequentially. It is deliberately NOT safe for concurrent
  in-flight requests — do not activate it in a live multi-request server.

* Nodes stay pure. Timing wraps each node from the outside (see
  engine/pipeline/langgraph_pipeline.py `_timed`); node functions are untouched.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroqCall:
    model:          str
    input_tokens:   int
    output_tokens:  int
    latency_ms:     float
    cached:         bool


@dataclass
class HttpCall:
    host:       str
    latency_ms: float


@dataclass
class RequestMetrics:
    """All measurements for a single pipeline invocation."""
    node_ms:     dict[str, float] = field(default_factory=dict)
    groq_calls:  list[GroqCall]   = field(default_factory=list)
    http_calls:  list[HttpCall]   = field(default_factory=list)
    total_ms:    float            = 0.0
    rate_limit_hits: int          = 0   # Groq 429s that exhausted retries this request

    # ── Derived aggregates (computed on demand) ──────────────────────────────
    @property
    def n_groq_api(self) -> int:
        return sum(1 for c in self.groq_calls if not c.cached)

    @property
    def n_groq_cache_hits(self) -> int:
        return sum(1 for c in self.groq_calls if c.cached)

    @property
    def n_http(self) -> int:
        return len(self.http_calls)

    @property
    def input_tokens(self) -> int:
        # Cache hits cost 0 tokens on THIS request — the honest per-request cost.
        return sum(c.input_tokens for c in self.groq_calls if not c.cached)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.groq_calls if not c.cached)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "total_ms":          round(self.total_ms, 2),
            "node_ms":           {k: round(v, 2) for k, v in self.node_ms.items()},
            "n_groq_api_calls":  self.n_groq_api,
            "n_groq_cache_hits": self.n_groq_cache_hits,
            "n_http_calls":      self.n_http,
            "input_tokens":      self.input_tokens,
            "output_tokens":     self.output_tokens,
            "total_tokens":      self.total_tokens,
            "rate_limit_hits":   self.rate_limit_hits,
            "http_hosts":        sorted({c.host for c in self.http_calls}),
        }


# ── Process-global collector ──────────────────────────────────────────────────
_lock:      threading.Lock            = threading.Lock()
_collector: Optional[RequestMetrics]  = None


def is_active() -> bool:
    return _collector is not None


def begin() -> None:
    """Start recording. Resets any prior state."""
    global _collector
    with _lock:
        _collector = RequestMetrics()


def end() -> Optional[RequestMetrics]:
    """Stop recording and return the collected metrics (or None if inactive)."""
    global _collector
    with _lock:
        m = _collector
        _collector = None
    return m


def record_groq_call(
    model:         str,
    input_tokens:  int,
    output_tokens: int,
    latency_ms:    float,
    cached:        bool = False,
) -> None:
    if _collector is None:
        return
    with _lock:
        if _collector is not None:
            _collector.groq_calls.append(
                GroqCall(model, int(input_tokens or 0), int(output_tokens or 0),
                         float(latency_ms or 0.0), bool(cached))
            )


def record_rate_limit() -> None:
    if _collector is None:
        return
    with _lock:
        if _collector is not None:
            _collector.rate_limit_hits += 1


def record_http_call(host: str, latency_ms: float) -> None:
    if _collector is None:
        return
    with _lock:
        if _collector is not None:
            _collector.http_calls.append(HttpCall(str(host), float(latency_ms or 0.0)))


def record_node(name: str, ms: float) -> None:
    if _collector is None:
        return
    with _lock:
        if _collector is not None:
            # Accumulate in case a node runs more than once in a graph.
            _collector.node_ms[name] = _collector.node_ms.get(name, 0.0) + float(ms)


@contextmanager
def node_timer(name: str):
    """Time a node body and record it (no-op when inactive)."""
    if _collector is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        record_node(name, (time.perf_counter() - start) * 1000.0)
