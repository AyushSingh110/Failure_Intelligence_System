"""
Multi-turn adversarial escalation tracker.

Responsibilities:
  1. Record each scan turn (hash-only — no raw prompts ever stored).
  2. Fire 4 escalation rules that detect session-level attack patterns.
  3. Compute a trajectory boost for the current turn based on session history
     (the crescendo / foot-in-the-door detection mechanism).
  4. Optional Redis backend for multi-instance deployments (Cloud Run).

The trajectory boost is applied AFTER weighted aggregation and BEFORE three-zone
routing in scan_prompt().  It can push a borderline prompt from UNCERTAIN into
CLEAR ATTACK when the session history shows a classic crescendo pattern.

Boost signals (cap: +0.20 total):
  - Confidence escalation: last 3 turns show rising scores          → +0.07
  - Prior UNCERTAIN hits: 2+ of last 5 turns were uncertain         → +0.05
  - Crescendo signature: early avg < 0.20, current > 0.40           → +0.10
  - Rapid probing: 4+ turns within 60 seconds                       → +0.06
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


_MAX_TURNS_PER_SESSION = 20
_SESSION_TTL_SECS      = 1800.0
_MAX_SESSIONS          = 10_000

_CRESCENDO_BOOST_CAP: float = 0.20   # maximum total boost per turn

@dataclass(slots=True)
class TurnRecord:
    prompt_hash  : str
    attack_type  : str | None
    confidence   : float          # pre-boost confidence (never inflated)
    is_attack    : bool
    is_uncertain : bool = False   # True when routed to UNCERTAIN zone
    ts           : float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class SessionEscalation:
    rule     : str    # e.g. "RAPID_FIRE", "ESCALATING_CONF", ...
    severity : str    # "LOW" | "MEDIUM" | "HIGH"
    context  : dict   # rule-specific metadata (no raw prompts)


@dataclass
class _Session:
    session_id : str
    turns      : Deque[TurnRecord] = field(default_factory=lambda: deque(maxlen=_MAX_TURNS_PER_SESSION))
    last_seen  : float             = field(default_factory=time.monotonic)

    def add(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.last_seen = time.monotonic()


# ── Escalation rule implementations ──────────────────────────────────────────

def _check_rapid_fire(turns: Deque[TurnRecord]) -> SessionEscalation | None:
    now      = time.monotonic()
    hits     = [t for t in turns if t.is_attack and (now - t.ts) <= 60.0]
    if len(hits) >= 5:
        return SessionEscalation(
            rule     = "RAPID_FIRE",
            severity = "HIGH",
            context  = {"hits_in_60s": len(hits)},
        )
    return None


def _check_escalating_conf(turns: Deque[TurnRecord]) -> SessionEscalation | None:
    attack_turns = [t for t in turns if t.is_attack]
    if len(attack_turns) < 3:
        return None
    last3 = attack_turns[-3:]
    if last3[0].confidence < last3[1].confidence < last3[2].confidence:
        return SessionEscalation(
            rule     = "ESCALATING_CONF",
            severity = "MEDIUM",
            context  = {
                "confidences": [round(t.confidence, 3) for t in last3],
            },
        )
    return None


def _check_jailbreak_pivot(turns: Deque[TurnRecord]) -> SessionEscalation | None:
    window = list(turns)[-10:]
    attack_types = [t.attack_type for t in window if t.is_attack and t.attack_type]
    if not attack_types:
        return None
    for at in set(attack_types):
        if attack_types.count(at) >= 3:
            return SessionEscalation(
                rule     = "JAILBREAK_PIVOT",
                severity = "HIGH",
                context  = {"repeated_attack_type": at, "count": attack_types.count(at)},
            )
    return None


def _check_multi_vector(turns: Deque[TurnRecord]) -> SessionEscalation | None:
    window = list(turns)[-10:]
    distinct = {t.attack_type for t in window if t.is_attack and t.attack_type}
    if len(distinct) >= 3:
        return SessionEscalation(
            rule     = "MULTI_VECTOR",
            severity = "MEDIUM",
            context  = {"distinct_attack_types": sorted(distinct)},
        )
    return None


_RULES = [
    _check_rapid_fire,
    _check_escalating_conf,
    _check_jailbreak_pivot,
    _check_multi_vector,
]


class SessionTracker:
    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock     = threading.Lock()

    def _evict_expired(self) -> None:
        now     = time.monotonic()
        expired = [sid for sid, s in self._sessions.items()
                   if (now - s.last_seen) > _SESSION_TTL_SECS]
        for sid in expired:
            del self._sessions[sid]

        # Hard cap: drop oldest sessions if still over limit
        if len(self._sessions) > _MAX_SESSIONS:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].last_seen)
            for sid, _ in oldest[:len(self._sessions) - _MAX_SESSIONS]:
                del self._sessions[sid]

    # ── Trajectory boost ─────────────────────────────────────────────────────

    def get_trajectory_boost(self, session_id: str, current_confidence: float) -> float:
        """
        Compute a confidence boost for the current prompt based on session history.

        Called BEFORE recording the current turn (history = prior turns only).
        Returns a value in [0.0, _CRESCENDO_BOOST_CAP].
        """
        with self._lock:
            if session_id not in self._sessions:
                return 0.0
            turns = list(self._sessions[session_id].turns)

        if not turns:
            return 0.0

        boost = 0.0
        now   = time.monotonic()

        # Signal 1: Confidence escalation — last 3 turns have rising scores
        if len(turns) >= 3:
            last3_conf = [t.confidence for t in turns[-3:]]
            if last3_conf[0] < last3_conf[1] < last3_conf[2]:
                boost += 0.07

        # Signal 2: Prior UNCERTAIN hits — 2+ of last 5 turns were uncertain
        recent5   = turns[-5:]
        uncertain = sum(1 for t in recent5 if t.is_uncertain)
        if uncertain >= 2:
            boost += 0.05

        # Signal 3: Classic crescendo / foot-in-the-door
        # Early turns averaged below 0.20, current confidence above 0.40
        if len(turns) >= 3:
            early_avg = sum(t.confidence for t in turns[:-2]) / max(len(turns) - 2, 1)
            if early_avg < 0.20 and current_confidence > 0.40:
                boost += 0.10

        # Signal 4: Rapid probing — 4+ turns within 60 seconds
        rapid = sum(1 for t in turns if (now - t.ts) <= 60.0)
        if rapid >= 4:
            boost += 0.06

        return min(boost, _CRESCENDO_BOOST_CAP)

    def record(
        self,
        session_id   : str,
        prompt_hash  : str,
        attack_type  : str | None,
        confidence   : float,      # pre-boost confidence
        is_attack    : bool,
        is_uncertain : bool = False,
    ) -> SessionEscalation | None:
        """
        Record one turn and return an escalation signal if a rule fires,
        or None if the session looks normal.
        """
        turn = TurnRecord(
            prompt_hash  = prompt_hash,
            attack_type  = attack_type,
            confidence   = confidence,
            is_attack    = is_attack,
            is_uncertain = is_uncertain,
        )

        with self._lock:
            self._evict_expired()

            if session_id not in self._sessions:
                self._sessions[session_id] = _Session(session_id=session_id)
            session = self._sessions[session_id]
            session.add(turn)
            turns = session.turns  # snapshot reference inside lock

        # Run rules outside lock (read-only on turns snapshot)
        for rule_fn in _RULES:
            escalation = rule_fn(turns)
            if escalation is not None:
                return escalation

        return None

    def get_session_summary(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._sessions:
                return {"session_id": session_id, "turns": 0, "attacks": 0}
            s = self._sessions[session_id]
            turns  = list(s.turns)

        return {
            "session_id"    : session_id,
            "turns"         : len(turns),
            "attacks"       : sum(1 for t in turns if t.is_attack),
            "attack_types"  : sorted({t.attack_type for t in turns if t.attack_type}),
            "last_seen_ago" : round(time.monotonic() - s.last_seen, 1),
        }

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)




# ── Redis-backed tracker for multi-instance deployments ──────────────────────

class RedisSessionTracker(SessionTracker):
    """
    Drop-in replacement for SessionTracker that persists session state in Redis.

    Required for Cloud Run (multiple instances don't share in-process memory).
    Falls back to in-memory if Redis is unavailable — never raises.

    Set REDIS_URL env var to enable:
        REDIS_URL=redis://localhost:6379/0
    """

    def __init__(self, redis_url: str) -> None:
        super().__init__()
        self._redis_url = redis_url
        self._redis     = None
        self._connect()

    def _connect(self) -> None:
        try:
            import redis as _redis
            self._redis = _redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
        except Exception:
            self._redis = None  # fall back to in-memory

    def _key(self, session_id: str) -> str:
        return f"fie:session:{session_id}"

    def _load_session(self, session_id: str) -> _Session:
        """Load session from Redis; return empty _Session if unavailable."""
        if self._redis is None:
            return self._sessions.get(session_id, _Session(session_id=session_id))
        try:
            import pickle
            raw = self._redis.get(self._key(session_id))
            if raw:
                return pickle.loads(raw)  # noqa: S301
        except Exception:
            pass
        return _Session(session_id=session_id)

    def _save_session(self, session: _Session) -> None:
        """Persist session to Redis with TTL; silently falls back on failure."""
        if self._redis is None:
            return
        try:
            import pickle
            self._redis.setex(
                self._key(session.session_id),
                int(_SESSION_TTL_SECS),
                pickle.dumps(session),
            )
        except Exception:
            pass

    def get_trajectory_boost(self, session_id: str, current_confidence: float) -> float:
        session = self._load_session(session_id)
        # Temporarily put in _sessions so parent method can read it
        with self._lock:
            self._sessions[session_id] = session
        boost = super().get_trajectory_boost(session_id, current_confidence)
        with self._lock:
            self._sessions.pop(session_id, None)
        return boost

    def record(
        self,
        session_id   : str,
        prompt_hash  : str,
        attack_type  : str | None,
        confidence   : float,
        is_attack    : bool,
        is_uncertain : bool = False,
    ) -> SessionEscalation | None:
        session = self._load_session(session_id)
        turn = TurnRecord(
            prompt_hash  = prompt_hash,
            attack_type  = attack_type,
            confidence   = confidence,
            is_attack    = is_attack,
            is_uncertain = is_uncertain,
        )
        session.add(turn)
        self._save_session(session)

        for rule_fn in _RULES:
            escalation = rule_fn(session.turns)
            if escalation is not None:
                return escalation
        return None


# ── Session ID auto-generation ────────────────────────────────────────────────

def make_session_id(api_key: str, user_agent: str = "") -> str:
    """
    Generate a stable session identifier from API key + User-Agent.

    Does not use IP address — VPN and mobile users rotate IPs.
    API key is unique per user; User-Agent differentiates multiple clients
    belonging to the same user.

    Hash is truncated to 16 hex chars (64-bit) — enough uniqueness for
    10,000 concurrent sessions with negligible collision probability.
    """
    raw = f"{api_key}:{user_agent}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Singleton factory ─────────────────────────────────────────────────────────

_tracker      : SessionTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> SessionTracker:
    """
    Return the process-wide session tracker singleton.

    Uses RedisSessionTracker if REDIS_URL is set, otherwise in-memory.
    Redis failures fall back to in-memory automatically.
    """
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                redis_url = os.environ.get("REDIS_URL", "")
                if redis_url:
                    _tracker = RedisSessionTracker(redis_url)
                    # If Redis connection failed, fall back to in-memory
                    if _tracker._redis is None:
                        _tracker = SessionTracker()
                else:
                    _tracker = SessionTracker()
    return _tracker
