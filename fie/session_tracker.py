"""Multi-turn adversarial escalation tracker. Hash-only storage, 4 escalation rules, in-process only."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


_MAX_TURNS_PER_SESSION = 20
_SESSION_TTL_SECS      = 1800.0
_MAX_SESSIONS          = 10_000

@dataclass(slots=True)
class TurnRecord:
    prompt_hash : str
    attack_type : str | None
    confidence  : float
    is_attack   : bool
    ts          : float = field(default_factory=time.monotonic)


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

    def record(
        self,
        session_id  : str,
        prompt_hash : str,
        attack_type : str | None,
        confidence  : float,
        is_attack   : bool,
    ) -> SessionEscalation | None:
        """
        Record one turn and return an escalation signal if a rule fires,
        or None if the session looks normal.
        """
        turn = TurnRecord(
            prompt_hash = prompt_hash,
            attack_type = attack_type,
            confidence  = confidence,
            is_attack   = is_attack,
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




_tracker: SessionTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> SessionTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = SessionTracker()
    return _tracker
