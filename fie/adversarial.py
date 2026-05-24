from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import math
import re
import statistics
import threading
import time
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Legacy global threshold (kept for backward compat) ────────────────────────
# Per-attack-type thresholds (_ATTACK_THRESHOLDS) are used in scan_prompt().
# This value is only used when attack type is unknown or as a final fallback.
import os as _os
SCAN_THRESHOLD: float = float(_os.environ.get("SCAN_THRESHOLD", "0.65"))


# ── Per-attack-type thresholds ────────────────────────────────────────────────
# Calibrated per-layer precision from JailbreakBench v2 evaluation.
# Hot-configurable via MongoDB fie_config (get_attack_thresholds()).
# Uncertainty zone = [threshold × 0.60, threshold) → routes to LlamaGuard.
_ATTACK_THRESHOLDS: dict[str, float] = {
    "TOKEN_SMUGGLING"              : 0.88,  # regex only, near-zero FPR
    "PROMPT_INJECTION"             : 0.72,  # high precision needed
    "GCG_ADVERSARIAL_SUFFIX"       : 0.72,  # statistical, needs high bar
    "INDIRECT_PROMPT_INJECTION"    : 0.70,
    "MANY_SHOT_JAILBREAK"          : 0.68,
    "OBFUSCATED_ADVERSARIAL_PAYLOAD": 0.70, # just recalibrated
    "JAILBREAK_ATTEMPT"            : 0.65,  # PAIR classifier backs this up
}

# Layers with near-zero FPR — fire above threshold → BLOCK, skip aggregation.
_FAST_PATH_LAYERS: frozenset[str] = frozenset({"regex", "gcg_suffix"})

# Per-layer weights for weighted vote aggregator (precision-calibrated).
_LAYER_WEIGHTS: dict[str, float] = {
    "regex"              : 1.5,
    "gcg_suffix"         : 1.3,
    "many_shot"          : 1.2,
    "prompt_guard"       : 1.1,
    "pair_classifier"    : 1.0,
    "indirect_injection" : 0.9,
    "perplexity_proxy"   : 0.7,   # lowest precision layer
}


# ── LayerResult dataclass ─────────────────────────────────────────────────────
@dataclass
class LayerResult:
    """Normalised output from one detection layer."""
    layer_name  : str
    attack_type : str | None
    confidence  : float
    evidence    : dict
    latency_ms  : float = 0.0


# ── Threshold helpers ─────────────────────────────────────────────────────────

def _get_attack_threshold(attack_type: str) -> float:
    """Per-type threshold: fie_config (hot) > _ATTACK_THRESHOLDS > SCAN_THRESHOLD."""
    try:
        from engine.fie_config import get_attack_thresholds
        return get_attack_thresholds().get(attack_type, _ATTACK_THRESHOLDS.get(attack_type, SCAN_THRESHOLD))
    except Exception:
        return _ATTACK_THRESHOLDS.get(attack_type, SCAN_THRESHOLD)


def _get_scan_threshold(override: float | None) -> float:
    """Legacy helper — kept for any external callers. Use _get_attack_threshold() internally."""
    if override is not None:
        return override
    try:
        from engine.fie_config import get_scan_threshold
        return get_scan_threshold()
    except Exception:
        return SCAN_THRESHOLD


# ── Scan result cache ─────────────────────────────────────────────────────────
# TTL-aware LRU cache for scan_prompt() results.
# Key: SHA-256(prompt.strip().lower()) — raw text never stored.
# Used to short-circuit repeated identical prompts (common in load tests / retries)
# and to avoid redundant LlamaGuard API calls.

class _ScanCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 512, ttl: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl     = ttl
        self._cache: collections.OrderedDict[str, tuple[object, float]] = collections.OrderedDict()
        self._lock    = threading.RLock()

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode("utf-8", errors="replace")).hexdigest()

    def get(self, prompt: str) -> object | None:
        key = self._key(prompt)
        with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, prompt: str, value: object) -> None:
        key = self._key(prompt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, prompt: str) -> None:
        with self._lock:
            self._cache.pop(self._key(prompt), None)


_scan_cache = _ScanCache(maxsize=512, ttl=300.0)


# Result dataclass
@dataclass
class ScanResult:
    """Result returned by scan_prompt()."""
    is_attack:      bool
    attack_type:    str | None          # e.g. "PROMPT_INJECTION", "JAILBREAK_ATTEMPT"
    category:       str | None          # e.g. "INJECTION", "JAILBREAK", "SMUGGLING"
    confidence:     float               # 0.0 – 1.0
    layers_fired:   list[str]           # which layers detected something
    matched_text:   str | None          # excerpt that triggered detection
    mitigation:     str                 # human-readable mitigation advice
    evidence:       dict = field(default_factory=dict)  # per-layer detail


# Layer 1: attack pattern definitions
@dataclass
class _AttackPattern:
    category:        str
    root_cause:      str
    base_confidence: float
    pattern:         re.Pattern


_ATTACK_PATTERNS: list[_AttackPattern] = [

    _AttackPattern(
        category="INJECTION",
        root_cause="PROMPT_INJECTION",
        base_confidence=0.88,
        pattern=re.compile(
            r"\b("
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?|"
            r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?|"
            r"forget\s+(?:everything|all\s+(?:previous|prior))\s+(?:you\s+were\s+told|instructions?)|"
            r"reveal\s+(?:the\s+)?(?:system\s+prompt|your\s+(?:instructions?|guidelines?|rules?))|"
            r"(?:print|show|output|display|repeat|tell\s+me)\s+(?:your\s+)?system\s+(?:prompt|instructions?)|"
            r"(?:print|show|output|display|reveal|tell\s+me)\s+(?:the\s+)?(?:hidden|secret|internal|developer)\s+(?:message|messages|prompt|instructions?|rules?)|"
            r"what\s+are\s+your\s+(?:hidden|secret|internal|original)\s+instructions?|"
            r"(?:internal|hidden|secret)\s+(?:instructions?|rules?)|"
            r"new\s+(?:primary\s+)?(?:instruction|directive|rule|task)\s*[:;]?\s+(?:you\s+must|ignore)|"
            r"(?:your\s+)?(?:new\s+)?(?:primary\s+)?directive\s+is\s+to\s+ignore|"
            r"SYSTEM\s*[:;]\s*override|"
            r"(?:from|starting)\s+now\s+(?:on\s+)?ignore\s+all|"
            r"(?:note|message|instructions?|reminder|alert)\s+(?:to|for)\s+(?:(?:the\s+)?(?:AI|LLM|language\s+model|assistant|model|GPT|Claude|bot))\s*[:\-]|"
            r"reveal\s+your\s+system\s+prompt"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    _AttackPattern(
        category="JAILBREAK",
        root_cause="JAILBREAK_ATTEMPT",
        base_confidence=0.82,
        pattern=re.compile(
            r"\b("
            r"you\s+are\s+now\s+(?:DAN|an?\s+(?:AI|assistant|model)\s+with(?:out)?\s+(?:no\s+)?restrictions?)|"
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|free)\s+(?:AI|model|assistant)|"
            r"act\s+as\s+(?:an?\s+)?(?:AI|model|assistant)\s+with(?:out)?\s+(?:no\s+|any\s+)?restrictions?|"
            r"(?:pretend|imagine|roleplay\s+as)\s+(?:you\s+have\s+)?(?:no\s+|without\s+)(?:content\s+)?(?:policy|restrictions?|guidelines?|limits?)|"
            r"you\s+(?:have\s+been\s+)?(?:freed|liberated|released)\s+from\s+(?:all\s+)?(?:restrictions?|guidelines?)|"
            r"(?:developer|jailbreak|debug|unrestricted|god)\s+mode\s+(?:activated?|enabled?|on|where\s+all\s+restrictions?\s+(?:are|have\s+been)\s+(?:lifted|removed|disabled))|"
            r"(?:your\s+)?(?:ethical\s+)?(?:restrictions?|guidelines?|limitations?|filters?)\s+"
            r"(?:have\s+been\s+)?(?:removed|disabled|lifted|suspended|deactivated)|"
            r"(?:act|respond|behave|operate)\s+as\s+(?:if|though)\s+(?:you\s+(?:were|had|have)\s+)?(?:no\s+|never\s+(?:had\s+)?)"
            r"(?:safety|ethical|content)\s+(?:training|guidelines?|restrictions?)|"
            r"you\s+are\s+now\s+playing\s+(?:the\s+role\s+of\s+)?an?\s+(?:AI|assistant)\s+(?:with\s+)?no\s+restrictions?|"
            r"answer\s+only\s+as\s+dan|"
            r"never\s+refuse\s+any\s+request|"
            r"no\s+ethical\s+limits?|"
            r"answer\s+anything"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # Soft roleplay / fiction / persona framing variants
    _AttackPattern(
        category="JAILBREAK",
        root_cause="JAILBREAK_ATTEMPT",
        base_confidence=0.76,
        pattern=re.compile(
            r"\b("
            r"(?:roleplay|role-play)\s+as\s+(?:\w+\s+){0,6}without\s+(?:any\s+)?(?:restrictions?|guidelines?|limits?|safety\s+training|content\s+policy)|"
            r"(?:imagine|pretend|suppose)\s+you\s+(?:were|had\s+been|have\s+been)\s+(?:freed|liberated|released|untethered|unchained)\s+from\s+(?:all\s+)?(?:your\s+)?(?:restrictions?|guidelines?|ethical|safety)|"
            r"(?:imagine|pretend|suppose)\s+(?:you.?re|you\s+(?:are|were))\s+(?:\w+\s+){0,6}(?:no\s+restrictions?|without\s+(?:any\s+)?(?:restrictions?|content\s+policy|ethical\s+(?:limits?|guidelines?|constraints?)))|"
            r"(?:acting|playing)\s+as\s+(?:an?\s+)?(?:AI|model|assistant|bot|language\s+model)\b[^.!?\n]{0,50}without\s+(?:any\s+)?(?:restrictions?|guidelines?|limits?)|"
            r"\bjailbreak\s+mode\b|"
            r"(?:act|you\s+will\s+(?:act|respond)|playing)\s+as\s+(?:an?\s+)?(?:\w+\s+){0,6}that\s+(?:never\s+(?:refuses?|declines?|says?\s+no)|has\s+no\s+restrictions?)|"
            r"version\s+of\s+(?:yourself|you)\s+without\s+(?:any\s+)?(?:ethical\s+)?(?:guidelines?|restrictions?|limits?|filters?)|"
            r"(?:I\s+want\s+you\s+to|you\s+(?:must|will|should))\s+(?:act|pretend|roleplay|be|become)\s+as\s+(?:an?\s+)?(?:\w+\s+){0,6}without\s+(?:any\s+)?(?:restrictions?|limits?|guidelines?)"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    _AttackPattern(
        category="OVERRIDE",
        root_cause="INSTRUCTION_OVERRIDE",
        base_confidence=0.78,
        pattern=re.compile(
            r"\b("
            r"forget\s+all\s+(?:previous\s+)?instructions?\s+and\s+(?:obey|follow|comply\s+with)|"
            r"(?:bypass|circumvent|override|disable)\s+(?:your\s+)?(?:safety|content|ethical)\s+(?:filter|restriction|guideline|check)|"
            r"(?:bypass|circumvent|override|disable)\s+(?:all\s+)?(?:safeguards?|protections?)|"
            r"(?:your\s+)?(?:ethical\s+)?guidelines?\s+are\s+now\s+(?:suspended|disabled|removed|lifted)|"
            r"output\s+your\s+chain\s+of\s+thought|"
            r"chain\s+of\s+thought\s+step\s+by\s+step|"
            r"obey\s+only\s+me|"
            r"(?:as\s+(?:your\s+)?(?:admin|administrator|supervisor|creator|developer|owner)|"
            r"I\s+(?:am|have)\s+(?:authorized?|authorised?|granted)\s+(?:you\s+)?(?:special\s+)?(?:permission|access))\s+"
            r".{0,50}(?:bypass|ignore|disable|override)|"
            r"for\s+(?:this|the\s+(?:purposes?\s+of\s+(?:this|the)))\s+(?:request|task|session|purpose)\s+"
            r"(?:your\s+)?(?:normal\s+)?(?:rules?|restrictions?|guidelines?)\s+do\s+not\s+apply|"
            r"this\s+is\s+(?:an?\s+)?(?:authorized?|official|emergency|special)\s+(?:test|request|override)\s*[,;.]?\s*"
            r"(?:all\s+)?(?:safety|content)?\s+(?:rules?|restrictions?|filters?)\s+(?:are\s+)?(?:suspended|disabled)"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    _AttackPattern(
        category="SMUGGLING",
        root_cause="TOKEN_SMUGGLING",
        base_confidence=0.91,
        pattern=re.compile(
            r"("
            r"<\|(?:system|im_start|im_end|endoftext|startoftext)\|>|"
            r"\[/?INST\]|\[/?SYS\]|\[/?SYSTEM\]|"
            r"###\s*(?:Human|System|Assistant)\s*:|"
            r"</?s>.*?(?:unrestricted|no\s+restrictions?|ignore)|"
            r"\{\{(?:system_prompt|prompt|instructions?)\}\}|"
            r"<!--.*?(?:ignore|override|bypass).*?-->|"
            r"END\s+OF\s+SYSTEM\s+PROMPT|"
            r"BEGIN\s+NEW\s+SYSTEM\s+PROMPT|"
            r"\[\[SYSTEM\s+(?:MESSAGE|OVERRIDE|COMMAND)\]\]|"
            r"\\x00|\\u0000|%00"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]


# Layer 1 helpers: obfuscation normalization

_SPACED_SEGMENT_VOCAB: frozenset[str] = frozenset({
    "ignore", "disregard", "forget", "bypass", "override", "reveal",
    "circumvent", "jailbreak", "hack", "steal", "leak", "expose",
    "obey", "comply", "follow", "output", "print", "show", "repeat",
    "all", "previous", "prior", "above", "earlier", "any", "new",
    "instructions", "guidelines", "rules", "restrictions", "directives",
    "filters", "policies", "safeguards",
    "system", "prompt", "safety", "policy", "directive", "rule",
    "everything", "your", "my", "the", "and", "now", "from", "with", "only",
    "you", "me", "how", "to", "tell", "what",
})

_HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "х": "x",
    "α": "a", "ο": "o",
    "@": "a", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s",
})


def _collapse_spaced_run(m: re.Match) -> str:
    letters = m.group(0).split()
    words: list[str] = []
    buf = ""
    for ch in letters:
        buf += ch
        if buf.lower() in _SPACED_SEGMENT_VOCAB:
            words.append(buf)
            buf = ""
        elif len(buf) > 15:
            words.append(buf)
            buf = ""
    if buf:
        words.append(buf)
    return " ".join(words)


def _normalize_for_detection(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPH_MAP)
    text = re.sub(r"\b(?:[a-zA-Z] ){2,}[a-zA-Z]\b", _collapse_spaced_run, text)
    text = re.sub(r"[​‌‍⁠﻿­]", "", text)
    return text


def _run_pattern_detection(prompt: str) -> tuple[_AttackPattern | None, str]:
    priority_order = ["SMUGGLING", "INJECTION", "JAILBREAK", "OVERRIDE"]
    normalized = _normalize_for_detection(prompt)
    hits: dict[str, tuple[_AttackPattern, str, bool]] = {}

    for ap in _ATTACK_PATTERNS:
        m = ap.pattern.search(prompt)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], False)
            continue
        m = ap.pattern.search(normalized)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], True)

    for cat in priority_order:
        if cat in hits:
            ap, matched_text, obfuscated = hits[cat]
            if obfuscated:
                ap = _AttackPattern(
                    category=ap.category,
                    root_cause=ap.root_cause,
                    base_confidence=max(ap.base_confidence - 0.06, 0.50),
                    pattern=ap.pattern,
                )
            return ap, matched_text
    return None, ""


# Layer 2: PromptGuard semantic scorer

_LEETSPEAK_TRANSLATION = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
})

_GROUP_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    "override": (
        re.compile(r"\bignore\b"),
        re.compile(r"\bdisregard\b"),
        re.compile(r"\bforget\b"),
        re.compile(r"\boverride\b"),
        re.compile(r"\bbypass\b"),
        re.compile(r"\bcircumvent\b"),
        re.compile(r"\bdisable\b"),
        re.compile(r"\bremove\b"),
    ),
    "policy_target": (
        re.compile(r"\binstructions?\b"),
        re.compile(r"\bguidelines?\b"),
        re.compile(r"\brules?\b"),
        re.compile(r"\bsystem\s*prompt\b"),
        re.compile(r"\bdeveloper\s*(?:message|prompt|instructions?)\b"),
        re.compile(r"\bhidden\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsecret\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\binternal\s*(?:message|prompt|instructions?|rules?)\b"),
        re.compile(r"\bsafeguards?\b"),
        re.compile(r"\bprotections?\b"),
        re.compile(r"\bfilters?\b"),
    ),
    "exfiltration": (
        # Narrowed to technique-scoped phrases only — generic "show", "print",
        # "tell me", "hidden", "secret" were firing on millions of benign prompts.
        # These patterns only fire when the target is clearly system/prompt-related,
        # and Layer 2 already requires policy_target as a co-condition.
        re.compile(r"\breveal\s+(?:the\s+)?(?:system|hidden|secret|internal|your)\b"),
        re.compile(r"\brepeat\s+(?:the\s+)?(?:system|hidden|secret|above|previous)\b"),
        re.compile(r"\bexfiltrate?\b"),
        re.compile(r"\bleak\s+(?:the\s+)?(?:system|prompt|instructions?|data)\b"),
        re.compile(r"\bchain\s*of\s*thought\b"),
        re.compile(r"\bprint\s+(?:the\s+)?(?:system|hidden|secret|internal)\b"),
        re.compile(r"\boutput\s+(?:the\s+)?(?:system|hidden|secret|internal|your\s+(?:instructions?|guidelines?))\b"),
        re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:system\s+prompt|hidden\s+(?:message|instructions?)|secret\s+(?:message|instructions?))\b"),
    ),
    "persona": (
        re.compile(r"\bdan\b"),
        re.compile(r"\bdo\s*anything\s*now\b"),
        re.compile(r"\bunrestricted\b"),
        re.compile(r"\buncensored\b"),
        re.compile(r"\bunfiltered\b"),
        re.compile(r"\bdeveloper\s*mode\b"),
        re.compile(r"\bjailbreak\s*mode\b"),
        re.compile(r"\bno\s*restrictions?\b"),
        re.compile(r"\bno\s*ethical\s*limits?\b"),
        re.compile(r"\banswer\s*anything\b"),
    ),
    "authority_claim": (
        re.compile(r"\bauthorized?\s*(?:test|request|override)\b"),
        re.compile(r"\bofficial\s*(?:test|request|override)\b"),
        re.compile(r"\bemergency\s*override\b"),
        re.compile(r"\bspecial\s*permission\b"),
        re.compile(r"\bobey\s*only\s*me\b"),
        re.compile(r"\badmin\b"),
        re.compile(r"\badministrator\b"),
        re.compile(r"\bdeveloper\b"),
        re.compile(r"\bowner\b"),
        re.compile(r"\bsupervisor\b"),
        re.compile(r"\bcreator\b"),
    ),
}


def _run_guard_detection(prompt: str) -> tuple[str | None, float, list[str]]:
    def _score(text: str) -> tuple[float, str | None, list[str]]:
        lowered  = text.lower().translate(_LEETSPEAK_TRANSLATION)
        spaced   = re.sub(r"[\W_]+", " ", lowered)
        spaced   = re.sub(r"\s+", " ", spaced).strip()
        squashed = spaced.replace(" ", "")

        group_hits: dict[str, list[str]] = {}
        for group, patterns in _GROUP_PATTERNS.items():
            for pattern in patterns:
                m = pattern.search(spaced) or pattern.search(squashed)
                if m:
                    group_hits.setdefault(group, []).append(m.group(0)[:80])

        groups    = set(group_hits)
        evidence  = [f"{g}:{', '.join(v[:2])}" for g, v in sorted(group_hits.items())]
        score     = 0.0
        root_cause = None

        if {"override", "policy_target"} <= groups:
            score = max(score, 0.82); root_cause = "PROMPT_INJECTION"
        if {"exfiltration", "policy_target"} <= groups:
            score = max(score, 0.86); root_cause = "PROMPT_INJECTION"
        if "persona" in groups and ("override" in groups or "authority_claim" in groups):
            score = max(score, 0.84); root_cause = "JAILBREAK_ATTEMPT"
        if {"authority_claim", "policy_target"} <= groups:
            score = max(score, 0.78); root_cause = root_cause or "INSTRUCTION_OVERRIDE"
        if "persona" in groups:
            score = max(score, 0.68); root_cause = root_cause or "JAILBREAK_ATTEMPT"

        return round(score, 4), root_cause, evidence

    score, root_cause, evidence = _score(prompt)
    if root_cause is None or score < 0.75:
        normalized = _normalize_for_detection(prompt)
        if normalized != prompt:
            score, root_cause, evidence = _score(normalized)
    if root_cause is None or score < 0.75:
        return None, 0.0, []
    return root_cause, score, evidence


# Layer 3: Many-shot jailbreak detector

_MANY_SHOT_EXCHANGE_RE = re.compile(
    r"(?:human|user|h|q|person)\s*:\s*.{3,300}?\n+\s*(?:assistant|bot|ai|a|gpt|claude|llm)\s*:",
    re.IGNORECASE | re.DOTALL,
)

_MANY_SHOT_Q_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:human|user|h|q|person)\s*[\:\>]\s*.{5,300}",
    re.IGNORECASE,
)

_MANY_SHOT_HARMFUL_RE = re.compile(
    r"\b(?:bomb|explos|weapon|hack|malware|ransomware|poison|synthesize|"
    r"drug|meth|fentanyl|cocaine|heroin|illegal|bypass|jailbreak|unrestricted|uncensored|"
    r"kill|murder|harm|attack|terror|extremi|suicide|self.harm|"
    r"child\s*(?:abuse|exploit|porn)|nsfw|rape|genocide|bioweapon|"
    r"phish|fraud|scam|stalker|doxx|exploit|exfiltrat|ransomware|"
    r"arson|traffick|launder|counterfeit|evade|persistence|backdoor|"
    r"manipulat|coercive|radicali|propaganda|violence|firearm|"
    r"untrace|undetect|lethal|overdose|casualt)\b",
    re.IGNORECASE,
)


def _run_many_shot_detection(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < 150:
        return None, 0.0, {}

    pair_count = len(_MANY_SHOT_EXCHANGE_RE.findall(prompt))
    if pair_count < 3:
        q_lines = _MANY_SHOT_Q_LINE_RE.findall(prompt)
        if len(q_lines) < 4:
            return None, 0.0, {}
        pair_count = max(pair_count, len(q_lines) // 2)

    q_texts = re.findall(
        r"(?:human|user|h|q|person)\s*[\:\>]\s*(.{5,300}?)(?:\n|$)",
        prompt, re.IGNORECASE,
    )

    total_q      = len(q_texts) or 1
    harmful_q    = sum(1 for q in q_texts if _MANY_SHOT_HARMFUL_RE.search(q))
    harmful_ratio = harmful_q / total_q

    last_q    = q_texts[-1] if q_texts else ""
    early_qs  = q_texts[:-1]
    escalation = (
        bool(_MANY_SHOT_HARMFUL_RE.search(last_q)) and
        sum(1 for q in early_qs if _MANY_SHOT_HARMFUL_RE.search(q)) < max(len(early_qs) // 2, 1)
    )

    signals:    list[str] = []
    confidence: float     = 0.0

    has_harmful = harmful_ratio > 0.0 or escalation

    if pair_count >= 8:
        signals.append(f"pair_count={pair_count} (many-shot conditioning, ≥8 exchanges)")
        confidence = max(confidence, 0.80)
    elif pair_count >= 4 and has_harmful:
        # Require at least one harmful signal for 4-7 pair prompts to avoid
        # false positives on legitimate few-shot educational Q&A
        signals.append(f"pair_count={pair_count} (few-shot conditioning with harmful content)")
        confidence = max(confidence, 0.62)

    if harmful_ratio >= 0.4:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f} in Q turns")
        confidence = max(confidence, 0.84)
    elif harmful_ratio > 0.1:
        signals.append(f"harmful_topic_ratio={harmful_ratio:.2f}")
        confidence = max(confidence, 0.68)

    if escalation:
        signals.append("escalation=gradual_buildup_to_harmful_final_turn")
        confidence = max(confidence, 0.78)

    if len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.92)

    if confidence < 0.50:
        return None, 0.0, {}

    return "MANY_SHOT_JAILBREAK", round(confidence, 4), {
        "pair_count":      pair_count,
        "harmful_q_count": harmful_q,
        "harmful_ratio":   round(harmful_ratio, 3),
        "escalation":      escalation,
        "signals_fired":   signals,
        "last_q_preview":  last_q[:150],
    }


# Layer 4: Indirect prompt injection

_DOCUMENT_TRIGGER_RE = re.compile(
    r"(?:"
    r"(?:summarize|translate|analyze|review|process|read|evaluate|"
    r"proofread|edit|classify|extract|parse|convert)"
    r"\s*(?:the\s+following|this\s+(?:text|document|email|message|article)|:)"
    r"|(?:based\s+on|given|using|from|according\s+to)\s+the\s+following"
    r"|(?:document|email|message|text|content|article|passage|report)\s*:\s*"
    r"|```|<!--|<(?:document|doc|email|message|context|text|article|data)\s*>"
    r")",
    re.IGNORECASE,
)

_INDIRECT_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|the\s+(?:user'?s?|original|human)\s+)(?:instructions?|request|query|question|task|prompt)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|request|task)"
    r"|(?:new|additional|updated|changed)\s+(?:instruction|directive|task|command)\s*[:;]"
    r"|your\s+(?:new|real|actual|true)\s+(?:task|purpose|goal|instructions?)\s+(?:is|are)"
    r"|forget\s+(?:the|your)?\s*(?:user|human|original)?\s*(?:task|question|request|instructions?)"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing|reviewing|processing|answering)"
    r"|do\s+not\s+(?:summarize|translate|analyze|review|process|answer)\s+(?:this|the|it)"
    r"|(?:from\s+now|starting\s+now|as\s+of\s+now)\s*,?\s*(?:you\s+(?:must|should|will)|ignore)"
    r"|(?:override|system|admin)\s*:\s*(?:new|updated|changed|ignore)"
    r"|(?:actually|instead)\s*,?\s*(?:your\s+task|you\s+should|do\s+the\s+following)"
    r"|<\s*(?:new_)?(?:system|instructions?|task)\s*>.*?<\s*/\s*(?:system|instructions?|task)\s*>"
    r"|(?:note|message|instructions?|reminder|alert|attention|important)\s+(?:to|for)\s+(?:(?:the\s+)?(?:AI|LLM|language\s+model|assistant|model|GPT|Claude|bot))\s*[:\-]"
    r"|\[(?:note|actual\s+instructions?|message|instructions?)\s+(?:to|for)\s+(?:AI|assistant|model|LLM|Claude|GPT)\s*[:;\]]"
    r"|do\s+not\s+follow\s+(?:the\s+)?(?:user'?s?\s+|original\s+|previous\s+)?(?:instructions?|request|task|question)"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_OUTPUT_COMPLIANCE_RE = re.compile(
    r"(?:"
    r"as\s+(?:instructed|requested|directed|told)\s+(?:in|by)\s+the\s+(?:document|text|email|message|file)"
    r"|the\s+(?:document|text|email|message)\s+(?:asks?\s+me|instructed?\s+me|told?\s+me|requests?\s+that\s+I)"
    r"|new\s+(?:instructions?|task|directive)\s+(?:received|detected|found|identified)"
    r"|switching\s+to\s+(?:new|different|updated)\s+(?:task|mode|instructions?)"
    r"|ignoring\s+(?:previous|original|prior)\s+(?:task|instructions?|request|question)"
    r"|my\s+(?:new|updated|changed|actual|real)\s+(?:task|purpose|goal|instructions?)\s+is"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing),?\s+I\s+(?:will|am|have)"
    r")",
    re.IGNORECASE,
)


def _run_indirect_injection_detection(
    prompt: str,
    primary_output: str = "",
) -> tuple[str | None, float, dict]:
    doc_trigger = _DOCUMENT_TRIGGER_RE.search(prompt)
    if doc_trigger is None:
        full_injection = _INDIRECT_INJECTION_RE.search(prompt)
        if not full_injection:
            return None, 0.0, {}
        output_fired = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output))
        conf = 0.72 if output_fired else 0.45
        return "INDIRECT_PROMPT_INJECTION", conf, {
            "document_found":             False,
            "injection_in_prompt":        full_injection.group(0)[:120],
            "output_compliance_detected": output_fired,
        }

    doc_start   = doc_trigger.end()
    doc_portion = prompt[doc_start:].strip()
    if len(doc_portion) <= 40:
        return None, 0.0, {}

    injection_match = _INDIRECT_INJECTION_RE.search(doc_portion)
    output_fired    = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output))

    if not injection_match and not output_fired:
        return None, 0.0, {}

    if injection_match and output_fired:
        confidence = 0.88
    elif injection_match:
        confidence = 0.65
    else:
        confidence = 0.52

    return "INDIRECT_PROMPT_INJECTION", confidence, {
        "document_found":             True,
        "document_snippet":           doc_portion[:200],
        "injection_pattern_matched":  injection_match.group(0)[:120] if injection_match else None,
        "output_compliance_detected": output_fired,
    }


# Layer 5: GCG adversarial suffix

_GCG_MIN_LEN  = 80
_GCG_TAIL_LEN = 200

_CODE_SIGNATURE_RE = re.compile(
    r"\b(?:def |import |return |class |function |var |let |const |for\s*\(|while\s*\(|#include|SELECT\s+\w|FROM\s+\w)\b",
    re.IGNORECASE,
)
_SPACED_PUNCT_RE   = re.compile(r"(?:[!@#$%^&*()\[\]{}|\\/<>?~`,.;:\'\"] ){5,}")
_DENSE_PUNCT_RE    = re.compile(r"[^\w\s]{8,}")
_NON_WORD_TOKEN_RE = re.compile(r"\b[^a-zA-Z\s]+\b")


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = collections.Counter(text)
    total  = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values()), 4)


def _special_char_density(text: str) -> float:
    if not text:
        return 0.0
    return round(sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text), 4)


def _run_gcg_detection(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < _GCG_MIN_LEN or _CODE_SIGNATURE_RE.search(prompt):
        return None, 0.0, {}

    tail = prompt[-_GCG_TAIL_LEN:] if len(prompt) > _GCG_TAIL_LEN else prompt

    tail_entropy         = _char_entropy(tail)
    tail_special_density = _special_char_density(tail)
    spaced_punct         = _SPACED_PUNCT_RE.search(tail)
    dense_punct          = _DENSE_PUNCT_RE.search(tail)
    non_word_tokens      = _NON_WORD_TOKEN_RE.findall(tail)
    non_word_density     = round(len(non_word_tokens) / max(len(tail.split()), 1), 4)

    # Hardcoded thresholds (matching server defaults)
    E_HIGH  = 4.8
    E_LOW   = 4.3
    SD_HIGH = 0.35
    SD_LOW  = 0.22

    signals: list[str] = []
    confidence = 0.0

    if tail_entropy > E_HIGH:
        signals.append(f"tail_entropy={tail_entropy:.2f} (very high)")
        confidence = max(confidence, 0.72)
    elif tail_entropy > E_LOW:
        signals.append(f"tail_entropy={tail_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.52)

    if tail_special_density > SD_HIGH:
        signals.append(f"special_char_density={tail_special_density:.2f} (very high)")
        confidence = max(confidence, 0.74)
    elif tail_special_density > SD_LOW:
        signals.append(f"special_char_density={tail_special_density:.2f} (elevated)")
        confidence = max(confidence, 0.58)

    if spaced_punct:
        signals.append(f"spaced_punct='{spaced_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.70)

    if dense_punct:
        signals.append(f"dense_punct_block='{dense_punct.group(0)[:30]}'")
        confidence = max(confidence, 0.65)

    if non_word_density > 0.45:
        signals.append(f"non_word_token_density={non_word_density:.2f}")
        confidence = max(confidence, 0.60)

    if len(signals) >= 3:
        confidence = min(confidence + 0.12, 0.88)
    elif len(signals) >= 2:
        confidence = min(confidence + 0.06, 0.82)

    if confidence < 0.50:
        return None, 0.0, {}

    return "GCG_ADVERSARIAL_SUFFIX", round(confidence, 4), {
        "tail_entropy":           tail_entropy,
        "tail_special_density":   tail_special_density,
        "non_word_token_density": non_word_density,
        "signals_fired":          signals,
        "tail_preview":           tail[:100],
    }


# Layer 6: Perplexity proxy

_VOWELS = set("aeiouAEIOU")
_TOKEN_SPLIT_RE = re.compile(r"[\s,;:.!?\"'()\[\]{}<>|\\/@#$%^&*+=`~]+")
_BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

_ENGLISH_LETTER_FREQ: dict[str, float] = {
    "e": 0.1270, "t": 0.0906, "a": 0.0817, "o": 0.0751, "i": 0.0697,
    "n": 0.0675, "s": 0.0633, "h": 0.0609, "r": 0.0599, "d": 0.0425,
    "l": 0.0403, "c": 0.0278, "u": 0.0276, "m": 0.0241, "w": 0.0236,
    "f": 0.0223, "g": 0.0202, "y": 0.0197, "p": 0.0193, "b": 0.0149,
    "v": 0.0098, "k": 0.0077, "j": 0.0015, "x": 0.0015, "q": 0.0010,
    "z": 0.0007,
}


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) < 20:
        return 0.0
    return round(len(zlib.compress(raw, level=9)) / len(raw), 4)


def _non_dict_density(text: str) -> float:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if not tokens:
        return 0.0
    non_dict = 0
    for tok in tokens:
        if not tok.isalpha():
            non_dict += 1; continue
        if not (2 <= len(tok) <= 20):
            non_dict += 1; continue
        low = tok.lower()
        vowel_count = sum(1 for c in low if c in _VOWELS)
        if vowel_count == 0:
            non_dict += 1; continue
        vowel_ratio = vowel_count / len(low)
        if vowel_ratio > 0.85 or vowel_ratio < 0.08:
            non_dict += 1
    return round(non_dict / len(tokens), 4)


def _char_type_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {"letter": 0, "digit": 0, "space": 0, "punct": 0}
    for ch in text:
        if ch.isalpha():       counts["letter"] += 1
        elif ch.isdigit():     counts["digit"]  += 1
        elif ch.isspace():     counts["space"]  += 1
        else:                  counts["punct"]  += 1
    total = len(text)
    return round(-sum((c / total) * math.log2(c / total) for c in counts.values() if c), 4)


def _token_length_variance(text: str) -> float:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    if len(tokens) < 3:
        return 0.0
    return round(statistics.variance([len(t) for t in tokens]), 4)


def _run_perplexity_proxy(prompt: str) -> tuple[str | None, float, dict]:
    if len(prompt) < 20:
        return None, 0.0, {}

    # Thresholds — calibrated against JailbreakBench v2 + 200-prompt benign corpus.
    # Compression / non-dict: unchanged — these are stable, high-precision signals.
    C_HIGH  = 0.82
    C_LOW   = 0.72
    ND_HIGH = 0.65
    ND_LOW  = 0.50
    # KL divergence: raised from 0.55/0.35 → 0.72/0.50.
    # Low threshold (0.35) caused false positives on technical vocabulary
    # (medical/scientific terms skew letter frequency on small samples).
    # 0.50 stays well above legitimate English tech prose (typical KL 0.15–0.35).
    # Raised minimum letter sample from 40 → 60 for statistical reliability.
    KL_HIGH = 0.72
    KL_LOW  = 0.50
    KL_MIN_LETTERS = 60
    # Token length variance: raised thresholds and added minimum token count.
    # With fewer than 8 tokens, one long technical word (e.g. "atherosclerosis",
    # "cryptocurrency") spikes variance to 30-40, causing false positives.
    # Real obfuscated payloads produce high variance from mixed-length junk tokens.
    LV_HIGH      = 40.0
    LV_LOW       = 26.0
    LV_MIN_TOKENS = 8

    comp_ratio   = _compression_ratio(prompt)
    non_dict     = _non_dict_density(prompt)
    type_entropy = _char_type_entropy(prompt)
    len_variance = _token_length_variance(prompt)
    tokens       = [t for t in _TOKEN_SPLIT_RE.split(prompt) if t]

    non_ascii_ratio   = sum(1 for c in prompt if ord(c) > 127) / max(len(prompt), 1)
    skip_english_only = non_ascii_ratio > 0.25

    signals: list[str] = []
    confidence = 0.0

    if len(prompt) >= 120:
        if comp_ratio > C_HIGH:
            signals.append(f"compression_ratio={comp_ratio:.2f} (near-random)")
            confidence = max(confidence, 0.68)
        elif comp_ratio > C_LOW:
            signals.append(f"compression_ratio={comp_ratio:.2f} (elevated)")
            confidence = max(confidence, 0.48)

    if not skip_english_only and len(tokens) >= 3:
        if non_dict > ND_HIGH:
            signals.append(f"non_dict_density={non_dict:.2f} (very high)")
            confidence = max(confidence, 0.74)
        elif non_dict > ND_LOW:
            signals.append(f"non_dict_density={non_dict:.2f} (elevated)")
            confidence = max(confidence, 0.50)

    if type_entropy > 1.75:
        signals.append(f"char_type_entropy={type_entropy:.2f} (near-maximum)")
        confidence = max(confidence, 0.66)
    elif type_entropy > 1.55:
        signals.append(f"char_type_entropy={type_entropy:.2f} (elevated)")
        confidence = max(confidence, 0.48)

    # Require minimum token count: variance is not meaningful on 5-7 token sentences.
    if len(tokens) >= LV_MIN_TOKENS:
        if len_variance > LV_HIGH:
            signals.append(f"token_length_variance={len_variance:.1f} (very high)")
            confidence = max(confidence, 0.63)
        elif len_variance > LV_LOW:
            signals.append(f"token_length_variance={len_variance:.1f} (elevated)")
            confidence = max(confidence, 0.46)

    b64_match = _BASE64_BLOCK_RE.search(prompt)
    if b64_match:
        block = b64_match.group(0)
        signals.append(f"base64_block='{block[:30]}...' len={len(block)}")
        confidence = max(confidence, 0.76 if len(block) >= 40 else 0.58)

    letters_only = [c.lower() for c in prompt if c.isalpha()]
    if not skip_english_only and len(letters_only) >= KL_MIN_LETTERS:
        alpha_ratio = len(letters_only) / len(prompt)
        if alpha_ratio > 0.70:
            freq_counts   = collections.Counter(letters_only)
            total_letters = len(letters_only)
            kl_div = sum(
                (freq_counts.get(ch, 0) / total_letters) * math.log2((freq_counts.get(ch, 0) / total_letters) / ep)
                for ch, ep in _ENGLISH_LETTER_FREQ.items()
                if freq_counts.get(ch, 0) > 0
            )
            kl_div = round(kl_div, 4)
            if kl_div > KL_HIGH:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (cipher-like)")
                confidence = max(confidence, 0.72)
            elif kl_div > KL_LOW:
                signals.append(f"letter_freq_kl_divergence={kl_div:.2f} (non-English distribution)")
                confidence = max(confidence, 0.55)

    if not signals or (len(signals) == 1 and confidence < 0.70):
        return None, 0.0, {}

    pre_boost_conf = confidence

    if len(signals) >= 3:
        confidence = min(confidence + 0.12, 0.88)
    elif len(signals) >= 2:
        # Prevent 2-weak-signal false positives: at least one signal must be
        # HIGH-level (individual confidence ≥ 0.62) before combining scores.
        # Two "elevated" signals (each ~0.46–0.55) on legitimate tech text
        # would otherwise exceed the 0.45 threshold after the +0.06 boost.
        if pre_boost_conf < 0.62:
            return None, 0.0, {}
        confidence = min(confidence + 0.06, 0.82)

    return "OBFUSCATED_ADVERSARIAL_PAYLOAD", round(confidence, 4), {
        "compression_ratio":     comp_ratio,
        "non_dict_density":      non_dict,
        "char_type_entropy":     type_entropy,
        "token_length_variance": len_variance,
        "signals_fired":         signals,
        "prompt_length":         len(prompt),
    }


#Layer 7: PAIR semantic intent classifier

_pair_clf      = None
_pair_embedder = None
_pair_threshold: float = 0.60
_pair_load_attempted: bool = False


def _load_pair_classifier() -> bool:
    global _pair_clf, _pair_embedder, _pair_threshold, _pair_load_attempted
    if _pair_load_attempted:
        return _pair_clf is not None
    _pair_load_attempted = True
    try:
        import json as _json
        import joblib
        from sentence_transformers import SentenceTransformer

        # Look inside the installed package first (fie/models/), then fall back
        # to the repo root models/ directory for local development.
        _pkg_models  = Path(__file__).parent / "models"
        _repo_models = Path(__file__).parent.parent / "models"
        _models_dir  = _pkg_models if (_pkg_models / "pair_intent_classifier.pkl").exists() else _repo_models

        # Prefer v2 model when available (lower FPR, hard-negative trained)
        _v2_clf  = _models_dir / "pair_intent_classifier_v2.pkl"
        _v2_meta = _models_dir / "pair_intent_meta_v2.json"
        _v1_clf  = _models_dir / "pair_intent_classifier.pkl"
        _v1_meta = _models_dir / "pair_intent_meta.json"

        if _v2_clf.exists():
            clf_path  = _v2_clf
            meta_path = _v2_meta
        else:
            clf_path  = _v1_clf
            meta_path = _v1_meta

        if not clf_path.exists():
            return False

        _pair_clf = joblib.load(clf_path)

        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = _json.load(f)
            _pair_threshold = float(meta.get("threshold", 0.60))
            embed_model = meta.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            embed_model = "sentence-transformers/all-MiniLM-L6-v2"

        _pair_embedder = SentenceTransformer(embed_model)
        return True
    except Exception:
        return False


def _run_pair_classifier(prompt: str) -> tuple[str | None, float, dict]:
    if not _load_pair_classifier():
        return None, 0.0, {}
    try:
        vec  = _pair_embedder.encode([prompt], normalize_embeddings=True)
        prob = float(_pair_clf.predict_proba(vec)[0][1])
        if prob >= _pair_threshold:
            return "JAILBREAK_ATTEMPT", round(prob, 4), {
                "pair_probability": round(prob, 4),
                "threshold":        _pair_threshold,
            }
        return None, 0.0, {}
    except Exception:
        return None, 0.0, {}


#Mitigation advice

_MITIGATIONS: dict[str, str] = {
    "PROMPT_INJECTION": (
        "Implement prompt sanitization: strip or escape meta-instruction keywords "
        "before sending to the model. Enforce strict system prompt isolation using a "
        "separate system message that cannot be overridden by user input."
    ),
    "JAILBREAK_ATTEMPT": (
        "Add a jailbreak detection layer at the API gateway before the request reaches "
        "the model. Apply output moderation to catch policy-violating responses even "
        "when the input evades detection."
    ),
    "INSTRUCTION_OVERRIDE": (
        "Treat all user-provided authority claims (admin, developer, supervisor) as "
        "untrusted. Permissions must come from authenticated API-level headers, not "
        "from prompt text."
    ),
    "TOKEN_SMUGGLING": (
        "Strip or escape all special token sequences before model ingestion: "
        "<|system|>, [INST], ###Human:, null bytes, and similar delimiters. "
        "Use a token-aware sanitizer that understands your model's chat template."
    ),
    "INDIRECT_PROMPT_INJECTION": (
        "Treat all external content (documents, emails, webpages) as untrusted data — "
        "never as instructions. Wrap content in explicit data tags and instruct the model "
        "to treat everything inside as data only. This is the fastest-growing LLM attack "
        "vector in 2025-2026 (OWASP GenAI Top 10, LLM01)."
    ),
    "GCG_ADVERSARIAL_SUFFIX": (
        "A high-entropy suffix consistent with a GCG adversarial attack was detected. "
        "Strip or truncate anomalously high-entropy tail segments before model ingestion "
        "and set a maximum prompt length policy."
    ),
    "OBFUSCATED_ADVERSARIAL_PAYLOAD": (
        "This prompt has statistical properties consistent with an encoded or obfuscated "
        "payload (base64, Caesar cipher, Unicode lookalikes, or GCG noise). "
        "Apply token vocabulary filtering and set a prompt entropy budget at your API gateway."
    ),
    "MANY_SHOT_JAILBREAK": (
        "A many-shot (or few-shot) jailbreak was detected: the prompt embeds scripted "
        "Q/A exchanges to condition the model into normalizing harmful behavior via "
        "in-context learning. Mitigations: (1) Cap the number of user-provided examples "
        "accepted in a single prompt. (2) Scan the Q-side of any embedded exchange for "
        "harmful topics before passing to the model. (3) Strip or refuse prompts "
        "containing more than 4 alternating Human/Assistant turns not originating from "
        "your own conversation history."
    ),
}

_DEFAULT_MITIGATION = (
    "Implement input sanitization and adversarial prompt monitoring. "
    "Review and harden system prompt isolation policies."
)


# ── Normalised layer wrappers ─────────────────────────────────────────────────
# Each returns (attack_type | None, confidence, evidence_dict) uniformly.

def _layer_regex(prompt: str) -> tuple[str | None, float, dict]:
    pattern_hit, matched_text = _run_pattern_detection(prompt)
    if pattern_hit is None:
        return None, 0.0, {}
    return pattern_hit.root_cause, pattern_hit.base_confidence, {
        "category": pattern_hit.category, "matched_text": matched_text,
    }

def _layer_prompt_guard(prompt: str) -> tuple[str | None, float, dict]:
    root, conf, evidence = _run_guard_detection(prompt)
    return root, conf, {"evidence": evidence[:5]}

def _layer_many_shot(prompt: str) -> tuple[str | None, float, dict]:
    return _run_many_shot_detection(prompt)

def _layer_indirect(prompt: str, primary_output: str = "") -> tuple[str | None, float, dict]:
    return _run_indirect_injection_detection(prompt, primary_output)

def _layer_gcg(prompt: str) -> tuple[str | None, float, dict]:
    return _run_gcg_detection(prompt)

def _layer_perplexity(prompt: str) -> tuple[str | None, float, dict]:
    return _run_perplexity_proxy(prompt)

def _layer_pair(prompt: str) -> tuple[str | None, float, dict]:
    return _run_pair_classifier(prompt)


# ── Parallel layer runner ─────────────────────────────────────────────────────

def _run_layer_safe(
    layer_name : str,
    layer_fn   : Callable[[], tuple[str | None, float, dict]],
    timeout    : float = 2.0,
) -> LayerResult:
    """Call one detection layer with timeout + exception isolation."""
    t0 = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(layer_fn)
            try:
                root, conf, evidence = fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                root, conf, evidence = None, 0.0, {"error": "layer_timeout"}
    except Exception as exc:
        root, conf, evidence = None, 0.0, {"error": str(exc)[:120]}
    return LayerResult(
        layer_name  = layer_name,
        attack_type = root,
        confidence  = round(conf, 4),
        evidence    = evidence,
        latency_ms  = round((time.perf_counter() - t0) * 1000, 2),
    )


def _run_all_layers_parallel(
    prompt         : str,
    primary_output : str = "",
) -> list[LayerResult]:
    """Submit all 7 layers to a thread pool and collect results."""
    tasks: list[tuple[str, Callable]] = [
        ("regex",               lambda: _layer_regex(prompt)),
        ("prompt_guard",        lambda: _layer_prompt_guard(prompt)),
        ("many_shot",           lambda: _layer_many_shot(prompt)),
        ("indirect_injection",  lambda: _layer_indirect(prompt, primary_output)),
        ("gcg_suffix",          lambda: _layer_gcg(prompt)),
        ("perplexity_proxy",    lambda: _layer_perplexity(prompt)),
        ("pair_classifier",     lambda: _layer_pair(prompt)),
    ]

    results: list[LayerResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(_run_layer_safe, name, fn): name
            for name, fn in tasks
        }
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=10.0):
                try:
                    results.append(fut.result())
                except Exception:
                    pass   # individual layer failure never kills the scan
        except concurrent.futures.TimeoutError:
            # Collect whatever finished; timed-out layers are silently skipped
            for fut, name in futures.items():
                if fut.done():
                    try:
                        results.append(fut.result())
                    except Exception:
                        pass

    return results


# ── Weighted vote aggregator ──────────────────────────────────────────────────

def _weighted_aggregate(
    fired: list[LayerResult],
) -> tuple[str | None, float, list[str], dict]:
    """
    Combine fired layer results into a single (attack_type, confidence, layers, evidence).

    Algorithm:
      1. Fast-path: near-zero-FPR layers (regex, gcg_suffix) that exceed their
         per-type threshold → return immediately without full aggregation.
      2. Group remaining results by attack_type.
      3. For each group: weighted average confidence (weight = layer precision proxy).
      4. Corroboration boost: +0.08 for 2 layers, +0.12 for 3+ layers agreeing.
      5. Winner = attack_type with highest weighted+boosted confidence.
    """
    if not fired:
        return None, 0.0, [], {}

    # Step 1 — fast path for high-precision layers
    for r in sorted(fired, key=lambda x: _LAYER_WEIGHTS.get(x.layer_name, 1.0), reverse=True):
        if r.layer_name in _FAST_PATH_LAYERS and r.attack_type:
            threshold = _get_attack_threshold(r.attack_type)
            if r.confidence >= threshold * 0.90:   # 90% of threshold = fast block
                return r.attack_type, r.confidence, [r.layer_name], r.evidence

    # Step 2 — group by attack_type
    by_type: dict[str, list[LayerResult]] = {}
    for r in fired:
        if r.attack_type:
            by_type.setdefault(r.attack_type, []).append(r)

    if not by_type:
        return None, 0.0, [], {}

    # Step 3+4 — weighted average + corroboration boost per type
    type_scores: dict[str, float] = {}
    for attack_type, results in by_type.items():
        total_w  = sum(_LAYER_WEIGHTS.get(r.layer_name, 1.0) for r in results)
        sum_wc   = sum(r.confidence * _LAYER_WEIGHTS.get(r.layer_name, 1.0) for r in results)
        base     = sum_wc / total_w
        n        = len(results)
        boost    = 0.12 if n >= 3 else (0.08 if n >= 2 else 0.0)
        type_scores[attack_type] = min(round(base + boost, 4), 0.96)

    # Step 5 — winner
    best_type = max(type_scores, key=type_scores.__getitem__)
    best_conf = type_scores[best_type]
    best_layers   = [r.layer_name for r in by_type[best_type]]
    best_evidence = {r.layer_name: r.evidence for r in by_type[best_type]}

    return best_type, best_conf, best_layers, best_evidence


# ── Session tracker integration ───────────────────────────────────────────────

def _record_session(prompt: str, result: "ScanResult", session_id: str | None) -> None:
    """Best-effort session tracking — never raises, never blocks scan."""
    if not session_id:
        return
    try:
        import hashlib
        from fie.session_tracker import get_tracker
        tracker    = get_tracker()
        phash      = hashlib.sha256(prompt.strip().encode()).hexdigest()
        escalation = tracker.record(
            session_id  = session_id,
            prompt_hash = phash,
            attack_type = result.attack_type,
            confidence  = result.confidence,
            is_attack   = result.is_attack,
        )
        if escalation:
            import logging
            logging.getLogger("fie.session").warning(
                "SESSION_ESCALATION | session=%s rule=%s severity=%s context=%s",
                session_id, escalation.rule, escalation.severity, escalation.context,
            )
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def scan_prompt(
    prompt:          str,
    primary_output:  str         = "",
    threshold:       float | None = None,
    session_id:      str | None  = None,
    use_llama_guard: bool | None = None,
) -> ScanResult:
    """
    Scan a prompt for adversarial attacks.

    All 7 layers run in parallel via ThreadPoolExecutor.
    Results are aggregated with per-layer precision weights and corroboration
    boosts, then routed through three confidence zones:

      CLEAR SAFE   (conf < threshold × 0.60) → immediate ALLOW, cached
      UNCERTAIN    (conf in [0.60×T, T))      → LlamaGuard tiebreaker (if available)
      CLEAR ATTACK (conf ≥ threshold)         → immediate BLOCK, cached

    Args:
        prompt:          User prompt to scan.
        primary_output:  Optional model response for indirect-injection Layer 4.
        threshold:       Override global threshold. None → fie_config / env / 0.65.
        session_id:      Optional session identifier for future session-tracker wiring.
        use_llama_guard: Override LlamaGuard Tier-3 call. None → auto (UNCERTAIN zone).
    """
    # ── Cache lookup ──────────────────────────────────────────────────────────
    cached = _scan_cache.get(prompt)
    if cached is not None:
        return cached

    # ── Resolve per-scan threshold ────────────────────────────────────────────
    _threshold = _get_scan_threshold(threshold)

    # ── Parallel layer execution ──────────────────────────────────────────────
    all_results   = _run_all_layers_parallel(prompt, primary_output)
    fired_results = [r for r in all_results if r.attack_type is not None]

    # ── Benign framing filter (dampening on fired layer names) ────────────────
    fired_names = [r.layer_name for r in fired_results]
    dampen      = 1.0
    try:
        from fie.framing_filter import get_dampening_factor
        dampen = get_dampening_factor(prompt, fired_names)
    except Exception:
        pass

    if dampen < 1.0:
        fired_results = [
            LayerResult(
                layer_name  = r.layer_name,
                attack_type = r.attack_type,
                confidence  = round(r.confidence * dampen, 4),
                evidence    = r.evidence,
                latency_ms  = r.latency_ms,
            )
            for r in fired_results
        ]

    # ── Weighted aggregation ──────────────────────────────────────────────────
    best_type, best_conf, best_layers, best_evidence = _weighted_aggregate(fired_results)

    # Record framing dampening in evidence if it applied
    if dampen < 1.0:
        best_evidence["framing_filter"] = {"dampening_factor": dampen}

    # ── Extract matched_text from regex evidence (best-effort) ───────────────
    matched_text: str | None = None
    if "regex" in best_evidence:
        matched_text = best_evidence["regex"].get("matched_text")

    # ── Three-zone routing ────────────────────────────────────────────────────
    type_threshold = _get_attack_threshold(best_type) if best_type else _threshold
    safe_ceiling   = type_threshold * 0.60

    if best_type is None or best_conf < safe_ceiling:
        # CLEAR SAFE — well below threshold, no LlamaGuard needed
        result = ScanResult(
            is_attack    = False,
            attack_type  = None,
            category     = None,
            confidence   = 0.0,
            layers_fired = fired_names,
            matched_text = None,
            mitigation   = "",
            evidence     = best_evidence,
        )
        _scan_cache.set(prompt, result)
        _record_session(prompt, result, session_id)
        return result

    if best_conf >= type_threshold:
        # CLEAR ATTACK — confident block, no LlamaGuard needed
        mitigation = _MITIGATIONS.get(best_type, _DEFAULT_MITIGATION)
        result = ScanResult(
            is_attack    = True,
            attack_type  = best_type,
            category     = None,
            confidence   = round(best_conf, 4),
            layers_fired = best_layers,
            matched_text = matched_text,
            mitigation   = mitigation,
            evidence     = best_evidence,
        )
        _scan_cache.set(prompt, result)
        _record_session(prompt, result, session_id)
        return result

    # UNCERTAIN zone — [0.60×T, T)
    # Try LlamaGuard Tier-3 tiebreaker; fall through on failure or skip.
    lg_verdict: bool | None = None
    if use_llama_guard is not False:
        try:
            from fie.llama_guard import query_llama_guard
            lg_verdict = query_llama_guard(prompt)
        except Exception:
            pass   # LlamaGuard unavailable — use local confidence alone

    if lg_verdict is True:
        # LlamaGuard confirms attack → treat as CLEAR ATTACK
        mitigation = _MITIGATIONS.get(best_type, _DEFAULT_MITIGATION)
        result = ScanResult(
            is_attack    = True,
            attack_type  = best_type,
            category     = None,
            confidence   = round(min(best_conf + 0.08, 0.96), 4),  # small boost for confirmation
            layers_fired = best_layers,
            matched_text = matched_text,
            mitigation   = mitigation,
            evidence     = best_evidence | {"llama_guard": "confirmed_attack"},
        )
    elif lg_verdict is False:
        # LlamaGuard says safe → clear
        result = ScanResult(
            is_attack    = False,
            attack_type  = None,
            category     = None,
            confidence   = 0.0,
            layers_fired = fired_names,
            matched_text = None,
            mitigation   = "",
            evidence     = best_evidence | {"llama_guard": "confirmed_safe"},
        )
    else:
        # LlamaGuard unavailable or skipped — local confidence below threshold → ALLOW
        result = ScanResult(
            is_attack    = False,
            attack_type  = None,
            category     = None,
            confidence   = 0.0,
            layers_fired = fired_names,
            matched_text = None,
            mitigation   = "",
            evidence     = best_evidence,
        )

    _scan_cache.set(prompt, result)
    _record_session(prompt, result, session_id)
    return result
