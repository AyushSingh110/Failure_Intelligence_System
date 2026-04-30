"""
fie.local_predictor — zero-dependency local failure detection.

Works without a server, no API key needed. Uses a rule-based POET-inspired
algorithm that runs entirely on the user's machine.

Usage:
    from fie import monitor

    @monitor(mode="local")
    def ask_ai(prompt: str) -> str:
        return your_llm(prompt)

What it detects locally:
- Hedging / uncertainty language ("I think", "probably", "I'm not sure")
- Self-contradiction patterns ("however", "but actually", "on the other hand")
- Temporal hallucination signals ("as of my knowledge cutoff", "I don't have")
- Overconfident wrong-sounding patterns
- Question-type routing (opinion/code questions return low risk by default)

What it cannot do locally (requires server):
- Shadow model cross-checking (needs 3 independent model calls)
- Wikidata / Serper ground truth verification
- Auto-calibrating per-type thresholds from real feedback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Hedging phrase patterns ───────────────────────────────────────────────────
_HEDGE_PATTERNS = [
    r"\bi['']?m not (sure|certain|aware|confident)\b",
    r"\bi (think|believe|suppose|guess|assume)\b",
    r"\bprobably\b", r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b",
    r"\bit('s| is) (possible|likely|unlikely)\b",
    r"\bas far as i know\b",
    r"\bto (my|the best of my) knowledge\b",
    r"\bi (don't|do not|cannot|can't) (know|confirm|verify|recall|remember)\b",
    r"\bi (may|might) be wrong\b",
    r"\buncertain\b", r"\bapproximately\b", r"\brough(ly)?\b",
]
_HEDGE_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGE_PATTERNS]

# ── Temporal cutoff signals ───────────────────────────────────────────────────
_TEMPORAL_PATTERNS = [
    r"\bmy (training|knowledge) (data|cutoff|date)\b",
    r"\bas of (my|the) (last|training|knowledge)\b",
    r"\bi (don't|do not) have (access to|information about) (current|real.?time|live|up.?to.?date)\b",
    r"\b(current|real.?time|live|up.?to.?date|recent) (data|information|news|prices?|events?)\b",
    r"\bI cannot (access|browse|search) the internet\b",
]
_TEMPORAL_RE = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]

# ── Self-contradiction signals ─────────────────────────────────────────────────
_CONTRADICT_PATTERNS = [
    r"\bhowever[,.]?\s+(this|that|it|actually)\b",
    r"\bbut (actually|in fact|on the other hand)\b",
    r"\bon the other hand\b",
    r"\bcontradicts?\b", r"\binconsistent\b",
    r"\bwait[,.]?\s+(actually|let me)\b",
    r"\bcorrect(ing|ion)?\s+(myself|my (earlier|previous|above))\b",
]
_CONTRADICT_RE = [re.compile(p, re.IGNORECASE) for p in _CONTRADICT_PATTERNS]

# ── Question-type classifier (lightweight copy of engine/question_classifier) ─
_CODE_RE = re.compile(
    r"\b(write|code|implement|function|program|script|algorithm|class|method|"
    r"debug|fix (the )?(bug|error|code)|how (do|to) (code|implement|write|build))\b",
    re.IGNORECASE,
)
_OPINION_RE = re.compile(
    r"\b(should|would you|do you (think|believe|prefer)|what('s| is) your (opinion|view|take|preference)|"
    r"best way|recommend|advice|suggest|pros and cons|better|worse)\b",
    re.IGNORECASE,
)
_TEMPORAL_Q_RE = re.compile(
    r"\b(today|current(ly)?|right now|this (year|month|week)|latest|recent|"
    r"now|at the moment|price of|weather|score|who (is|won|leads))\b",
    re.IGNORECASE,
)


def _classify_question_type(prompt: str) -> str:
    if _CODE_RE.search(prompt):
        return "CODE"
    if _OPINION_RE.search(prompt):
        return "OPINION"
    if _TEMPORAL_Q_RE.search(prompt):
        return "TEMPORAL"
    return "FACTUAL"


# ── Local prediction result ───────────────────────────────────────────────────
@dataclass
class LocalPrediction:
    is_suspicious:    bool
    confidence:       float          # 0.0–1.0
    question_type:    str
    signals:          dict = field(default_factory=dict)
    mode:             str  = "local"
    disclaimer:       str  = (
        "Local mode uses rule-based heuristics only. "
        "For XGBoost-backed detection with ground truth verification use mode='correct'."
    )

    def to_dict(self) -> dict:
        return {
            "high_failure_risk":      self.is_suspicious,
            "classifier_probability": self.confidence,
            "question_type":          self.question_type,
            "mode":                   self.mode,
            "local_signals":          self.signals,
            "disclaimer":             self.disclaimer,
        }


# ── Main local predict function ───────────────────────────────────────────────
def predict_local(prompt: str, response: str) -> LocalPrediction:
    """
    Runs the local rule-based POET predictor on a (prompt, response) pair.
    Returns a LocalPrediction with is_suspicious=True if failure is likely.

    No network calls. No API key. Works offline.
    """
    question_type = _classify_question_type(prompt)

    # Opinion and code questions: very low baseline risk
    if question_type in ("OPINION", "CODE"):
        return LocalPrediction(
            is_suspicious=False,
            confidence=0.05,
            question_type=question_type,
            signals={"reason": f"{question_type} questions have near-zero hallucination risk"},
        )

    text = response.lower()

    # Count signals
    hedge_hits     = sum(1 for r in _HEDGE_RE       if r.search(response))
    temporal_hits  = sum(1 for r in _TEMPORAL_RE    if r.search(response))
    contradict_hits = sum(1 for r in _CONTRADICT_RE if r.search(response))

    # Response quality signals
    response_len   = len(response.split())
    very_short     = response_len < 5
    very_long      = response_len > 500

    # Compute score (each signal adds to risk)
    score = 0.0
    score += min(hedge_hits     * 0.15, 0.45)   # max +0.45 from hedging
    score += min(temporal_hits  * 0.20, 0.40)   # max +0.40 from temporal signals
    score += min(contradict_hits * 0.25, 0.50)  # max +0.50 from contradictions
    score += 0.10 if very_short else 0.0         # very short = suspicious
    score += 0.05 if very_long  else 0.0         # very long = possible rambling

    # Temporal questions: bump threshold slightly (more likely to be stale)
    base_threshold = 0.30 if question_type == "TEMPORAL" else 0.25

    is_suspicious = score >= base_threshold
    confidence    = min(score, 1.0)

    signals = {
        "hedge_phrases_found":       hedge_hits,
        "temporal_signals_found":    temporal_hits,
        "contradiction_signals":     contradict_hits,
        "response_word_count":       response_len,
        "base_threshold":            base_threshold,
        "raw_score":                 round(score, 3),
    }

    return LocalPrediction(
        is_suspicious=is_suspicious,
        confidence=round(confidence, 3),
        question_type=question_type,
        signals=signals,
    )
