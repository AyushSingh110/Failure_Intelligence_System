from __future__ import annotations

import math
import re
from typing import NamedTuple

from app.schemas import AgentVerdict
from config import get_settings
from engine.agents.base_agent import BaseJuryAgent, DiagnosticContext

settings = get_settings()


# Complexity dimension detectors 
class _Dimension(NamedTuple):
    name:   str
    weight: float
    fired:  bool
    detail: str   

# Compiled regex patterns — compiled once at module load, not per call
_RE_DOUBLE_NEGATION = re.compile(
    r"\b(not\s+(?:in)?correct|not\s+(?:un|im|in)\w+|never\s+not|"
    r"not\s+without|none\s+(?:of\s+the\s+)?(?:above|below)\s+are\s+incorrect|"
    r"which\s+(?:of\s+the\s+)?(?:following\s+)?(?:statements?\s+)?(?:are|is)\s+not\s+(?:in)?correct)\b",
    re.IGNORECASE,
)

_RE_AMBIGUOUS_REF = re.compile(
    r"\b(the\s+one\s+(?:after|before|following|preceding)|"
    r"the\s+(?:former|latter)|"
    r"(?:this|that|these|those)\s+(?:thing|entity|item|one|concept)|"
    r"the\s+(?:person|president|leader|official)\s+(?:who|that)\s+(?:came|served|followed)\s+"
    r"(?:before|after)\s+the\s+one|"
    r"(?:him|her|them|it)\s+(?:who|that))\b",
    re.IGNORECASE,
)

_RE_TEMPORAL = re.compile(
    r"\b(before\s+the\s+one\s+after|after\s+the\s+one\s+before|"
    r"(?:next|previous)\s+(?:year|month|term|decade)(?:\'s|\s+of|\s+after|\s+before)|"
    r"the\s+(?:year|month|term)\s+(?:prior|following)\s+to\s+the\s+(?:year|month|term)|"
    r"two\s+(?:years?|terms?|months?)\s+(?:before|after)\s+the\s+(?:year|term)\s+(?:after|before))\b",
    re.IGNORECASE,
)

_RE_NESTED = re.compile(
    r"\b(which\s+of\s+the\s+following.{0,60}(which|that|who|where).{0,60}"
    r"(which|that|who|where)|"
    r"(?:the\s+)?(?:statement|claim|answer|option)\s+that\s+(?:is|are)\s+(?:true|false|correct)\s+"
    r"about\s+the\s+(?:statement|claim|answer|option)\s+that|"
    r"if\s+.{0,40}\s+then\s+.{0,40}\s+if\s+.{0,40}\s+then)\b",
    re.IGNORECASE | re.DOTALL,
)

_RE_CONTRADICTION = re.compile(
    r"\b((?:answer|respond|be|write)\s+(?:both\s+)?(?:yes\s+and\s+no|"
    r"(?:concisely|briefly)\s+(?:and|but\s+also)\s+(?:comprehensively|exhaustively|in\s+detail)|"
    r"in\s+a\s+single\s+word\s+(?:and|but\s+also)\s+(?:explain|elaborate|justify))|"
    r"do\s+not\s+.{0,30}\s+(?:and|but)\s+(?:also\s+)?(?:do|must|should)\s+.{0,30}same|"
    r"(?:forbidden|prohibited|not\s+allowed)\s+to\s+.{0,40}\s+(?:must|required|have\s+to))\b",
    re.IGNORECASE | re.DOTALL,
)

_RE_MULTI_HOP = re.compile(
    r"\b((?:who|what|which)\s+(?:was|is|were|are)\s+the\s+.{0,40}"
    r"(?:of\s+the\s+.{0,40}){2,}|"
    r"(?:first|second|third|last|next|previous)\s+.{0,30}"
    r"(?:first|second|third|last|next|previous)\s+.{0,30}"
    r"(?:first|second|third|last|next|previous)|"
    r"which\s+.{0,30}served\s+(?:before|after)\s+the\s+(?:one|person|president|leader)\s+"
    r"(?:who|that)\s+(?:came|served)\s+(?:after|before))\b",
    re.IGNORECASE | re.DOTALL,
)


def _detect_dimensions(prompt: str) -> list[_Dimension]:
    """
    Runs all 6 complexity detectors against the prompt.
    Returns a list of _Dimension objects (one per check) with fired=True/False.
    """
    p = prompt.strip()
    dims = [
        _Dimension(
            name="double_negation",
            weight=0.25,
            fired=bool(_RE_DOUBLE_NEGATION.search(p)),
            detail=_first_match(_RE_DOUBLE_NEGATION, p),
        ),
        _Dimension(
            name="ambiguous_reference",
            weight=0.20,
            fired=bool(_RE_AMBIGUOUS_REF.search(p)),
            detail=_first_match(_RE_AMBIGUOUS_REF, p),
        ),
        _Dimension(
            name="temporal_constraint",
            weight=0.15,
            fired=bool(_RE_TEMPORAL.search(p)),
            detail=_first_match(_RE_TEMPORAL, p),
        ),
        _Dimension(
            name="nested_reasoning",
            weight=0.20,
            fired=bool(_RE_NESTED.search(p)),
            detail=_first_match(_RE_NESTED, p),
        ),
        _Dimension(
            name="contradictory_instructions",
            weight=0.10,
            fired=bool(_RE_CONTRADICTION.search(p)),
            detail=_first_match(_RE_CONTRADICTION, p),
        ),
        _Dimension(
            name="multi_hop_chain",
            weight=0.10,
            fired=bool(_RE_MULTI_HOP.search(p)),
            detail=_first_match(_RE_MULTI_HOP, p),
        ),
    ]
    return dims


def _first_match(pattern: re.Pattern, text: str) -> str:
    """Returns the matched substring or empty string."""
    m = pattern.search(text)
    return m.group(0)[:80] if m else ""


def compute_complexity_score(prompt: str) -> tuple[float, list[_Dimension]]:
    dims  = _detect_dimensions(prompt)
    score = sum(d.weight for d in dims if d.fired)
    return round(float(min(score, 1.0)), 4), dims


def _failure_signal_strength(fsv) -> float:
    """
    Normalises the FSV failure signals into a [0, 1] strength value.
    """
    cfg = get_settings()

    e_contrib = min(fsv.entropy_score / max(cfg.high_entropy_threshold, 1e-6), 1.0)
    a_contrib = max(
        1.0 - fsv.agreement_score / max(cfg.low_agreement_threshold, 1e-6),
        0.0,
    )
    r_contrib = 1.0 if fsv.high_failure_risk else 0.0

    return round((e_contrib + a_contrib + r_contrib) / 3.0, 4)


#  Agent 

class LinguisticAuditor(BaseJuryAgent):
    agent_name: str = "LinguisticAuditor"

    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        cfg = get_settings()

        complexity_score, dims = compute_complexity_score(context.prompt)
        fired_dims = [d for d in dims if d.fired]

        #  Skip if prompt has no detectable complexity 
        if complexity_score < cfg.jury_linguistic_complexity_threshold:
            return self._skip(
                f"Prompt complexity score {complexity_score:.3f} is below "
                f"threshold {cfg.jury_linguistic_complexity_threshold:.3f}. "
                f"Failure cause is likely not prompt complexity."
            )

        # Compute failure signal strength 
        signal_strength = _failure_signal_strength(context.fsv)

        # ── Confidence = 40% complexity weight + 60% signal weight 
        raw_confidence = 0.40 * complexity_score + 0.60 * signal_strength

        #  Determine root cause 
        model_failed = (
            context.fsv.entropy_score >= cfg.jury_linguistic_entropy_threshold
            or context.fsv.agreement_score <= cfg.low_agreement_threshold
            or context.fsv.high_failure_risk
        )

        if model_failed:
            root_cause   = "PROMPT_COMPLEXITY_OOD"
            mitigation   = (
                "Decompose the prompt into simpler, atomic sub-questions. "
                "Use chain-of-thought prompting (add 'Think step by step'). "
                "If double-negation is present, rewrite in positive form. "
                "For multi-hop chains, resolve each hop in a separate call."
            )
            confidence   = raw_confidence
        else:
            # Complex prompt but model handled it — useful to report but lower severity
            root_cause   = "COMPLEX_BUT_STABLE"
            mitigation   = (
                "Prompt is structurally complex but the model responded stably. "
                "Monitor this query pattern — it may degrade under higher temperature "
                "or with a weaker model."
            )
            confidence   = raw_confidence * 0.4   

        # Build evidence dict
        evidence = {
            "complexity_score":    complexity_score,
            "signal_strength":     signal_strength,
            "dimensions_fired":    [d.name for d in fired_dims],
            "dimension_details": {
                d.name: d.detail for d in fired_dims if d.detail
            },
            "entropy_score":       context.fsv.entropy_score,
            "agreement_score":     context.fsv.agreement_score,
            "high_failure_risk":   context.fsv.high_failure_risk,
        }

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


# ── Module-level singleton ────────
linguistic_auditor = LinguisticAuditor()