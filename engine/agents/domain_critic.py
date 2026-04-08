#domain_critic.py
from __future__ import annotations

import re
import logging
from typing import NamedTuple

import numpy as np

from app.schemas import AgentVerdict
from config import get_settings
from engine.agents.base_agent import BaseJuryAgent, DiagnosticContext
from engine.encoder import get_encoder

logger   = logging.getLogger(__name__)
settings = get_settings()

_RE_HEDGE = re.compile(
    r"\b("
    r"I\s+(?:think|believe|suppose|guess|assume|suspect|imagine|reckon)|"
    r"I(?:'m|\s+am)\s+not\s+(?:sure|certain|confident|aware)|"
    r"I(?:'m|\s+am)\s+(?:unsure|uncertain|not\s+100\s*%)|"
    r"(?:as\s+far\s+as\s+I\s+(?:know|can\s+tell|recall|remember))|"
    r"(?:to\s+(?:my|the\s+best\s+of\s+my)\s+knowledge)|"
    r"(?:based\s+on\s+my\s+(?:training|knowledge|understanding))|"
    r"(?:my\s+(?:training|knowledge|information)\s+(?:may\s+be|is|could\s+be)\s+(?:outdated|limited|incomplete))|"
    r"(?:it(?:'s|\s+is)\s+(?:possible|likely|probable)\s+that)|"
    r"(?:(?:might|may|could)\s+(?:not\s+be\s+)?(?:accurate|correct|right|true))|"
    r"(?:(?:I\s+)?(?:cannot|can't|could\s+not)\s+(?:confirm|verify|guarantee))|"
    r"(?:you(?:\s+might|\s+should|\s+may)?\s+want\s+to\s+(?:verify|check|confirm|look\s+up))|"
    r"(?:please\s+(?:verify|check|confirm|double.check))|"
    r"(?:I\s+(?:would\s+)?(?:recommend|suggest)\s+(?:checking|verifying|confirming))|"
    r"(?:I(?:'m|\s+am)\s+not\s+(?:entirely|completely|fully|100\s*%)\s+(?:sure|certain|confident))|"
    r"(?:this\s+(?:may|might|could)\s+(?:be\s+)?(?:wrong|incorrect|inaccurate|outdated))|"
    r"(?:take\s+this\s+with\s+a\s+grain\s+of\s+salt)|"
    r"(?:don(?:'t|\s+not)\s+quote\s+me\s+on\s+(?:this|that))"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

# Permanent-fact patterns — questions whose answers NEVER change over time.
# If a prompt or answer matches these, temporal routing is suppressed.
_RE_PERMANENT_FACT = re.compile(
    r"\b("
    # Chemical / element questions
    r"(?:chemical\s+(?:symbol|formula|name|element)\s+(?:for|of))|"
    r"(?:what\s+(?:element|symbol)\s+is)|"
    r"(?:atomic\s+(?:number|mass|weight)\s+(?:of|for))|"
    r"(?:molecular\s+(?:formula|weight|mass)\s+(?:of|for))|"
    # Pure math
    r"(?:what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+)|"
    r"(?:square\s+root\s+of)|"
    r"(?:how\s+many\s+(?:sides|angles|vertices|faces)\s+(?:does|in)\s+a)|"
    # Well-known physical constants
    r"(?:speed\s+of\s+light)|"
    r"(?:gravitational\s+constant)|"
    r"(?:boiling\s+point\s+of\s+water)|"
    r"(?:freezing\s+point\s+of\s+water)|"
    r"(?:melting\s+point\s+of)|"
    # Historical fixed facts
    r"(?:when\s+was\s+(?:the\s+)?(?:earth|universe|solar\s+system)\s+formed)|"
    r"(?:how\s+many\s+(?:planets|moons|continents|oceans)\s+(?:are\s+there|does\s+earth\s+have))"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

_RE_TEMPORAL = re.compile(
    r"\b("
    r"(?:latest|most\s+recent|newest|current|up.to.date|up\s+to\s+date)\s+"
    r"(?:news|version|update|release|price|score|ranking|report|data|information|status)|"
    r"(?:right\s+now|as\s+of\s+(?:today|now|this\s+(?:moment|year|month|week)))|"
    r"(?:today(?:'s)?\s+(?:price|score|news|weather|rate|stock|update))|"
    r"(?:today(?:'s)?)|"
    r"(?:live\s+(?:score|match|update|result|feed|data))|"
    r"(?:real.time\s+(?:data|price|update|score|feed))|"
    r"(?:in\s+(?:2024|2025|2026))|"
    r"(?:what\s+is\s+the\s+current\s+(?:price|rate|status|version|score|ranking|ceo|president|pm|prime\s+minister))|"
    r"(?:what\s+is\s+the\s+(?:latest|newest|current)\s+(?:iphone|android|python|version|model|release))|"
    r"(?:who\s+is\s+(?:currently|now|the\s+current)\s+(?:the\s+)?(?:president|pm|prime\s+minister|ceo|head|leader|champion))|"
    r"(?:(?:stock|share|market|crypto|bitcoin|eth(?:ereum)?)\s+(?:price|value|rate)\s+(?:today|now|currently))|"
    r"(?:(?:live|current)\s+(?:score|weather|price|version|result|status))|"
    r"(?:(?:won|win|winner|champion)\s+(?:the\s+)?(?:latest|recent|last|this\s+year(?:'s)?)\s+)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


class _LayerResult(NamedTuple):
    fired:        bool
    score:        float
    evidence_key: str
    detail:       str


def _run_contradiction_signal(context: DiagnosticContext) -> _LayerResult:
    cfg = get_settings()
    fsv = context.fsv

    entropy_excess = max(
        (fsv.entropy_score - cfg.high_entropy_threshold) /
        max(1.0 - cfg.high_entropy_threshold, 1e-6),
        0.0,
    )
    agreement_deficit = max(
        (cfg.low_agreement_threshold - fsv.agreement_score) /
        max(cfg.low_agreement_threshold, 1e-6),
        0.0,
    )
    risk_bonus = 0.15 if fsv.high_failure_risk else 0.0
    raw_score  = min(0.5 * entropy_excess + 0.5 * agreement_deficit + risk_bonus, 1.0)
    fired      = raw_score > 0.05   # lowered from 0.10 to catch subtle outlier cases

    detail = (
        f"entropy={fsv.entropy_score:.3f} (threshold={cfg.high_entropy_threshold:.2f}), "
        f"agreement={fsv.agreement_score:.3f} (threshold={cfg.low_agreement_threshold:.2f}), "
        f"high_failure_risk={fsv.high_failure_risk}"
    )
    return _LayerResult(fired=fired, score=round(raw_score, 4),
                        evidence_key="contradiction_signal", detail=detail)


def _run_self_contradiction(context: DiagnosticContext) -> _LayerResult:
    encoder = get_encoder()

    if not encoder.available:
        return _LayerResult(False, 0.0, "self_contradiction",
                            "SentenceEncoder unavailable.")

    primary   = context.primary_output.strip()
    secondary = context.secondary_output.strip()

    if not secondary or primary == secondary:
        return _LayerResult(False, 0.0, "self_contradiction",
                            "Primary and secondary outputs identical.")

    try:
        vecs       = encoder.encode_batch([primary, secondary])
        similarity = float(np.dot(vecs[0], vecs[1]))
        similarity = max(-1.0, min(1.0, similarity))
    except Exception as exc:
        return _LayerResult(False, 0.0, "self_contradiction", f"Encoding error: {exc}")

    CONSISTENCY_CEILING = 0.85
    raw_score = max(0.0, (CONSISTENCY_CEILING - similarity) / CONSISTENCY_CEILING)
    fired     = similarity < 0.70

    detail = (
        f"primary_vs_secondary cosine={similarity:.4f} "
        f"({'DIVERGENT' if fired else 'consistent'}). "
        f"primary[:80]={primary[:80]!r} secondary[:80]={secondary[:80]!r}"
    )
    return _LayerResult(fired=fired, score=round(raw_score, 4),
                        evidence_key="self_contradiction", detail=detail)


def _run_hedge_detection(context: DiagnosticContext) -> _LayerResult:
    all_outputs    = list(context.model_outputs) or [context.primary_output]
    total_hedges   = 0
    outputs_hedged = 0
    matched_phrases: list[str] = []

    for output in all_outputs:
        matches = _RE_HEDGE.findall(output)
        if matches:
            outputs_hedged += 1
            total_hedges   += len(matches)
            matched_phrases.extend(m[:60] for m in matches[:3])

    n_outputs     = max(len(all_outputs), 1)
    hedge_rate    = outputs_hedged / n_outputs
    hedge_density = min(total_hedges / (n_outputs * 2.0), 1.0)
    raw_score     = round(0.6 * hedge_rate + 0.4 * hedge_density, 4)

    detail = (
        f"hedge_phrases_found={total_hedges} across {outputs_hedged}/{n_outputs} outputs. "
        f"examples={matched_phrases[:5]}"
    )
    return _LayerResult(fired=outputs_hedged > 0, score=raw_score,
                        evidence_key="hedge_detection", detail=detail)


def _run_temporal_detection(context: DiagnosticContext) -> _LayerResult:
    # Suppress temporal detection for prompts about permanently fixed facts
    # (chemistry, math, physics constants, geography fundamentals).
    # These facts don't change with time so they should never be routed to
    # real-time search or escalated as TEMPORAL_KNOWLEDGE_CUTOFF.
    if _RE_PERMANENT_FACT.search(context.prompt):
        return _LayerResult(
            fired=False,
            score=0.0,
            evidence_key="temporal_detection",
            detail=(
                "Permanent-fact pattern matched in prompt — temporal routing suppressed. "
                "Chemical, mathematical, and physical constant facts do not change over time."
            ),
        )

    matches = _RE_TEMPORAL.findall(context.prompt)
    fired   = bool(matches)
    score   = min(len(matches) * 0.35, 1.0)
    detail  = (
        f"temporal_phrases_in_prompt={len(matches)}: {matches[:5]}"
        if fired else "No temporal/recency phrases detected in prompt."
    )
    return _LayerResult(fired=fired, score=round(score, 4),
                        evidence_key="temporal_detection", detail=detail)


def _run_external_verification(context: DiagnosticContext) -> _LayerResult:
    if re.fullmatch(r"[\s\d\+\-\*\/\(\)=\?\.]+", context.prompt):
        return _LayerResult(
            False,
            0.0,
            "external_verification",
            "Skipped external verification for arithmetic-style prompt.",
        )

    try:
        from engine.rag_grounder import ground_with_wikipedia, compare_with_ground_truth

        grounding = ground_with_wikipedia(context.prompt, context.primary_output)
        if not grounding.success:
            return _LayerResult(
                False,
                0.0,
                "external_verification",
                grounding.error or "External verifier unavailable.",
            )

        check = compare_with_ground_truth(context.primary_output, grounding.grounded_answer)
        detail = (
            f"{check.reason}. grounded_answer={grounding.grounded_answer[:160]!r} "
            f"source={grounding.source}"
        )
        return _LayerResult(
            fired=not check.matches,
            score=round(check.confidence, 4),
            evidence_key="external_verification",
            detail=detail,
        )
    except Exception as exc:
        return _LayerResult(
            False,
            0.0,
            "external_verification",
            f"External verification error: {exc}",
        )


def _failure_signal_strength(context: DiagnosticContext) -> float:
    cfg = get_settings()
    fsv = context.fsv
    e   = min(fsv.entropy_score / max(cfg.high_entropy_threshold, 1e-6), 1.0)
    a   = max(1.0 - fsv.agreement_score / max(cfg.low_agreement_threshold, 1e-6), 0.0)
    r   = 1.0 if fsv.high_failure_risk else 0.0
    return round((e + a + r) / 3.0, 4)


class DomainCritic(BaseJuryAgent):
    """
    Agent 3 — Domain Critic
    Detects factual hallucination, knowledge boundary failures,
    and temporal cutoff mismatches.
    Layer weights: contradiction 0.40, self_contradiction 0.35,
                   hedge 0.15, temporal 0.10
    """

    agent_name: str = "DomainCritic"

    _WEIGHTS = {
        "contradiction_signal": 0.40,
        "self_contradiction":   0.35,
        "hedge_detection":      0.15,
        "temporal_detection":   0.10,
        "external_verification": 0.45,
    }

    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        cfg = get_settings()

        layer1 = _run_contradiction_signal(context)
        layer2 = _run_self_contradiction(context)
        layer3 = _run_hedge_detection(context)
        layer4 = _run_temporal_detection(context)
        layer5 = _run_external_verification(context)

        layers       = [layer1, layer2, layer3, layer4, layer5]
        fired_layers = [l for l in layers if l.fired]

        if not fired_layers:
            return self._skip(
                "All four DomainCritic layers returned no signal. "
                "Model outputs are consistent, certain, and non-temporal. "
                "Factual hallucination is not the likely failure cause."
            )

        raw_confidence = sum(
            l.score * self._WEIGHTS[l.evidence_key]
            for l in layers
            if l.evidence_key in self._WEIGHTS
        )

        fss               = _failure_signal_strength(context)
        scaled_confidence = round(0.55 * raw_confidence + 0.45 * (raw_confidence * fss), 4)

        temporal_fired     = layer4.fired
        external_fired     = layer5.fired
        high_contradiction = (layer1.score > 0.30 or layer2.score > 0.40)
        moderate_signal    = scaled_confidence >= cfg.jury_domain_confidence_threshold

        if temporal_fired:
            root_cause = "TEMPORAL_KNOWLEDGE_CUTOFF"
            mitigation = (
                "The prompt requests information beyond the model's training cutoff. "
                "Connect a web search tool or real-time data feed. "
                "Consider routing 'current/latest/live' queries to a RAG pipeline."
            )
        elif (external_fired or high_contradiction) and moderate_signal:
            root_cause = "FACTUAL_HALLUCINATION"
            mitigation = (
                "Ground-truth verification detected a factual mismatch or high output inconsistency. "
                "Lower the temperature to reduce sampling variance. "
                "Add a self-consistency check: sample 3-5 outputs and take the plurality answer. "
                "Consider grounding the response with a RAG retrieval step before generation."
            )
        elif moderate_signal:
            root_cause = "KNOWLEDGE_BOUNDARY_FAILURE"
            mitigation = (
                "Model shows uncertainty signals but not full hallucination. "
                "The query may be at the boundary of the model's training data. "
                "Add retrieval augmentation (RAG) for this domain."
            )
        else:
            root_cause = "DOMAIN_CORRECT"
            mitigation = "Minor signals below confidence threshold. Monitor this query pattern."

        if root_cause == "DOMAIN_CORRECT" and scaled_confidence < 0.08:
            return self._skip(
                f"DomainCritic confidence {scaled_confidence:.3f} is below minimum "
                f"threshold — outputs are broadly consistent. No factual failure detected."
            )

        evidence = {
            "root_cause_selected":     root_cause,
            "raw_confidence":          round(raw_confidence, 4),
            "failure_signal_strength": fss,
            "scaled_confidence":       scaled_confidence,
            "layers_fired":            [l.evidence_key for l in fired_layers],
            "layer_scores":            {l.evidence_key: l.score for l in layers},
            "layer_details":           {l.evidence_key: l.detail for l in layers if l.fired},
            "fsv_snapshot": {
                "entropy_score":         context.fsv.entropy_score,
                "agreement_score":       context.fsv.agreement_score,
                "fsd_score":             context.fsv.fsd_score,
                "ensemble_disagreement": context.fsv.ensemble_disagreement,
                "high_failure_risk":     context.fsv.high_failure_risk,
            },
            "n_model_outputs": len(context.model_outputs),
        }

        logger.debug(
            "DomainCritic | root_cause=%s | confidence=%.3f | layers_fired=%s | fss=%.3f",
            root_cause, scaled_confidence,
            [l.evidence_key for l in fired_layers], fss,
        )

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(scaled_confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


domain_critic = DomainCritic()
