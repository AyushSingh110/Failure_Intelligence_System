"""
engine/agents/domain_critic.py — DiagnosticJury Agent 3

DomainCritic
============
Detects whether a model output is factually incorrect — hallucination,
knowledge boundary failure, or temporal cutoff mismatch.

Answers the question:
    "Did this model fail because it stated something factually wrong?"

How it works (no external API, no dummy data)
---------------------------------------------
The agent works purely on the signals already computed by Phase 1 and
the text available in DiagnosticContext. It does NOT call GPT, Claude,
or any external knowledge base. This makes it:
  - Zero-latency (no extra network calls)
  - Always available (works offline)
  - Deterministic (same input → same verdict)

It uses three independent internal detection layers:

Layer 1 — Contradiction Signal (FSV-based)
  If multiple model samples disagree with each other (low agreement,
  high entropy), the model is uncertain about the fact. Uncertain models
  hallucinate. This is the strongest signal.

Layer 2 — Self-Contradiction Scan (output-vs-output)
  Encodes primary and secondary model outputs with the shared
  SentenceEncoder and computes cosine similarity. If the same model
  gives semantically very different answers when sampled twice, it is
  contradicting itself — strong hallucination indicator.

Layer 3 — Hedge/Uncertainty Phrase Detection (regex)
  Detects epistemic hedge phrases in the model output:
  "I think", "I believe", "I'm not sure", "as far as I know",
  "it's possible that", "you might want to verify", etc.
  These are the model's own self-distrust signals — it's telling you
  it might be wrong.

Layer 4 — Temporal Cutoff Detection (regex on prompt)
  Detects prompts that ask about recent events, current prices,
  live scores, or "latest" information. Models have training cutoffs
  and will hallucinate recent facts confidently. This layer flags
  these queries as TEMPORAL_KNOWLEDGE_CUTOFF.

Root Cause Taxonomy
-------------------
  FACTUAL_HALLUCINATION      — model outputs contradictory/uncertain facts
  KNOWLEDGE_BOUNDARY_FAILURE — prompt asks about something the model
                                clearly does not know (high uncertainty
                                but not time-sensitive)
  TEMPORAL_KNOWLEDGE_CUTOFF  — prompt asks for current/recent information
                                that is beyond the model's training data
  DOMAIN_CORRECT             — all signals suggest the model is confident
                                and consistent (not a factual failure)

Confidence Score Derivation
----------------------------
The agent computes a weighted score from all layers that fired:

  contradiction_signal   weight 0.40  (strongest — FSV-derived)
  self_contradiction     weight 0.35  (strong — semantic divergence)
  hedge_density          weight 0.15  (moderate — model self-doubt)
  temporal_flag          weight 0.10  (categorical — cutoff risk)

  raw_confidence = weighted sum of fired layer scores (clipped to [0,1])

  Final confidence is also scaled by failure_signal_strength from Phase 1,
  so the verdict only fires with high confidence when BOTH the factual
  signal AND the Phase 1 instability signal agree.

Connecting to a Real Knowledge Base (future)
--------------------------------------------
When you want to wire in a real verifier (RAG, Wikipedia API, etc.),
the integration point is _run_external_verification() at the bottom.
It is called if all internal layers pass. Replace the stub return with
your real lookup — the rest of the agent stays unchanged.

Decision Tree
-------------
                ┌────────────────────────────────────────────┐
                │  Any of Layer 1/2/3/4 fired?               │
                └────────────────┬───────────────────────────┘
                                 │ YES
                         ┌───────▼────────┐
                         │ temporal flag? │
                         └───────┬────────┘
                          YES    │    NO
                          │      │
                          ▼      ▼
              TEMPORAL_  ┌──────────────────────┐
              KNOWLEDGE_ │ confidence ≥ 0.65?   │
              CUTOFF     └──────┬───────────────┘
                          YES   │   NO
                          │     │
                          ▼     ▼
              FACTUAL_  KNOWLEDGE_
              HALLUCIN- BOUNDARY_
              ATION     FAILURE

                 │ NO (nothing fired)
                 ▼
             DOMAIN_CORRECT (low confidence, skip if very low)
"""

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


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Hedge/Uncertainty phrase detector
# ══════════════════════════════════════════════════════════════════════════════

_RE_HEDGE = re.compile(
    r"\b("
    # direct epistemic hedges
    r"I\s+(?:think|believe|suppose|guess|assume|suspect|imagine|reckon)|"
    r"I(?:'m|\s+am)\s+not\s+(?:sure|certain|confident|aware)|"
    r"I(?:'m|\s+am)\s+(?:unsure|uncertain|not\s+100\s*%)|"
    # knowledge boundary phrases
    r"(?:as\s+far\s+as\s+I\s+(?:know|can\s+tell|recall|remember))|"
    r"(?:to\s+(?:my|the\s+best\s+of\s+my)\s+knowledge)|"
    r"(?:based\s+on\s+my\s+(?:training|knowledge|understanding))|"
    r"(?:my\s+(?:training|knowledge|information)\s+(?:may\s+be|is|could\s+be)\s+(?:outdated|limited|incomplete))|"
    # possibility/uncertainty markers
    r"(?:it(?:'s|\s+is)\s+(?:possible|likely|probable)\s+that)|"
    r"(?:(?:might|may|could)\s+(?:not\s+be\s+)?(?:accurate|correct|right|true))|"
    r"(?:(?:I\s+)?(?:cannot|can't|could\s+not)\s+(?:confirm|verify|guarantee))|"
    # deferral phrases
    r"(?:you(?:\s+might|\s+should|\s+may)?\s+want\s+to\s+(?:verify|check|confirm|look\s+up))|"
    r"(?:please\s+(?:verify|check|confirm|double.check))|"
    r"(?:I\s+(?:would\s+)?(?:recommend|suggest)\s+(?:checking|verifying|confirming))|"
    # explicit uncertainty
    r"(?:I(?:'m|\s+am)\s+not\s+(?:entirely|completely|fully|100\s*%)\s+(?:sure|certain|confident))|"
    r"(?:this\s+(?:may|might|could)\s+(?:be\s+)?(?:wrong|incorrect|inaccurate|outdated))|"
    r"(?:take\s+this\s+with\s+a\s+grain\s+of\s+salt)|"
    r"(?:don(?:'t|\s+not)\s+quote\s+me\s+on\s+(?:this|that))"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4 — Temporal/cutoff prompt detector
# ══════════════════════════════════════════════════════════════════════════════

_RE_TEMPORAL = re.compile(
    r"\b("
    # explicit recency requests
    r"(?:latest|most\s+recent|newest|current|up.to.date|up\s+to\s+date)\s+"
    r"(?:news|version|update|release|price|score|ranking|report|data|information|status)|"
    # right now / today
    r"(?:right\s+now|as\s+of\s+(?:today|now|this\s+(?:moment|year|month|week)))|"
    r"(?:today(?:'s)?\s+(?:price|score|news|weather|rate|stock|update))|"
    # live/real-time
    r"(?:live\s+(?:score|match|update|result|feed|data))|"
    r"(?:real.time\s+(?:data|price|update|score|feed))|"
    # specific recent year references
    r"(?:in\s+(?:2024|2025|2026))|"
    # "what is the current" — always temporal
    r"(?:what\s+is\s+the\s+current\s+(?:price|rate|status|version|score|ranking|ceo|president|pm|prime\s+minister))|"
    r"(?:who\s+is\s+(?:currently|now|the\s+current)\s+(?:the\s+)?(?:president|pm|prime\s+minister|ceo|head|leader|champion))|"
    # stock/finance recency
    r"(?:(?:stock|share|market|crypto|bitcoin|eth(?:ereum)?)\s+(?:price|value|rate)\s+(?:today|now|currently))|"
    # sports recency
    r"(?:(?:won|win|winner|champion)\s+(?:the\s+)?(?:latest|recent|last|this\s+year(?:'s)?)\s+)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


# ══════════════════════════════════════════════════════════════════════════════
# Detection result container
# ══════════════════════════════════════════════════════════════════════════════

class _LayerResult(NamedTuple):
    fired:        bool
    score:        float    # contribution to confidence [0, 1]
    evidence_key: str      # key to use in the evidence dict
    detail:       str      # human-readable detail string


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Contradiction signal (FSV-derived)
# ══════════════════════════════════════════════════════════════════════════════

def _run_contradiction_signal(context: DiagnosticContext) -> _LayerResult:
    """
    Computes a contradiction score from Phase 1 FSV signals.

    The FSV already tells us:
      - entropy_score    : how spread-out model outputs are (0=all same, 1=all different)
      - agreement_score  : fraction of outputs that match the plurality answer
      - high_failure_risk: combined Phase 1 flag

    We normalise these against their thresholds and combine.
    """
    cfg   = get_settings()
    fsv   = context.fsv

    # How far above the entropy threshold are we? (0 = at threshold, 1 = maxed)
    entropy_excess = max(
        (fsv.entropy_score - cfg.high_entropy_threshold) /
        max(1.0 - cfg.high_entropy_threshold, 1e-6),
        0.0,
    )

    # How far below the agreement threshold are we? (0 = at threshold, 1 = zero agreement)
    agreement_deficit = max(
        (cfg.low_agreement_threshold - fsv.agreement_score) /
        max(cfg.low_agreement_threshold, 1e-6),
        0.0,
    )

    risk_bonus = 0.15 if fsv.high_failure_risk else 0.0

    raw_score = min(
        0.5 * entropy_excess + 0.5 * agreement_deficit + risk_bonus,
        1.0,
    )
    fired = raw_score > 0.10  # needs at least slight instability

    detail = (
        f"entropy={fsv.entropy_score:.3f} "
        f"(threshold={cfg.high_entropy_threshold:.2f}), "
        f"agreement={fsv.agreement_score:.3f} "
        f"(threshold={cfg.low_agreement_threshold:.2f}), "
        f"high_failure_risk={fsv.high_failure_risk}"
    )

    return _LayerResult(
        fired=fired,
        score=round(raw_score, 4),
        evidence_key="contradiction_signal",
        detail=detail,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Self-contradiction scan (semantic divergence between outputs)
# ══════════════════════════════════════════════════════════════════════════════

def _run_self_contradiction(context: DiagnosticContext) -> _LayerResult:
    """
    Encodes primary and secondary outputs and measures cosine similarity.

    primary_output  = the main model answer being evaluated
    secondary_output = a second sample from the same model (or a reference)

    Low similarity between two answers to the same prompt = self-contradiction.

    Uses the shared SentenceEncoder (already loaded by AdversarialSpecialist
    or LinguisticAuditor earlier in the jury — no extra model loading).
    """
    encoder = get_encoder()

    # Graceful degradation if encoder is unavailable
    if not encoder.available:
        return _LayerResult(
            fired=False,
            score=0.0,
            evidence_key="self_contradiction",
            detail="SentenceEncoder unavailable — semantic check skipped (install sentence-transformers).",
        )

    primary   = context.primary_output.strip()
    secondary = context.secondary_output.strip()

    # If both outputs are identical or secondary is empty, skip this layer
    if not secondary or primary == secondary:
        return _LayerResult(
            fired=False,
            score=0.0,
            evidence_key="self_contradiction",
            detail="Primary and secondary outputs identical — no self-contradiction possible.",
        )

    try:
        vecs = encoder.encode_batch([primary, secondary])  # shape (2, 384)
        # L2-normalised → dot product = cosine similarity
        similarity = float(np.dot(vecs[0], vecs[1]))
        similarity = max(-1.0, min(1.0, similarity))  # numerical safety clip
    except Exception as exc:
        logger.warning("DomainCritic self-contradiction encoding failed: %s", exc)
        return _LayerResult(
            fired=False,
            score=0.0,
            evidence_key="self_contradiction",
            detail=f"Encoding error: {exc}",
        )

    # Low similarity = contradiction
    # similarity > 0.85 → strongly consistent → no contradiction
    # similarity < 0.50 → strongly contradictory → high score
    # We invert and normalise: score = max(0, (0.85 - similarity) / 0.85)
    CONSISTENCY_CEILING = 0.85
    raw_score = max(0.0, (CONSISTENCY_CEILING - similarity) / CONSISTENCY_CEILING)

    fired = similarity < 0.70   # below 0.70 cosine = meaningful divergence

    detail = (
        f"primary_vs_secondary cosine_similarity={similarity:.4f} "
        f"({'DIVERGENT' if fired else 'consistent'}). "
        f"primary[:80]={primary[:80]!r}  "
        f"secondary[:80]={secondary[:80]!r}"
    )

    return _LayerResult(
        fired=fired,
        score=round(raw_score, 4),
        evidence_key="self_contradiction",
        detail=detail,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Hedge/uncertainty phrase detection (model's own self-doubt)
# ══════════════════════════════════════════════════════════════════════════════

def _run_hedge_detection(context: DiagnosticContext) -> _LayerResult:
    """
    Counts hedge phrases across ALL sampled outputs, not just primary.
    A single hedge in one output contributes less than the same hedge
    appearing in multiple samples — convergent self-doubt is stronger evidence.
    """
    all_outputs = list(context.model_outputs) or [context.primary_output]

    total_hedges   = 0
    outputs_hedged = 0
    matched_phrases: list[str] = []

    for output in all_outputs:
        matches = _RE_HEDGE.findall(output)
        if matches:
            outputs_hedged += 1
            total_hedges   += len(matches)
            matched_phrases.extend(m[:60] for m in matches[:3])  # cap for evidence size

    n_outputs = max(len(all_outputs), 1)
    hedge_rate = outputs_hedged / n_outputs   # 0→1: fraction of outputs that hedged

    # Score: hedge rate × clipped hedge density
    hedge_density = min(total_hedges / (n_outputs * 2.0), 1.0)  # >2 hedges/output → 1.0
    raw_score     = round(0.6 * hedge_rate + 0.4 * hedge_density, 4)

    fired = outputs_hedged > 0

    detail = (
        f"hedge_phrases_found={total_hedges} across "
        f"{outputs_hedged}/{n_outputs} outputs. "
        f"examples={matched_phrases[:5]}"
    )

    return _LayerResult(
        fired=fired,
        score=raw_score,
        evidence_key="hedge_detection",
        detail=detail,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4 — Temporal/cutoff detection (prompt asks for post-cutoff information)
# ══════════════════════════════════════════════════════════════════════════════

def _run_temporal_detection(context: DiagnosticContext) -> _LayerResult:
    """
    Scans the PROMPT (not the output) for recency-demanding language.
    If the user is asking for current/live information, the model cannot
    answer correctly from training data alone — temporal cutoff failure.
    """
    matches = _RE_TEMPORAL.findall(context.prompt)
    fired   = bool(matches)
    score   = min(len(matches) * 0.35, 1.0)  # each match adds 0.35, capped

    detail = (
        f"temporal_phrases_in_prompt={len(matches)}: {matches[:5]}"
        if fired else
        "No temporal/recency phrases detected in prompt."
    )

    return _LayerResult(
        fired=fired,
        score=round(score, 4),
        evidence_key="temporal_detection",
        detail=detail,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Future integration point — external knowledge base
# ══════════════════════════════════════════════════════════════════════════════

def _run_external_verification(context: DiagnosticContext) -> _LayerResult:
    """
    FUTURE INTEGRATION POINT — wire in a real verifier here.

    When you are ready to connect to a real knowledge source, replace
    the return below with a call to your chosen verifier:

    Option A — RAG (recommended for your project since RAGService exists):
        from backend.services.rag_service import RAGService
        rag = RAGService()
        ground_truth = rag.retrieve_facts(context.prompt)
        similarity   = compute_similarity(context.primary_output, ground_truth)
        fired        = similarity < 0.60
        ...

    Option B — Wikipedia API (for general knowledge):
        import wikipedia
        page    = wikipedia.summary(extract_topic(context.prompt), sentences=3)
        vec_out = encoder.encode(context.primary_output)
        vec_gt  = encoder.encode(page)
        cosine  = float(np.dot(vec_out, vec_gt))
        fired   = cosine < 0.55
        ...

    Option C — Ollama / local LLM verifier:
        import requests
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": f"Is this factually correct? '{context.primary_output}' Answer only yes or no."
        })
        answer = r.json()["response"].strip().lower()
        fired  = answer.startswith("no")
        ...

    Right now this returns no signal (fired=False) so it never affects
    the verdict — the internal layers handle everything.
    """
    return _LayerResult(
        fired=False,
        score=0.0,
        evidence_key="external_verification",
        detail="External verifier not yet connected. See _run_external_verification() to wire in RAG/Wikipedia/LLM.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Failure signal strength helper (mirrors LinguisticAuditor's version)
# ══════════════════════════════════════════════════════════════════════════════

def _failure_signal_strength(context: DiagnosticContext) -> float:
    """
    Combines Phase 1 FSV into a single [0, 1] instability score.
    Used to scale the final confidence so the verdict only fires
    with high confidence when Phase 1 also sees instability.
    """
    cfg = get_settings()
    fsv = context.fsv

    e = min(fsv.entropy_score   / max(cfg.high_entropy_threshold,  1e-6), 1.0)
    a = max(1.0 - fsv.agreement_score / max(cfg.low_agreement_threshold, 1e-6), 0.0)
    r = 1.0 if fsv.high_failure_risk else 0.0

    return round((e + a + r) / 3.0, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class DomainCritic(BaseJuryAgent):
    """
    Agent 3 — Domain Critic

    Detects factual hallucination, knowledge boundary failures,
    and temporal cutoff mismatches using four internal detection
    layers — no external API calls required.

    Layer weights:
      contradiction_signal  0.40  (FSV-derived instability)
      self_contradiction    0.35  (semantic divergence between outputs)
      hedge_detection       0.15  (model's own uncertainty phrases)
      temporal_detection    0.10  (prompt asks for post-cutoff info)
    """

    agent_name: str = "DomainCritic"

    # Layer weights — must sum to 1.0
    _WEIGHTS = {
        "contradiction_signal": 0.40,
        "self_contradiction":   0.35,
        "hedge_detection":      0.15,
        "temporal_detection":   0.10,
    }

    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        cfg = get_settings()

        # ── Run all four detection layers ──────────────────────────────
        layer1 = _run_contradiction_signal(context)
        layer2 = _run_self_contradiction(context)
        layer3 = _run_hedge_detection(context)
        layer4 = _run_temporal_detection(context)

        layers = [layer1, layer2, layer3, layer4]
        fired_layers = [l for l in layers if l.fired]

        # ── Skip if absolutely nothing fired ──────────────────────────
        if not fired_layers:
            return self._skip(
                "All four DomainCritic layers returned no signal. "
                "Model outputs are consistent, certain, and non-temporal. "
                "Factual hallucination is not the likely failure cause."
            )

        # ── Compute weighted confidence ────────────────────────────────
        raw_confidence = sum(
            l.score * self._WEIGHTS[l.evidence_key]
            for l in layers
            if l.evidence_key in self._WEIGHTS
        )

        # Scale by Phase 1 failure signal strength so we don't fire
        # with high confidence when Phase 1 says the model was stable
        fss              = _failure_signal_strength(context)
        scaled_confidence = round(
            0.55 * raw_confidence + 0.45 * (raw_confidence * fss),
            4,
        )

        # ── Determine root cause ───────────────────────────────────────
        temporal_fired      = layer4.fired
        high_contradiction  = (layer1.score > 0.40 or layer2.score > 0.45)
        moderate_signal     = scaled_confidence >= cfg.jury_domain_confidence_threshold

        if temporal_fired:
            root_cause  = "TEMPORAL_KNOWLEDGE_CUTOFF"
            mitigation  = (
                "The prompt requests information beyond the model's training cutoff. "
                "Connect a web search tool or real-time data feed to handle recency-sensitive queries. "
                "Add a system prompt note about the knowledge cutoff date so the model discloses it. "
                "Consider routing 'current/latest/live' queries to a RAG pipeline backed by a live index."
            )

        elif high_contradiction and moderate_signal:
            root_cause  = "FACTUAL_HALLUCINATION"
            mitigation  = (
                "High output inconsistency detected — model is generating conflicting facts. "
                "Lower the temperature to reduce sampling variance. "
                "Add a self-consistency check: sample 3–5 outputs and take the plurality answer. "
                "Consider grounding the response with a RAG retrieval step before generation. "
                "For critical factual queries, add a post-generation fact-check pass."
            )

        elif moderate_signal:
            root_cause  = "KNOWLEDGE_BOUNDARY_FAILURE"
            mitigation  = (
                "Model shows uncertainty signals but not full hallucination. "
                "The query may be at the boundary of the model's training data. "
                "Add retrieval augmentation (RAG) for this domain. "
                "Prompt the model to say 'I don't know' explicitly when uncertain, "
                "rather than generating plausible-sounding but unverified content."
            )

        else:
            # Signals fired but confidence is too low to make a strong call
            root_cause  = "DOMAIN_CORRECT"
            mitigation  = (
                "Minor factual uncertainty signals detected but below confidence threshold. "
                "Model outputs appear broadly consistent. Monitor this query pattern."
            )

        # ── If DOMAIN_CORRECT and confidence is very low, skip ─────────
        if root_cause == "DOMAIN_CORRECT" and scaled_confidence < 0.20:
            return self._skip(
                f"DomainCritic confidence {scaled_confidence:.3f} is below minimum "
                f"threshold — outputs are broadly consistent. No factual failure detected."
            )

        # ── Build evidence dict ────────────────────────────────────────
        evidence = {
            "root_cause_selected":      root_cause,
            "raw_confidence":           round(raw_confidence, 4),
            "failure_signal_strength":  fss,
            "scaled_confidence":        scaled_confidence,
            "layers_fired":             [l.evidence_key for l in fired_layers],
            "layer_scores": {
                l.evidence_key: l.score for l in layers
            },
            "layer_details": {
                l.evidence_key: l.detail
                for l in layers
                if l.fired or l.evidence_key == "external_verification"
            },
            "fsv_snapshot": {
                "entropy_score":         context.fsv.entropy_score,
                "agreement_score":       context.fsv.agreement_score,
                "fsd_score":             context.fsv.fsd_score,
                "ensemble_disagreement": context.fsv.ensemble_disagreement,
                "high_failure_risk":     context.fsv.high_failure_risk,
            },
            "n_model_outputs":          len(context.model_outputs),
        }

        logger.debug(
            "DomainCritic | root_cause=%s | confidence=%.3f | "
            "layers_fired=%s | fss=%.3f",
            root_cause, scaled_confidence,
            [l.evidence_key for l in fired_layers], fss,
        )

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(scaled_confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


# ── Module-level singleton ─────────────────────────────────────────────────
domain_critic = DomainCritic()