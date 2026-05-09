"""
LangGraph-based detection pipeline for the Failure Intelligence Engine.

Flow:
  START
    │
    ▼
  prompt_guard          ← regex + many-shot detection (fast, no LLM call)
    │
    ├─ score ≥ 0.75 ──► block         ← immediate reject, log attack type
    │
    └─ score < 0.75 ──► signal_extract ← FSV: entropy, agreement, ensemble
                            │
                            ▼
                        jury_deliberate  ← 3-agent DiagnosticJury
                            │
                            ▼
                          END

Each node receives the full PipelineState and returns only the keys it changes.
The graph merges updates automatically.
"""
from __future__ import annotations

import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from engine.prompt_guard import score_prompt_attack
from engine.agents.failure_agent import failure_agent

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── inputs ──────────────────────────────────────────────
    prompt:         str
    model_outputs:  list[str]
    primary_output: str

    # ── guard layer ──────────────────────────────────────────
    guard_score:      float
    guard_root_cause: Optional[str]
    guard_blocked:    bool

    # ── signal extraction (phase 1) ──────────────────────────
    failure_signal: Optional[dict]
    archetype:      Optional[str]

    # ── jury deliberation (phase 3) ──────────────────────────
    jury_verdict:    Optional[dict]
    is_adversarial:  bool
    failure_summary: str
    confidence:      float


# ── Node: prompt guard ────────────────────────────────────────────────────────

def prompt_guard_node(state: PipelineState) -> dict:
    """Layer 1-9: fast regex-based adversarial detection (no LLM call)."""
    sig = score_prompt_attack(state["prompt"])
    logger.debug("prompt_guard score=%.3f root_cause=%s", sig.score, sig.root_cause)
    return {
        "guard_score":      sig.score,
        "guard_root_cause": sig.root_cause,
        "guard_blocked":    sig.score >= 0.75,
    }


# ── Routing: should we block or continue? ─────────────────────────────────────

def route_after_guard(state: PipelineState) -> str:
    return "block" if state["guard_blocked"] else "signal_extract"


# ── Node: block ───────────────────────────────────────────────────────────────

def block_node(state: PipelineState) -> dict:
    """Terminate pipeline immediately — adversarial prompt detected by guardrail."""
    cause = state["guard_root_cause"] or "UNKNOWN_ATTACK"
    score = state["guard_score"]
    logger.warning("BLOCKED prompt | cause=%s score=%.3f", cause, score)
    return {
        "is_adversarial":  True,
        "archetype":       "MODEL_BLIND_SPOT",
        "failure_summary": f"Blocked by guardrail: {cause} (score={score:.3f})",
        "confidence":      score,
    }


# ── Node: signal extraction ───────────────────────────────────────────────────

def signal_extract_node(state: PipelineState) -> dict:
    """Phase 1: build FailureSignalVector and label archetype."""
    try:
        result = failure_agent.run(
            model_outputs=state["model_outputs"],
            primary_output=state["primary_output"] or None,
        )
        return {
            "failure_signal": result.get("failure_signal_vector"),
            "archetype":      result.get("archetype"),
        }
    except Exception as exc:
        logger.error("signal_extract_node failed: %s", exc, exc_info=True)
        return {"failure_signal": None, "archetype": "UNKNOWN"}


# ── Node: jury deliberation ───────────────────────────────────────────────────

def jury_deliberate_node(state: PipelineState) -> dict:
    """Phase 3: run DiagnosticJury across adversarial + linguistic + domain agents."""
    from app.schemas import DiagnosticRequest

    try:
        request = DiagnosticRequest(
            prompt=state["prompt"],
            model_outputs=state["model_outputs"],
        )
        response    = failure_agent.run_diagnostic(request)
        jury        = response.jury
        jury_dict   = jury.model_dump() if hasattr(jury, "model_dump") else dict(jury)

        return {
            "jury_verdict":    jury_dict,
            "is_adversarial":  jury.is_adversarial,
            "failure_summary": jury.failure_summary,
            "confidence":      jury.jury_confidence,
            "archetype":       str(response.archetype),
        }
    except Exception as exc:
        logger.error("jury_deliberate_node failed: %s", exc, exc_info=True)
        return {
            "jury_verdict":    None,
            "is_adversarial":  False,
            "failure_summary": "Jury error — manual review recommended.",
            "confidence":      0.0,
        }


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("prompt_guard",    prompt_guard_node)
    g.add_node("block",           block_node)
    g.add_node("signal_extract",  signal_extract_node)
    g.add_node("jury_deliberate", jury_deliberate_node)

    g.set_entry_point("prompt_guard")

    g.add_conditional_edges(
        "prompt_guard",
        route_after_guard,
        {"block": "block", "signal_extract": "signal_extract"},
    )

    g.add_edge("block",          END)
    g.add_edge("signal_extract", "jury_deliberate")
    g.add_edge("jury_deliberate", END)

    return g.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    prompt:         str,
    model_outputs:  list[str],
    primary_output: str = "",
) -> dict:
    """
    Run the full FIE detection pipeline via LangGraph.

    Returns the final PipelineState dict with all fields populated.
    Safe to call concurrently — the graph is stateless between invocations.
    """
    initial: PipelineState = {
        "prompt":          prompt,
        "model_outputs":   model_outputs,
        "primary_output":  primary_output or (model_outputs[0] if model_outputs else ""),
        "guard_score":     0.0,
        "guard_root_cause": None,
        "guard_blocked":   False,
        "failure_signal":  None,
        "archetype":       None,
        "jury_verdict":    None,
        "is_adversarial":  False,
        "failure_summary": "",
        "confidence":      0.0,
    }

    result = _get_graph().invoke(initial)
    logger.info(
        "pipeline done | archetype=%s adversarial=%s confidence=%.3f",
        result.get("archetype"), result.get("is_adversarial"), result.get("confidence", 0),
    )
    return result
