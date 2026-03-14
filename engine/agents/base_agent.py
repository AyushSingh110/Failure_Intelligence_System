from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.schemas import AgentVerdict, FailureSignalVector


# ── Context container ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiagnosticContext:
    """
    Read-only context bundle passed to every agent.
    Frozen to prevent agents from mutating shared state.

    Attributes
    ----------
    prompt            : Original user-facing input text.
    primary_output    : Primary model response (model under test).
    secondary_output  : Secondary model response (reference/ensemble model).
    model_outputs     : All sampled outputs used for consistency analysis.
    fsv               : FailureSignalVector from Phase 1.
    latency_ms        : Optional. Inference latency in milliseconds.
    """
    prompt:           str
    primary_output:   str
    secondary_output: str
    model_outputs:    tuple[str, ...]    # tuple so dataclass remains frozen/hashable
    fsv:              FailureSignalVector
    latency_ms:       Optional[float] = None

    @classmethod
    def build(
        cls,
        prompt:           str,
        primary_output:   str,
        secondary_output: str,
        model_outputs:    list[str],
        fsv:              FailureSignalVector,
        latency_ms:       Optional[float] = None,
    ) -> "DiagnosticContext":
        return cls(
            prompt=prompt,
            primary_output=primary_output,
            secondary_output=secondary_output,
            model_outputs=tuple(model_outputs),
            fsv=fsv,
            latency_ms=latency_ms,
        )


# ── Base agent ─────────────────────────────────────────────────────────────

class BaseJuryAgent(ABC):
    """
    Abstract base class for all DiagnosticJury agents.

    Subclass contract:
      - Set agent_name as a class-level constant string.
      - Implement analyze(context) returning an AgentVerdict.
      - Never raise exceptions; always return a verdict (possibly skipped).
    """

    #: Override in every subclass — used for logging and verdict attribution.
    agent_name: str = "UNNAMED_AGENT"

    @abstractmethod
    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        """
        Core analysis method.

        Parameters
        ----------
        context : DiagnosticContext
            All signals and raw data for this inference event.

        Returns
        -------
        AgentVerdict
            Structured diagnosis. Set skipped=True if analysis is not applicable.
        """
        ...

    # ── Helpers available to all subclasses ───────────────────────────

    def _skip(self, reason: str) -> AgentVerdict:
        """Convenience: return a skipped verdict with a clean reason string."""
        return AgentVerdict(
            agent_name=self.agent_name,
            root_cause="NOT_APPLICABLE",
            confidence_score=0.0,
            mitigation_strategy="",
            skipped=True,
            skip_reason=reason,
        )

    def _verdict(
        self,
        root_cause:          str,
        confidence_score:    float,
        mitigation_strategy: str,
        evidence:            Optional[dict] = None,
    ) -> AgentVerdict:
        """Convenience: return a full verdict with this agent's name pre-filled."""
        return AgentVerdict(
            agent_name=self.agent_name,
            root_cause=root_cause,
            confidence_score=round(float(confidence_score), 4),
            mitigation_strategy=mitigation_strategy,
            evidence=evidence or {},
            skipped=False,
        )