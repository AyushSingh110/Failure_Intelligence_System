"""
Centralized exception hierarchy for the Failure Intelligence Engine.

All custom exceptions inherit from FIEException so callers can catch a single
base class while still handling specific failure modes at the point of recovery.
"""
from __future__ import annotations


class FIEException(Exception):
    """Base class for all FIE domain exceptions."""

    def __init__(self, message: str, code: str = "FIE_ERROR") -> None:
        super().__init__(message)
        self.code = code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={str(self)!r})"


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(FIEException):
    """Required configuration value is missing or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


# ── Model loading ─────────────────────────────────────────────────────────────

class ModelUnavailableError(FIEException):
    """An ML model (XGBoost, PAIR classifier, sentence encoder) cannot be loaded."""

    def __init__(self, model_name: str, reason: str = "") -> None:
        msg = f"Model '{model_name}' unavailable"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, code="MODEL_UNAVAILABLE")
        self.model_name = model_name


# ── Shadow models ─────────────────────────────────────────────────────────────

class ShadowModelError(FIEException):
    """All shadow model calls failed — no ensemble comparison possible."""

    def __init__(self, reason: str = "") -> None:
        msg = "All shadow model calls failed"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, code="SHADOW_MODEL_ERROR")


# ── Adversarial detection ─────────────────────────────────────────────────────

class AdversarialBlockedError(FIEException):
    """Prompt was blocked by the adversarial guardrail."""

    def __init__(self, attack_type: str, confidence: float) -> None:
        super().__init__(
            f"Prompt blocked: {attack_type} (confidence={confidence:.3f})",
            code="ADVERSARIAL_BLOCKED",
        )
        self.attack_type = attack_type
        self.confidence  = confidence


# ── Ground truth verification ─────────────────────────────────────────────────

class VerificationError(FIEException):
    """A ground truth verifier (Wikidata, Serper, self-consistency) failed."""

    def __init__(self, verifier: str, reason: str) -> None:
        super().__init__(f"{verifier} verification failed: {reason}", code="VERIFICATION_ERROR")
        self.verifier = verifier


# ── Escalation ────────────────────────────────────────────────────────────────

class EscalationRequired(FIEException):
    """Auto-correction is not possible; human review is required."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason, code="ESCALATION_REQUIRED")
        self.reason = reason


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineError(FIEException):
    """A LangGraph pipeline node encountered an unrecoverable error."""

    def __init__(self, node: str, reason: str) -> None:
        super().__init__(f"Pipeline node '{node}' failed: {reason}", code="PIPELINE_ERROR")
        self.node   = node
        self.reason = reason


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(FIEException):
    """MongoDB or vault storage operation failed."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(f"Storage '{operation}' failed: {reason}", code="STORAGE_ERROR")
        self.operation = operation
