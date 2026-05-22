"""
Backward-compatibility shim.

The adversarial detection code has been split into
engine/agents/adversarial/ (10 focused sub-modules).
This file re-exports the public interface so existing importers
(e.g. engine/agents/failure_agent.py) continue to work unchanged.
"""
from engine.agents.adversarial import AdversarialSpecialist, adversarial_specialist  # noqa: F401

__all__ = ["AdversarialSpecialist", "adversarial_specialist"]
