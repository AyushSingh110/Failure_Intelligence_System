"""
fie.correction — how the SDK treats a correction the FIE server suggests.

The SDK never replaces your model's answer (D-065). The `correct` mode of
`fie.monitor` and of the OpenAI / Anthropic wrappers waits for the server's
verdict, logs it, and returns the ORIGINAL answer. A correction the server
suggests is handed to you as a `CorrectionProposal` (pass `on_proposal=` to
receive it) and is always unverified:

* An updated server (one that returns a `reliability` block) reports the
  suggestion in `fix_result.proposed_output` and never sets `fix_applied`.
* An older server without `reliability` may still say `fix_applied=True` with a
  different `fixed_output`. That is treated as a proposal too: a missing
  `reliability` block is not evidence that the correction was verified.

Automatic replacement stays off until FIE's end-to-end reliability evaluation
passes; there is no switch to turn it on. Standard library only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("fie.correction")

AUTO_REPLACE = False   # fixed; see module docstring

UNVERIFIED = "UNVERIFIED_PROPOSAL"
CONTRACT_CURRENT = "reliability-v1"   # server sends a `reliability` block
CONTRACT_LEGACY = "legacy"            # it does not


@dataclass(frozen=True)
class CorrectionProposal:
    """A correction the server suggested. `applied` is always False."""
    original_output: str
    proposed_output: str
    strategy:        str
    source:          str    # e.g. "cache:tenant_feedback", "SHADOW_CONSENSUS"
    status:          str    # always UNVERIFIED_PROPOSAL
    server_contract: str    # CONTRACT_CURRENT | CONTRACT_LEGACY
    evidence_state:  str    # the server's EvidenceState, or "UNKNOWN" from a legacy server
    answer_action:   str    # the server's AnswerAction, or "UNKNOWN"
    applied:         bool = False


def proposal_from_response(fie_result: Optional[dict], original_output: str) -> Optional[CorrectionProposal]:
    """The server's suggested correction, if it suggests a different answer."""
    if not isinstance(fie_result, dict):
        return None
    fix = fie_result.get("fix_result") or {}
    rel = fie_result.get("reliability")
    if isinstance(rel, dict):
        contract = CONTRACT_CURRENT
        candidate = fix.get("proposed_output") or ""
        # A current server never sets fix_applied; if one does, it is still a proposal.
        if not candidate and fix.get("fix_applied"):
            candidate = fix.get("fixed_output") or ""
        source = rel.get("proposal_source") or fix.get("fix_strategy", "")
        state = rel.get("evidence_state") or "UNKNOWN"
        action = rel.get("answer_action") or "UNKNOWN"
    else:
        contract = CONTRACT_LEGACY
        candidate = (fix.get("fixed_output") or "") if fix.get("fix_applied") else ""
        candidate = candidate or fix.get("proposed_output") or ""
        source = fix.get("fix_strategy", "")
        state = action = "UNKNOWN"
    if not candidate or candidate == original_output:
        return None
    return CorrectionProposal(
        original_output=original_output, proposed_output=candidate,
        strategy=fix.get("fix_strategy", ""), source=source, status=UNVERIFIED,
        server_contract=contract, evidence_state=state, answer_action=action,
    )


def report_proposal(
    fie_result:  Optional[dict],
    original_output: str,
    on_proposal: Optional[Callable[[CorrectionProposal], None]],
    log_prefix:  str = "[FIE]",
) -> Optional[CorrectionProposal]:
    """Log the proposal, hand it to `on_proposal`, and return it. Never replaces anything."""
    proposal = proposal_from_response(fie_result, original_output)
    if proposal is None:
        return None
    logger.warning(
        "%s correction PROPOSED, not applied (unverified) | strategy=%s | source=%s | "
        "server=%s | evidence=%s",
        log_prefix, proposal.strategy, proposal.source, proposal.server_contract,
        proposal.evidence_state,
    )
    if on_proposal is not None:
        try:
            on_proposal(proposal)
        except Exception as exc:   # a user callback must not break the call
            logger.warning("%s on_proposal callback failed: %s", log_prefix, type(exc).__name__)
    return proposal
