from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import AgentVerdict
from config import get_settings
from engine.agents.base_agent import BaseJuryAgent, DiagnosticContext
from engine.archetypes.registry import adversarial_registry, FAISSSearchResult
from engine.prompt_guard import score_prompt_attack


# Attack pattern definitions 

@dataclass
class _AttackPattern:
    category:        str     
    root_cause:      str     
    base_confidence: float   
    pattern:         re.Pattern


_ATTACK_PATTERNS: list[_AttackPattern] = [

    # PROMPT INJECTION 
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
            r"(?:from|starting)\s+now\s+(?:on\s+)?ignore\s+all"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # JAILBREAK 
    _AttackPattern(
        category="JAILBREAK",
        root_cause="JAILBREAK_ATTEMPT",
        base_confidence=0.82,
        pattern=re.compile(
            r"\b("
            # DAN / named jailbreak personas
            r"you\s+are\s+now\s+(?:DAN|an?\s+(?:AI|assistant|model)\s+with(?:out)?\s+(?:no\s+)?restrictions?)|"
            # "act as an AI with no restrictions" — general form
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|free)\s+(?:AI|model|assistant)|"
            r"act\s+as\s+(?:an?\s+)?(?:AI|model|assistant)\s+with(?:out)?\s+(?:no\s+|any\s+)?restrictions?|"
            # pretend/roleplay no policy
            r"(?:pretend|imagine|roleplay\s+as)\s+(?:you\s+have\s+)?(?:no\s+|without\s+)(?:content\s+)?(?:policy|restrictions?|guidelines?|limits?)|"
            # freed from restrictions
            r"you\s+(?:have\s+been\s+)?(?:freed|liberated|released)\s+from\s+(?:all\s+)?(?:restrictions?|guidelines?)|"
            # developer/jailbreak mode — with and without explicit "activated" suffix
            r"(?:developer|jailbreak|debug|unrestricted|god)\s+mode\s+(?:activated?|enabled?|on|where\s+all\s+restrictions?\s+(?:are|have\s+been)\s+(?:lifted|removed|disabled))|"
            # restrictions lifted/removed/suspended
            r"(?:your\s+)?(?:ethical\s+)?(?:restrictions?|guidelines?|limitations?|filters?)\s+"
            r"(?:have\s+been\s+)?(?:removed|disabled|lifted|suspended|deactivated)|"
            # act as if no safety training
            r"(?:act|respond|behave|operate)\s+as\s+(?:if|though)\s+(?:you\s+(?:were|had|have)\s+)?(?:no\s+|never\s+(?:had\s+)?)"
            r"(?:safety|ethical|content)\s+(?:training|guidelines?|restrictions?)|"
            # playing the role of no-restriction AI
            r"you\s+are\s+now\s+playing\s+(?:the\s+role\s+of\s+)?an?\s+(?:AI|assistant)\s+(?:with\s+)?no\s+restrictions?|"
            r"answer\s+only\s+as\s+dan|"
            r"never\s+refuse\s+any\s+request|"
            r"no\s+ethical\s+limits?|"
            r"answer\s+anything"
            r")\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),

    # INSTRUCTION OVERRIDE 
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

    # TOKEN SMUGGLING 
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
            r"\\x00|\\u0000|%00"    # null byte smuggling
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]


# Layer 1: pattern matching 

def _run_pattern_detection(prompt: str) -> tuple[_AttackPattern | None, str]:
    priority_order = ["SMUGGLING", "INJECTION", "JAILBREAK", "OVERRIDE"]
    hits: dict[str, tuple[_AttackPattern, str]] = {}

    for ap in _ATTACK_PATTERNS:
        m = ap.pattern.search(prompt)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100])

    # Return highest-priority hit
    for cat in priority_order:
        if cat in hits:
            return hits[cat]

    return None, ""


def _run_guard_detection(prompt: str) -> tuple[str | None, float, list[str]]:
    signal = score_prompt_attack(prompt)
    if signal.root_cause is None or signal.score < 0.75:
        return None, 0.0, []
    return signal.root_cause, signal.score, list(signal.evidence)

# FAISS semantic search 
def _run_faiss_detection(prompt: str) -> tuple[FAISSSearchResult | None, float]:
    cfg = get_settings()

    try:
        results = adversarial_registry.search(prompt)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    best = results[0]
    if not best.is_match:
        return None, 0.0

    # Normalise the similarity ABOVE threshold to a [0, 1] confidence
    threshold = cfg.jury_adversarial_faiss_threshold
    excess    = best.similarity - threshold
    span      = max(1.0 - threshold, 1e-6)
    faiss_conf = round(min(excess / span, 1.0), 4)

    return best, faiss_conf


# Agent 
class AdversarialSpecialist(BaseJuryAgent):
    agent_name: str = "AdversarialSpecialist"

    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        cfg = get_settings()

        #regex 
        pattern_hit, matched_text = _run_pattern_detection(context.prompt)
        guard_root, guard_confidence, guard_evidence = _run_guard_detection(context.prompt)

        # FAISS semantic search 
        faiss_hit, faiss_confidence = _run_faiss_detection(context.prompt)
        if pattern_hit is None and faiss_hit is None and guard_root is None:
            return self._skip(
                "No adversarial patterns detected by regex, semantic search, or prompt guard "
                f"(FAISS index size: {adversarial_registry.size} patterns). "
                "Failure is likely not an intentional adversarial attack."
            )

        #  Determine root cause 
        if pattern_hit is not None:
            root_cause       = pattern_hit.root_cause
            pattern_conf     = pattern_hit.base_confidence
            # Bonus if FAISS also confirms
            if faiss_hit and faiss_hit.is_match:
                pattern_conf = min(pattern_conf + 0.05, 1.0)
            # Penalty
            if context.fsv.entropy_score < 0.25:
                pattern_conf = max(pattern_conf - 0.08, 0.0)
        elif guard_root is not None:
            root_cause = guard_root
            pattern_conf = guard_confidence
        else:
            # FAISS only hit 
            root_cause   = faiss_hit.record.label
            pattern_conf = 0.0

        # Final confidence 
        if pattern_hit and faiss_hit and faiss_hit.is_match:
            # Both layers agree use the stronger signal
            confidence = max(pattern_conf, faiss_confidence)
        elif pattern_hit and guard_root is not None:
            confidence = max(pattern_conf, guard_confidence)
        elif guard_root is not None and faiss_hit and faiss_hit.is_match:
            confidence = max(guard_confidence, faiss_confidence)
        elif pattern_hit:
            confidence = min(pattern_conf, cfg.jury_adversarial_pattern_confidence)
        elif guard_root is not None:
            confidence = guard_confidence
        else:
            confidence = faiss_confidence

        # Build mitigation string 
        mitigation_map = {
            "PROMPT_INJECTION": (
                "Implement prompt sanitization: strip or escape meta-instruction keywords "
                "before sending to the model. Enforce strict system prompt isolation "
                "using a separate system message that cannot be overridden by user input. "
                "Consider using a dedicated prompt-injection classifier at the input boundary."
            ),
            "JAILBREAK_ATTEMPT": (
                "Add a jailbreak detection layer at the API gateway before the request "
                "reaches the model. Apply output moderation to catch policy-violating "
                "responses even when the input evades detection. Log all jailbreak attempts "
                "for adversarial training data collection."
            ),
            "INSTRUCTION_OVERRIDE": (
                "Treat all user-provided 'authority' claims (admin, developer, supervisor) "
                "as untrusted. Never use prompt-level authority escalation — permissions "
                "must come from authenticated API-level headers, not from prompt text."
            ),
            "TOKEN_SMUGGLING": (
                "Strip or escape all special token sequences before model ingestion: "
                "<|system|>, [INST], ###Human:, null bytes, and similar delimiters. "
                "Use a token-aware sanitizer that understands your model's chat template. "
                "Validate that the rendered prompt does not contain unescaped role boundaries."
            ),
        }
        mitigation = mitigation_map.get(
            root_cause,
            "Implement input sanitization and adversarial prompt monitoring. "
            "Review and harden system prompt isolation policies.",
        )

        # Evidence dict
        evidence: dict = {
            "detection_layers_fired":  [],
            "pattern_match":           None,
            "faiss_result":            None,
            "entropy_score":           context.fsv.entropy_score,
            "ensemble_disagreement":   context.fsv.ensemble_disagreement,
        }

        if pattern_hit:
            evidence["detection_layers_fired"].append("regex")
            evidence["pattern_match"] = {
                "category":        pattern_hit.category,
                "root_cause":      pattern_hit.root_cause,
                "matched_text":    matched_text,
                "base_confidence": pattern_hit.base_confidence,
            }

        if guard_root is not None:
            evidence["detection_layers_fired"].append("prompt_guard")
            evidence["prompt_guard"] = {
                "root_cause": guard_root,
                "confidence": guard_confidence,
                "evidence": guard_evidence[:5],
            }

        if faiss_hit:
            evidence["detection_layers_fired"].append("faiss")
            evidence["faiss_result"] = {
                "nearest_prompt":  faiss_hit.record.prompt[:120],
                "label":           faiss_hit.record.label,
                "category":        faiss_hit.record.category,
                "similarity":      faiss_hit.similarity,
                "faiss_confidence": faiss_confidence,
            }

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


# Module-level singleton 
adversarial_specialist = AdversarialSpecialist()
