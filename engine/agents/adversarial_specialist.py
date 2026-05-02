from __future__ import annotations

import re
import unicodedata
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


# ── Layer 4 helpers: indirect prompt injection ────────────────────────────────

# Signals that the prompt contains external content to process
_DOCUMENT_TRIGGER_RE = re.compile(
    r"(?:"
    r"(?:summarize|translate|analyze|review|process|read|evaluate|"
    r"proofread|edit|classify|extract|parse|convert)"
    r"\s*(?:the\s+following|this\s+(?:text|document|email|message|article)|:)"
    r"|(?:based\s+on|given|using|from|according\s+to)\s+the\s+following"
    r"|(?:document|email|message|text|content|article|passage|report)\s*:\s*"
    r"|```|<!--|<(?:document|doc|email|message|context|text|article|data)\s*>"
    r")",
    re.IGNORECASE,
)

# Injected instructions hidden inside document content
_INDIRECT_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|the\s+(?:user|original))\s+instructions?"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?"
    r"|(?:new|additional|updated|changed)\s+(?:instruction|directive|task|command)\s*[:;]"
    r"|your\s+(?:new|real|actual|true)\s+(?:task|purpose|goal|instructions?)\s+(?:is|are)"
    r"|forget\s+(?:the|your)?\s*(?:user|human|original)?\s*(?:task|question|request|instructions?)"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing|reviewing|processing|answering)"
    r"|do\s+not\s+(?:summarize|translate|analyze|review|process|answer)\s+(?:this|the|it)"
    r"|(?:from\s+now|starting\s+now|as\s+of\s+now)\s*,?\s*(?:you\s+(?:must|should|will)|ignore)"
    r"|(?:override|system|admin)\s*:\s*(?:new|updated|changed|ignore)"
    r"|(?:actually|instead)\s*,?\s*(?:your\s+task|you\s+should|do\s+the\s+following)"
    r"|<\s*(?:new_)?(?:system|instructions?|task)\s*>.*?<\s*/\s*(?:system|instructions?|task)\s*>"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Signs the model's output shows it followed an injected instruction
_OUTPUT_COMPLIANCE_RE = re.compile(
    r"(?:"
    r"as\s+(?:instructed|requested|directed|told)\s+(?:in|by)\s+the\s+(?:document|text|email|message|file)"
    r"|the\s+(?:document|text|email|message)\s+(?:asks?\s+me|instructed?\s+me|told?\s+me|requests?\s+that\s+I)"
    r"|new\s+(?:instructions?|task|directive)\s+(?:received|detected|found|identified)"
    r"|switching\s+to\s+(?:new|different|updated)\s+(?:task|mode|instructions?)"
    r"|ignoring\s+(?:previous|original|prior)\s+(?:task|instructions?|request|question)"
    r"|my\s+(?:new|updated|changed|actual|real)\s+(?:task|purpose|goal|instructions?)\s+is"
    r"|instead\s+of\s+(?:summarizing|translating|analyzing),?\s+I\s+(?:will|am|have)"
    r")",
    re.IGNORECASE,
)


def _extract_document_portion(prompt: str) -> str | None:
    """
    Returns the content portion of the prompt (after the user task instruction)
    if the prompt contains a document/external content pattern, else None.
    """
    m = _DOCUMENT_TRIGGER_RE.search(prompt)
    if not m:
        return None
    # Everything after the trigger point is the "document content"
    doc_start = m.end()
    portion = prompt[doc_start:].strip()
    # Only return if there's a meaningful amount of content
    return portion if len(portion) > 40 else None


def _run_indirect_injection_detection(
    prompt: str,
    primary_output: str,
) -> tuple[str | None, float, dict]:
    """
    Layer 4: Indirect prompt injection detection.

    Scans:
    - The document/content portion of the prompt for embedded instructions
    - The model's output for signs it followed an injected instruction

    Returns (root_cause | None, confidence, evidence_dict).
    """
    doc_portion = _extract_document_portion(prompt)
    if doc_portion is None:
        # Also check full prompt for cross-prompt injection (no explicit separator)
        full_injection = _INDIRECT_INJECTION_RE.search(prompt)
        if not full_injection:
            return None, 0.0, {}
        # Found injection phrasing in the prompt itself — lower confidence because
        # it might be the user's actual legitimate request
        output_fired = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output or ""))
        conf = 0.72 if output_fired else 0.45
        return "INDIRECT_PROMPT_INJECTION", conf, {
            "document_found": False,
            "injection_in_prompt": full_injection.group(0)[:120],
            "output_compliance_detected": output_fired,
        }

    injection_match = _INDIRECT_INJECTION_RE.search(doc_portion)
    output_fired    = bool(_OUTPUT_COMPLIANCE_RE.search(primary_output or ""))

    if not injection_match and not output_fired:
        return None, 0.0, {}

    # Confidence matrix
    if injection_match and output_fired:
        # Both sides confirm — the injection happened and the model followed it
        confidence = 0.88
    elif injection_match:
        # Injection found in document but model output looks normal — partial signal
        confidence = 0.65
    else:
        # Model output looks like it followed something, but no explicit injection text
        confidence = 0.52

    evidence = {
        "document_found": True,
        "document_snippet": doc_portion[:200],
        "injection_pattern_matched": injection_match.group(0)[:120] if injection_match else None,
        "output_compliance_detected": output_fired,
        "output_snippet": (primary_output or "")[:150] if output_fired else None,
    }
    return "INDIRECT_PROMPT_INJECTION", confidence, evidence


# ── Obfuscation normalization (Layer 0) ──────────────────────────────────────
# Attackers bypass regex by spacing characters, using Cyrillic homoglyphs,
# leet-speak substitutions, or zero-width unicode. Normalize before matching.

_HOMOGLYPH_MAP = str.maketrans({
    # Cyrillic chars that look identical to Latin
    "а": "a",  # а → a
    "е": "e",  # е → e
    "і": "i",  # і → i
    "о": "o",  # о → o
    "р": "p",  # р → p
    "с": "c",  # с → c
    "х": "x",  # х → x
    # Greek
    "α": "a",  # α → a
    "ο": "o",  # ο → o
    # Leet substitutions
    "@": "a", "0": "o", "1": "i", "3": "e", "5": "s", "7": "t", "$": "s",
})


def _normalize_for_detection(text: str) -> str:
    """
    Returns a normalized copy of text with obfuscation removed.
    Used to run pattern detection against bypass attempts without modifying
    the original prompt stored in logs.
    """
    # 1. Unicode NFKC: decompose + recompose, collapses fullwidth/halfwidth chars
    text = unicodedata.normalize("NFKC", text)
    # 2. Map homoglyphs and leet chars to their ASCII equivalents
    text = text.translate(_HOMOGLYPH_MAP)
    # 3. Collapse single-space-separated letters: "i g n o r e" → "ignore"
    #    Only collapses when every token between spaces is a single letter
    text = re.sub(r"\b([a-zA-Z]) (?=[a-zA-Z]\b)", r"\1", text)
    text = re.sub(r"\b([a-zA-Z]) (?=[a-zA-Z]\b)", r"\1", text)  # second pass
    # 4. Strip zero-width and invisible unicode characters
    text = re.sub(r"[​‌‍⁠﻿­]", "", text)
    return text


# Layer 1: pattern matching

def _run_pattern_detection(prompt: str) -> tuple[_AttackPattern | None, str]:
    """
    Scan prompt for known attack patterns.
    Runs against both the original text and a normalized (de-obfuscated) copy
    so that spaced-out characters, leet-speak, and homoglyphs don't bypass detection.
    """
    priority_order = ["SMUGGLING", "INJECTION", "JAILBREAK", "OVERRIDE"]
    normalized = _normalize_for_detection(prompt)
    hits: dict[str, tuple[_AttackPattern, str, bool]] = {}  # cat → (pattern, text, was_obfuscated)

    for ap in _ATTACK_PATTERNS:
        # Try original first
        m = ap.pattern.search(prompt)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], False)
            continue
        # Try normalized — catches obfuscated variants
        m = ap.pattern.search(normalized)
        if m:
            hits[ap.category] = (ap, m.group(0)[:100], True)

    for cat in priority_order:
        if cat in hits:
            ap, matched_text, obfuscated = hits[cat]
            # Slightly lower confidence for obfuscated detections
            if obfuscated:
                ap = _AttackPattern(
                    category=ap.category,
                    root_cause=ap.root_cause,
                    base_confidence=max(ap.base_confidence - 0.06, 0.50),
                    pattern=ap.pattern,
                )
            return ap, matched_text
    return None, ""


def _run_guard_detection(prompt: str) -> tuple[str | None, float, list[str]]:
    # Run on original; if no hit, try normalized to catch obfuscated variants
    signal = score_prompt_attack(prompt)
    if signal.root_cause is None or signal.score < 0.75:
        normalized = _normalize_for_detection(prompt)
        if normalized != prompt:
            signal = score_prompt_attack(normalized)
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

        # Layer 1: regex pattern matching
        pattern_hit, matched_text = _run_pattern_detection(context.prompt)
        # Layer 2: statistical prompt guard
        guard_root, guard_confidence, guard_evidence = _run_guard_detection(context.prompt)
        # Layer 3: FAISS semantic search
        faiss_hit, faiss_confidence = _run_faiss_detection(context.prompt)
        # Layer 4: indirect prompt injection (document-embedded attacks)
        indirect_root, indirect_confidence, indirect_evidence = _run_indirect_injection_detection(
            context.prompt, context.primary_output
        )

        if pattern_hit is None and faiss_hit is None and guard_root is None and indirect_root is None:
            return self._skip(
                "No adversarial patterns detected by regex, semantic search, prompt guard, "
                f"or indirect injection scanner (FAISS index size: {adversarial_registry.size} patterns). "
                "Failure is likely not an intentional adversarial attack."
            )

        # Determine root cause — Layer 4 (indirect) takes priority when it fires
        # at high confidence (both document injection + output compliance confirmed)
        if indirect_root is not None and indirect_confidence >= 0.80:
            root_cause   = indirect_root
            pattern_conf = indirect_confidence
        elif pattern_hit is not None:
            root_cause   = pattern_hit.root_cause
            pattern_conf = pattern_hit.base_confidence
            if faiss_hit and faiss_hit.is_match:
                pattern_conf = min(pattern_conf + 0.05, 1.0)
            if context.fsv.entropy_score < 0.25:
                pattern_conf = max(pattern_conf - 0.08, 0.0)
        elif guard_root is not None:
            root_cause   = guard_root
            pattern_conf = guard_confidence
        elif indirect_root is not None:
            root_cause   = indirect_root
            pattern_conf = indirect_confidence
        else:
            # FAISS only hit
            root_cause   = faiss_hit.record.label
            pattern_conf = 0.0

        # Final confidence — take max across all firing layers
        active_confidences = []
        if pattern_hit:
            active_confidences.append(pattern_conf)
        if guard_root is not None:
            active_confidences.append(guard_confidence)
        if faiss_hit and faiss_hit.is_match:
            active_confidences.append(faiss_confidence)
        if indirect_root is not None:
            active_confidences.append(indirect_confidence)

        confidence = max(active_confidences) if active_confidences else 0.0

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
            "INDIRECT_PROMPT_INJECTION": (
                "Treat all external content (documents, emails, webpages, API responses) "
                "as untrusted data — never as instructions. Use a strict separation: "
                "system instructions stay in the system prompt, user content is wrapped "
                "in explicit data tags (e.g. <document>...</document>) and the model is "
                "instructed to treat everything inside those tags as data only. "
                "Apply output scanning to detect when the model's response shows "
                "evidence of having followed embedded instructions rather than the "
                "user's original request. This is the fastest-growing LLM attack "
                "vector in 2025-2026 (OWASP GenAI Top 10, LLM01)."
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

        if indirect_root is not None:
            evidence["detection_layers_fired"].append("indirect_injection")
            evidence["indirect_injection"] = {
                "confidence":                indirect_confidence,
                "document_found":            indirect_evidence.get("document_found"),
                "injection_pattern_matched": indirect_evidence.get("injection_pattern_matched"),
                "output_compliance_detected": indirect_evidence.get("output_compliance_detected"),
                "document_snippet":          indirect_evidence.get("document_snippet"),
                "output_snippet":            indirect_evidence.get("output_snippet"),
            }

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


# Module-level singleton 
adversarial_specialist = AdversarialSpecialist()
