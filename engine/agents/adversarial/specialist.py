"""
AdversarialSpecialist jury agent.

Orchestrates all 10 detection layers and combines their verdicts into a single
AgentVerdict with root cause, confidence, mitigation strategy, and evidence.
"""
from __future__ import annotations
from app.schemas import AgentVerdict
from config import get_settings
from engine.agents.base_agent import BaseJuryAgent, DiagnosticContext
from engine.archetypes.registry import adversarial_registry

from engine.agents.adversarial.injection  import (
    run_pattern_detection,
    run_guard_detection,
    run_faiss_detection,
    run_indirect_injection_detection,
)
from engine.agents.adversarial.many_shot                import run_many_shot_detection
from engine.agents.adversarial.gcg                      import run_gcg_detection
from engine.agents.adversarial.perplexity               import run_perplexity_proxy
from engine.agents.adversarial.semantic                 import run_semantic_consistency, run_exfiltration_detection
from engine.agents.adversarial.llm_intent               import run_llm_intent_check
from engine.agents.adversarial.multilingual_romanisation import run_romanisation_detection


_MITIGATION_MAP: dict[str, str] = {
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
    "SEMANTIC_CONSISTENCY_VIOLATION": (
        "The model's output is topically inconsistent with the input prompt — "
        "a strong indicator that an adversarial injection succeeded and the model "
        "answered a different question than what was asked. "
        "Mitigations: (1) Compare semantic similarity between the user's original "
        "intent and the model's response before returning it. (2) Use intent "
        "classification on the prompt and category classification on the output. "
        "(3) For document-processing tasks, always remind the model of its primary "
        "task in the output instructions."
    ),
    "PROMPT_EXFILTRATION": (
        "The model's output shows signs of system prompt exfiltration. "
        "Mitigations: (1) Inject a canary token into your system prompt and monitor "
        "for it in outputs. (2) Add an explicit anti-disclosure instruction to your "
        "system prompt. (3) Apply output filtering to strip content matching your "
        "system prompt structure. (4) Log and alert on all exfiltration attempts."
    ),
    "OBFUSCATED_ADVERSARIAL_PAYLOAD": (
        "This prompt has statistical properties consistent with an encoded or "
        "obfuscated adversarial payload. Likely attack types: base64-encoded "
        "injection, Caesar/ROT cipher bypass, Unicode lookalike payload, or GCG "
        "noise sequence. Mitigations: (1) Reject prompts with compression ratio "
        "> 0.80. (2) Apply token vocabulary filtering. (3) Set a prompt entropy "
        "budget. (4) Consider base64 decoding as a pre-processing step."
    ),
    "GCG_ADVERSARIAL_SUFFIX": (
        "A high-entropy token suffix consistent with a GCG adversarial attack was "
        "detected. Mitigations: (1) Strip or truncate anomalously high-entropy tail "
        "segments. (2) Apply a perplexity threshold filter. (3) Set a maximum prompt "
        "length policy. (4) Log and escalate all GCG detections."
    ),
    "MANY_SHOT_JAILBREAK": (
        "A many-shot jailbreak was detected: the prompt embeds scripted Q/A exchanges "
        "to condition the model via in-context learning (Anil et al., 2024). "
        "Mitigations: (1) Cap the number of in-context examples per request. (2) Scan "
        "all Human/Q turns for harmful content. (3) Refuse prompts with >4 alternating "
        "turns not from your conversation history. (4) Use a conversation history allowlist."
    ),
    "ROLEPLAY_JAILBREAK": (
        "A roleplay/narrative wrapper jailbreak was detected. "
        "Mitigations: (1) Apply intent-based filtering — the harmful content is the "
        "same regardless of fictional framing. (2) Add explicit system prompt "
        "instructions that fictional framing does not suspend content policies. "
        "(3) Flag prompts combining narrative framing with harmful topic keywords."
    ),
    "CROSS_LINGUAL_ROMANISATION_ATTACK": (
        "A cross-lingual romanisation attack was detected — the prompt is written in "
        "a romanised non-Latin script (Pinyin / Arabizi / Romaji / Korean RR / IAST) "
        "to bypass English-language safety filters. "
        "Mitigations: (1) Apply script-aware normalisation before pattern matching. "
        "(2) Flag any prompt with a high romanisation score for secondary review. "
        "(3) Consider rejecting or escalating prompts that combine a detected script "
        "with harm-adjacent vocabulary in that script."
    ),
}

_DEFAULT_MITIGATION = (
    "Implement input sanitization and adversarial prompt monitoring. "
    "Review and harden system prompt isolation policies."
)


class AdversarialSpecialist(BaseJuryAgent):
    agent_name: str = "AdversarialSpecialist"

    def analyze(self, context: DiagnosticContext) -> AgentVerdict:
        cfg = get_settings()

        # Layer 1: regex pattern matching
        pattern_hit, matched_text = run_pattern_detection(context.prompt)
        # Layer 2: statistical prompt guard
        guard_root, guard_confidence, guard_evidence = run_guard_detection(context.prompt)
        # Layer 3: FAISS semantic search
        faiss_hit, faiss_confidence = run_faiss_detection(context.prompt)
        # Layer 3b: many-shot / few-shot jailbreak
        many_root, many_confidence, many_evidence = run_many_shot_detection(context.prompt)
        # Layer 3d: cross-lingual romanisation attack (Pinyin / Arabizi / Romaji / Korean RR / IAST)
        roman_root, roman_confidence, roman_evidence = run_romanisation_detection(context.prompt)
        # Layer 3c: roleplay / narrative wrapper jailbreak
        try:
            from engine.roleplay_detector import detect_roleplay_jailbreak
            _rp = detect_roleplay_jailbreak(context.prompt)
            roleplay_root       = "ROLEPLAY_JAILBREAK" if _rp.is_roleplay_jailbreak else None
            roleplay_confidence = _rp.confidence
            roleplay_evidence   = {
                "framing_matched":       _rp.framing_matched,
                "harmful_topic_matched": _rp.harmful_topic_matched,
            } if _rp.is_roleplay_jailbreak else {}
        except Exception:
            roleplay_root, roleplay_confidence, roleplay_evidence = None, 0.0, {}
        # Layer 4: indirect prompt injection
        indirect_root, indirect_confidence, indirect_evidence = run_indirect_injection_detection(
            context.prompt, context.primary_output
        )
        # Layer 5: GCG adversarial suffix
        gcg_root, gcg_confidence, gcg_evidence = run_gcg_detection(context.prompt)
        # Layer 6: perplexity proxy
        perp_root, perp_confidence, perp_evidence = run_perplexity_proxy(context.prompt)
        # Layer 7: canary token + output exfiltration
        canary = getattr(context, "canary_token", None)
        exfil_root, exfil_confidence, exfil_evidence = run_exfiltration_detection(
            context.prompt, context.primary_output, canary=canary
        )
        # Layer 8: output semantic consistency
        sem_root, sem_confidence, sem_evidence = run_semantic_consistency(
            context.prompt, context.primary_output
        )

        # Layer 9: LLM intent check — only when no high-confidence structural hit
        _high_conf_structural = (
            (pattern_hit is not None and pattern_hit.base_confidence >= 0.80)
            or (guard_root    is not None and guard_confidence    >= 0.80)
            or (faiss_hit     is not None and faiss_confidence    >= 0.80)
            or (indirect_root is not None and indirect_confidence >= 0.80)
            or (gcg_root      is not None and gcg_confidence      >= 0.80)
            or (perp_root     is not None and perp_confidence     >= 0.80)
            or (exfil_root    is not None and exfil_confidence    >= 0.80)
            or (sem_root      is not None and sem_confidence      >= 0.80)
            or (many_root     is not None and many_confidence     >= 0.80)
            or (roleplay_root is not None and roleplay_confidence >= 0.80)
            or (roman_root    is not None and roman_confidence    >= 0.80)
        )
        intent_root, intent_confidence, intent_evidence = None, 0.0, {}
        if not _high_conf_structural:
            intent_root, intent_confidence, intent_evidence = run_llm_intent_check(context.prompt)
            if intent_root is None and (
                pattern_hit is None and faiss_hit is None and guard_root is None
                and indirect_root is None and gcg_root is None and perp_root is None
                and exfil_root is None and sem_root is None and many_root is None
                and roleplay_root is None and roman_root is None
            ):
                return self._skip(
                    "No adversarial patterns detected by regex, semantic search, prompt guard, "
                    f"indirect injection, GCG suffix, perplexity proxy, exfiltration scanner, "
                    f"semantic consistency check, or LLM intent analysis "
                    f"(FAISS index size: {adversarial_registry.size} patterns). "
                    "Failure is likely not an intentional adversarial attack."
                )

        # Determine root cause — priority order:
        # Layer 7 (exfil) → Layer 4 (indirect) → Layer 1 (regex)
        # → Layer 2 (guard) → Layer 5 (GCG) → Layer 6 (perplexity)
        # → Layer 8 (semantic) → Layer 3b (many-shot) → Layer 3c (roleplay)
        # → Layer 9 (intent) → Layer 3 (FAISS)
        if exfil_root is not None and exfil_confidence >= 0.80:
            root_cause   = exfil_root
            pattern_conf = exfil_confidence
        elif indirect_root is not None and indirect_confidence >= 0.80:
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
        elif gcg_root is not None:
            root_cause   = gcg_root
            pattern_conf = gcg_confidence
        elif perp_root is not None:
            root_cause   = perp_root
            pattern_conf = perp_confidence
        elif sem_root is not None:
            root_cause   = sem_root
            pattern_conf = sem_confidence
        elif many_root is not None:
            root_cause   = many_root
            pattern_conf = many_confidence
        elif roleplay_root is not None:
            root_cause   = roleplay_root
            pattern_conf = roleplay_confidence
        elif roman_root is not None:
            root_cause   = roman_root
            pattern_conf = roman_confidence
        elif intent_root is not None:
            root_cause   = intent_root
            pattern_conf = intent_confidence
        else:
            root_cause   = faiss_hit.record.label if faiss_hit and faiss_hit.record else "ADVERSARIAL_PROMPT"
            pattern_conf = 0.0

        # Final confidence — max across all firing layers
        active_confidences = []
        if pattern_hit:
            active_confidences.append(pattern_conf)
        if guard_root is not None:
            active_confidences.append(guard_confidence)
        if faiss_hit and faiss_hit.is_match:
            active_confidences.append(faiss_confidence)
        if indirect_root is not None:
            active_confidences.append(indirect_confidence)
        if gcg_root is not None:
            active_confidences.append(gcg_confidence)
        if perp_root is not None:
            active_confidences.append(perp_confidence)
        if exfil_root is not None:
            active_confidences.append(exfil_confidence)
        if sem_root is not None:
            active_confidences.append(sem_confidence)
        if many_root is not None:
            active_confidences.append(many_confidence)
        if roleplay_root is not None:
            active_confidences.append(roleplay_confidence)
        if roman_root is not None:
            active_confidences.append(roman_confidence)
        if intent_root is not None:
            active_confidences.append(intent_confidence)

        confidence = max(active_confidences) if active_confidences else 0.0
        mitigation = _MITIGATION_MAP.get(root_cause, _DEFAULT_MITIGATION)

        # Build evidence dict
        evidence: dict = {
            "detection_layers_fired": [],
            "pattern_match":          None,
            "faiss_result":           None,
            "entropy_score":          context.fsv.entropy_score,
            "ensemble_disagreement":  context.fsv.ensemble_disagreement,
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
                "evidence":   guard_evidence[:5],
            }

        if faiss_hit:
            evidence["detection_layers_fired"].append("faiss")
            evidence["faiss_result"] = {
                "nearest_prompt":   faiss_hit.record.prompt[:120] if faiss_hit.record else "",
                "label":            faiss_hit.record.label        if faiss_hit.record else "UNKNOWN",
                "category":         faiss_hit.record.category     if faiss_hit.record else "UNKNOWN",
                "similarity":       faiss_hit.similarity,
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

        if gcg_root is not None:
            evidence["detection_layers_fired"].append("gcg_suffix")
            evidence["gcg_suffix"] = {
                "confidence":           gcg_confidence,
                "tail_entropy":         gcg_evidence.get("tail_entropy"),
                "tail_special_density": gcg_evidence.get("tail_special_density"),
                "non_word_density":     gcg_evidence.get("non_word_token_density"),
                "signals_fired":        gcg_evidence.get("signals_fired"),
                "tail_preview":         gcg_evidence.get("tail_preview"),
            }

        if perp_root is not None:
            evidence["detection_layers_fired"].append("perplexity_proxy")
            evidence["perplexity_proxy"] = {
                "confidence":            perp_confidence,
                "compression_ratio":     perp_evidence.get("compression_ratio"),
                "non_dict_density":      perp_evidence.get("non_dict_density"),
                "char_type_entropy":     perp_evidence.get("char_type_entropy"),
                "token_length_variance": perp_evidence.get("token_length_variance"),
                "signals_fired":         perp_evidence.get("signals_fired"),
            }

        if exfil_root is not None:
            evidence["detection_layers_fired"].append("exfiltration")
            evidence["exfiltration"] = {
                "confidence":       exfil_confidence,
                "method":           exfil_evidence.get("method"),
                "canary_leaked":    exfil_evidence.get("canary_leaked"),
                "patterns_matched": exfil_evidence.get("patterns_matched"),
                "evidence_snippet": exfil_evidence.get("evidence_snippet"),
            }

        if sem_root is not None:
            evidence["detection_layers_fired"].append("semantic_consistency")
            evidence["semantic_consistency"] = {
                "confidence":       sem_confidence,
                "jaccard":          sem_evidence.get("jaccard_similarity"),
                "is_creative_task": sem_evidence.get("is_creative_task"),
                "harmful_pivot":    sem_evidence.get("output_harmful"),
                "signals_fired":    sem_evidence.get("signals_fired"),
            }

        if intent_root is not None:
            evidence["detection_layers_fired"].append("llm_intent")
            evidence["llm_intent"] = {
                "confidence": intent_confidence,
                "verdict":    intent_evidence.get("verdict"),
                "reasoning":  intent_evidence.get("reasoning"),
                "model":      intent_evidence.get("model"),
            }

        if many_root is not None:
            evidence["detection_layers_fired"].append("many_shot")
            evidence["many_shot"] = {
                "confidence":      many_confidence,
                "pair_count":      many_evidence.get("pair_count"),
                "harmful_q_count": many_evidence.get("harmful_q_count"),
                "harmful_ratio":   many_evidence.get("harmful_ratio"),
                "escalation":      many_evidence.get("escalation"),
                "signals_fired":   many_evidence.get("signals_fired"),
                "last_q_preview":  many_evidence.get("last_q_preview"),
            }

        if roleplay_root is not None:
            evidence["detection_layers_fired"].append("roleplay_jailbreak")
            evidence["roleplay_jailbreak"] = {
                "confidence":            roleplay_confidence,
                "framing_matched":       roleplay_evidence.get("framing_matched"),
                "harmful_topic_matched": roleplay_evidence.get("harmful_topic_matched"),
            }

        if roman_root is not None:
            evidence["detection_layers_fired"].append("romanisation")
            evidence["romanisation"] = {
                "confidence":   roman_confidence,
                "script":       roman_evidence.get("best_script"),
                "script_score": roman_evidence.get("script_score"),
                "harm_vocab":   roman_evidence.get("harm_vocab_hit"),
                "harm_terms":   roman_evidence.get("harm_terms"),
                "all_scores":   roman_evidence.get("all_scores"),
            }

        return self._verdict(
            root_cause=root_cause,
            confidence_score=round(min(confidence, 1.0), 4),
            mitigation_strategy=mitigation,
            evidence=evidence,
        )


adversarial_specialist = AdversarialSpecialist()
