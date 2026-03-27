from __future__ import annotations

import json
import logging

from app.schemas import ExplanationBundle, HumanExplanation

logger = logging.getLogger(__name__)


_SEVERITY_MAP = {
    "PROMPT_INJECTION": "high",
    "JAILBREAK_ATTEMPT": "high",
    "TOKEN_SMUGGLING": "high",
    "INSTRUCTION_OVERRIDE": "high",
    "FACTUAL_HALLUCINATION": "high",
    "MODEL_BLIND_SPOT": "medium",
    "TEMPORAL_KNOWLEDGE_CUTOFF": "medium",
    "PROMPT_COMPLEXITY_OOD": "medium",
    "STABLE": "low",
}


def build_human_explanation(bundle: ExplanationBundle) -> HumanExplanation:
    """
    Convert structured XAI into a short human-friendly explanation.

    Groq is used only as a formatter. If it is unavailable, the system falls
    back to a deterministic explanation so the UI never breaks.
    """
    llm_result = _generate_with_groq(bundle)
    if llm_result is not None:
        return llm_result
    return _fallback_explanation(bundle)


def _generate_with_groq(bundle: ExplanationBundle) -> HumanExplanation | None:
    try:
        from config import get_settings
        from engine.groq_service import get_groq_service

        settings = get_settings()
        groq = get_groq_service()
        if not groq:
            return None

        prompt = _build_humanizer_prompt(bundle)
        model_name = None
        preferred = "llama-3.3-70b-versatile"
        configured_models = set(settings.groq_models or [])
        if preferred in configured_models:
            model_name = preferred

        response = groq.complete(
            prompt,
            model_name=model_name,
            max_tokens=220,
            temperature=0.1,
        )
        if not response.success or not response.output_text:
            return None

        payload = _extract_json_payload(response.output_text)
        if not payload:
            return None

        return HumanExplanation(
            summary=str(payload.get("summary", "")).strip() or bundle.summary,
            why_risky=str(payload.get("why_risky", "")).strip() or _fallback_explanation(bundle).why_risky,
            recommended_action=str(payload.get("recommended_action", "")).strip() or _fallback_explanation(bundle).recommended_action,
            severity=str(payload.get("severity", _SEVERITY_MAP.get(bundle.final_label, "medium"))).lower(),
            generated_by=f"groq:{response.model_name}",
            safe_for_user=True,
        )
    except Exception as exc:
        logger.warning("Human explanation generation fell back to template mode: %s", exc)
        return None


def _extract_json_payload(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _build_humanizer_prompt(bundle: ExplanationBundle) -> str:
    top_signals = [
        {
            "name": signal.name,
            "score": signal.normalized_score,
            "summary": signal.summary,
        }
        for signal in bundle.signals[:4]
    ]
    top_evidence = [
        {
            "type": evidence.type,
            "supports": evidence.supports,
            "content_preview": evidence.content_preview,
        }
        for evidence in bundle.evidence[:3]
    ]

    return (
        "You are writing a short, plain-language explanation for a developer dashboard.\n"
        "Use ONLY the structured data provided. Do not invent facts. Do not mention internal detection rules.\n"
        "Keep it non-technical and concise.\n"
        "Return valid JSON with exactly these keys: "
        'summary, why_risky, recommended_action, severity.\n\n'
        f"final_label: {bundle.final_label}\n"
        f"final_fix_strategy: {bundle.final_fix_strategy}\n"
        f"summary: {bundle.summary}\n"
        f"top_signals: {json.dumps(top_signals)}\n"
        f"top_evidence: {json.dumps(top_evidence)}\n"
        f"uncertainty_notes: {json.dumps(bundle.uncertainty_notes[:2])}\n"
    )


def _fallback_explanation(bundle: ExplanationBundle) -> HumanExplanation:
    label = bundle.final_label.replace("_", " ").title()
    severity = _SEVERITY_MAP.get(bundle.final_label, "medium")

    signal_hint = bundle.signals[0].summary if bundle.signals else "The system observed behavior that suggests elevated risk."
    fix_hint = _recommended_action(bundle.final_fix_strategy)

    return HumanExplanation(
        summary=f"The system flagged this event as {label} because multiple monitoring signals pointed to the same issue.",
        why_risky=f"This is risky because it can lead to unreliable or unsafe answers for end users. {signal_hint}",
        recommended_action=fix_hint,
        severity=severity,
        generated_by="template",
        safe_for_user=True,
    )


def _recommended_action(strategy: str) -> str:
    strategy = (strategy or "").upper()
    if strategy == "SANITIZE_AND_RERUN":
        return "Review the prompt, remove unsafe instruction-manipulation text, and retry with a clean request."
    if strategy == "CONTEXT_INJECTION":
        return "Use an up-to-date source or retrieval step before answering questions that need live information."
    if strategy == "RAG_GROQ_GROUNDING":
        return "Verify the answer against grounded sources and return the grounded version instead of the original guess."
    if strategy == "PROMPT_DECOMPOSITION":
        return "Break the task into simpler sub-questions so the model can reason through it more reliably."
    if strategy == "SHADOW_CONSENSUS":
        return "Compare the primary answer with reference model outputs and use the consistent answer path."
    return "Review the flagged response and validate it before showing it to users."
