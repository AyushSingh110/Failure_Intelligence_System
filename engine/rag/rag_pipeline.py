from __future__ import annotations

import logging

from engine.rag.retriever import fetch_wikipedia_summary

logger = logging.getLogger(__name__)


def build_rag_prompt(prompt: str, context: str | None = None) -> str:
    """
    Wikipedia context ke saath augmented prompt banata hai.
    Agar Wikipedia context nahi mila → original prompt return karta hai.
    """
    context = context or fetch_wikipedia_summary(prompt)

    if not context:
        logger.debug("No Wikipedia context found for: %s", prompt[:60])
        return prompt

    logger.debug("RAG context found (%d chars) for: %s", len(context), prompt[:60])

    return f"""You are answering a factual question using verified reference context.
Use only the context below.
If the answer is clearly present, answer directly and confidently.
For short factoid questions, answer in exactly one complete sentence.
Do not say the context is insufficient unless the answer truly is missing.

Context (from Wikipedia):
{context}

Question:
{prompt}

Answer:"""


def get_grounded_answer(prompt: str, context: str | None = None) -> str:
    """
    RAG + Groq se grounded answer generate karta hai.
    Ollama ki jagah Groq use karta hai.

    Returns:
        Grounded answer string, or empty string on failure.
    """
    try:
        from engine.groq_service import get_groq_service

        groq = get_groq_service()
        if not groq:
            logger.warning("GroqService unavailable for RAG grounding")
            return ""

        rag_prompt = build_rag_prompt(prompt, context)
        response   = groq.complete(
            rag_prompt,
            model_name  = "llama-3.1-8b-instant",
            max_tokens  = 120,
            temperature = 0.0,
        )

        if response.success:
            logger.info("RAG grounded answer: %s...", response.output_text[:80])
            return response.output_text
        else:
            logger.warning("Groq RAG completion failed: %s", response.error)
            return ""

    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc)
        return ""
