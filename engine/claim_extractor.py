from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """
    A single atomic factual claim extracted from a model output.
    """
    subject:  str
    property: str
    value:    str
    raw_text: str


# Extraction prompt 
_EXTRACTION_PROMPT = """\
You are a fact extraction system. Read the question and the answer below, then extract the MAIN factual claim.

Return ONLY a valid JSON object with exactly these three fields:
  "subject"  — the thing the question is asking about (e.g. "telephone", "Australia", "Einstein")
  "property" — the relationship or attribute being asked about (e.g. "inventor", "capital", "birthplace")
  "value"    — the stated answer value (e.g. "Alexander Graham Bell", "Sydney", "Ulm")

Rules:
- The subject must come from the QUESTION, not from the answer — it is what the question is ABOUT
- If there is no clear, verifiable factual claim, return: null
- Do not include explanation, markdown, or extra text — ONLY the JSON or null
- Use simple lowercase snake_case for property names
- Keep values as concise proper nouns

Question: {question}
Answer: {text}

JSON:"""

# Fallback prompt when no question is available
_EXTRACTION_PROMPT_NO_QUESTION = """\
You are a fact extraction system. Read the sentence below and extract the MAIN factual claim.

Return ONLY a valid JSON object with exactly these three fields:
  "subject"  — the thing being described (e.g. "telephone", "Einstein", "Tesla Inc")
  "property" — the relationship or attribute (e.g. "inventor", "birthplace", "founded_by")
  "value"    — the stated value or answer (e.g. "Alexander Graham Bell", "Germany", "Elon Musk")

Rules:
- If there is no clear, verifiable factual claim, return: null
- Do not include explanation, markdown, or extra text — ONLY the JSON or null
- Use simple lowercase snake_case for property names
- Keep values as concise proper nouns

Sentence: {text}

JSON:"""


def extract_claim(text: str, question: str = "") -> Optional[ExtractedClaim]:
    """
    Extracts the main atomic factual claim from a model output string.
    """
    if not text or len(text.strip()) < 3:
        return None

    try:
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if not groq:
            logger.debug("Groq unavailable — claim extraction skipped")
            return None

        if question and len(question.strip()) >= 5:
            prompt = _EXTRACTION_PROMPT.format(
                question=question.strip()[:300],
                text=text[:500],
            )
        else:
            prompt = _EXTRACTION_PROMPT_NO_QUESTION.format(text=text[:500])
        response = groq.complete(
            prompt,
            model_name  = "llama-3.1-8b-instant",  
            max_tokens  = 80,
            temperature = 0.0,                     
        )

        if not response.success or not response.output_text:
            logger.debug("Claim extraction Groq call failed: %s", response.error)
            return None

        raw = response.output_text.strip()

        # Model sometimes returns null explicitly
        if raw.lower() in ("null", "none", ""):
            logger.debug("No extractable claim in: %.80s", text)
            return None

        # Strip markdown code fences if model added them
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        if not isinstance(data, dict):
            return None

        subject  = str(data.get("subject",  "")).strip()
        property_ = str(data.get("property", "")).strip()
        value    = str(data.get("value",    "")).strip()

        if not subject or not property_ or not value:
            logger.debug("Claim extraction returned incomplete fields: %s", data)
            return None

        claim = ExtractedClaim(
            subject  = subject,
            property = property_,
            value    = value,
            raw_text = text,
        )
        logger.info(
            "Claim extracted | subject=%s | property=%s | value=%s",
            claim.subject, claim.property, claim.value,
        )
        return claim

    except json.JSONDecodeError as exc:
        logger.debug("Claim extraction JSON parse failed: %s | raw=%s", exc, raw[:100])
        return None
    except Exception as exc:
        logger.warning("Claim extraction error: %s", exc)
        return None
