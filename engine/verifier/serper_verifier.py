from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_SERPER_API_URL = "https://google.serper.dev/search"
_DEFAULT_TIMEOUT = 8


@dataclass
class SerperResult:
    """
    Result from the Serper real-time search verification.

    found            : True if Serper returned useful results
    skip             : True if Serper is not configured (no API key)
    search_answer    : Best snippet or knowledge graph answer from search
    grounded_answer  : Answer after Groq verifies against search results
    matches_output   : True if the model output agrees with search results
    confidence       : How confident we are in the match/mismatch (0–1)
    source           : Source label for explainability
    error            : Non-empty if search failed
    """
    found:           bool
    skip:            bool  = False
    search_answer:   str   = ""
    grounded_answer: str   = ""
    matches_output:  bool  = True
    confidence:      float = 0.0
    source:          str   = "Serper (Google Search)"
    error:           str   = ""


def verify_with_serper(prompt: str, primary_output: str) -> SerperResult:
    settings = _get_settings()
    if not settings:
        return SerperResult(found=False, skip=True, error="Settings unavailable")

    if not settings.serper_enabled or not settings.serper_api_key:
        logger.debug(
            "Serper not configured. Add SERPER_API_KEY and SERPER_ENABLED=true to .env"
        )
        return SerperResult(
            found = False,
            skip  = True,
            error = "Serper not configured — set SERPER_API_KEY in .env to enable real-time verification",
        )

    try:
        search_data = _search_google(
            query      = prompt,
            api_key    = settings.serper_api_key,
            timeout    = settings.serper_timeout_seconds,
        )

        if not search_data:
            return SerperResult(found=False, error="Serper returned no results")

        # Extract the best answer from the search results
        search_answer = _extract_best_answer(search_data)
        if not search_answer:
            return SerperResult(found=False, error="Could not extract answer from search results")

        logger.info("Serper search answer: %.100s", search_answer)

        # Ask Groq to compare the primary output against the search answer
        matches, confidence, grounded = _verify_via_groq(
            primary_output = primary_output,
            search_answer  = search_answer,
            prompt         = prompt,
        )

        return SerperResult(
            found           = True,
            search_answer   = search_answer,
            grounded_answer = grounded or search_answer,
            matches_output  = matches,
            confidence      = confidence,
            source          = "Google Search via Serper.dev",
        )

    except requests.exceptions.Timeout:
        logger.warning("Serper API timed out for query: %.80s", prompt)
        return SerperResult(found=False, error="Serper API timeout")
    except Exception as exc:
        logger.warning("Serper verification error: %s", exc)
        return SerperResult(found=False, error=str(exc))


def _search_google(query: str, api_key: str, timeout: int) -> Optional[dict]:
    """Calls the Serper API and returns the raw JSON response."""
    headers = {
        "X-API-KEY":    api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 5}

    resp = requests.post(_SERPER_API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_best_answer(data: dict) -> str:
    """
    Extracts the most relevant text from Serper search results.
    """
    #Direct answer box
    answer_box = data.get("answerBox", {})
    if answer_box:
        for key in ("answer", "snippet", "title"):
            val = answer_box.get(key, "")
            if val:
                return str(val).strip()

    # Knowledge graph
    kg = data.get("knowledgeGraph", {})
    if kg:
        desc = kg.get("description", "")
        if desc:
            return str(desc).strip()

    # First organic result snippet
    organic = data.get("organic", [])
    if organic:
        snippet = organic[0].get("snippet", "")
        if snippet:
            return str(snippet).strip()

    return ""


def _verify_via_groq(
    primary_output: str,
    search_answer:  str,
    prompt:         str,
) -> tuple[bool, float, str]:
    """
    Uses Groq to compare the primary output against the live search result.
    """
    verification_prompt = f"""You are a fact-checker comparing a model output to a live search result.

Original question: {prompt[:200]}

Model output: {primary_output[:300]}

Live search result (from Google): {search_answer[:400]}

Answer with EXACTLY this format (3 lines):
VERDICT: CONSISTENT or INCONSISTENT or UNCERTAIN
CONFIDENCE: 0.0 to 1.0
GROUNDED_ANSWER: <most accurate answer based on the search result>"""

    try:
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if not groq:
            # Fallback: if output and search answer share significant words → consistent
            return _heuristic_match(primary_output, search_answer)

        resp = groq.complete(
            verification_prompt,
            model_name  = "llama-3.1-8b-instant",
            max_tokens  = 100,
            temperature = 0.0,
        )
        if not resp.success:
            return _heuristic_match(primary_output, search_answer)

        return _parse_groq_verdict(resp.output_text)

    except Exception as exc:
        logger.debug("Groq Serper verification failed: %s", exc)
        return _heuristic_match(primary_output, search_answer)


def _parse_groq_verdict(text: str) -> tuple[bool, float, str]:
    """Parses the 3-line Groq verdict response."""
    matches         = True
    confidence      = 0.0
    grounded_answer = ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
            matches = (verdict != "INCONSISTENT")
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        elif line.upper().startswith("GROUNDED_ANSWER:"):
            grounded_answer = line.split(":", 1)[1].strip()

    return matches, confidence, grounded_answer


def _heuristic_match(output: str, search: str) -> tuple[bool, float, str]:
    """Simple word-overlap fallback when Groq is unavailable."""
    import re
    def words(t: str) -> set:
        return set(re.findall(r'\b[a-zA-Z]{4,}\b', t.lower()))
    o_words = words(output)
    s_words = words(search)
    if not o_words or not s_words:
        return True, 0.0, search
    overlap = len(o_words & s_words) / max(len(s_words), 1)
    matches = overlap > 0.25
    return matches, round(min(overlap, 0.7), 3), search


def _get_settings():
    try:
        from config import get_settings
        return get_settings()
    except Exception:
        return None
