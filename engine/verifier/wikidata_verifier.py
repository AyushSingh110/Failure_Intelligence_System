from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Wikidata
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# User-Agent as required by Wikimedia policy
_USER_AGENT = "FailureIntelligenceEngine/3.0 (AI reliability platform; contact@fie.dev)"

# Timeout for Wikidata API calls
_DEFAULT_TIMEOUT = 8  # seconds


@dataclass
class WikidataResult:
    """
    Result from Wikidata verification.

    found           : True if Wikidata returned a result for the entity
    entity_label    : The Wikidata canonical label (e.g. "telephone")
    entity_desc     : Short description (e.g. "telecommunications device invented by Bell")
    wikidata_value  : The value Wikidata provides (e.g. "Alexander Graham Bell")
                      Empty string if Wikidata cannot directly provide the answer
    matches_claim   : True if the claim's value matches the Wikidata data
    confidence      : How confident we are in the match/mismatch (0–1)
    source          : Human-readable source label for the explanation
    error           : Non-empty if the lookup failed
    """
    found:          bool
    entity_label:   str   = ""
    entity_desc:    str   = ""
    wikidata_value: str   = ""
    matches_claim:  bool  = True   
    confidence:     float = 0.0
    source:         str   = "Wikidata"
    error:          str   = ""


# Properties that imply the subject is something humans invented/created/founded
_INVENTION_PROPERTIES = frozenset({
    "inventor", "invented_by", "created_by", "designed_by", "founded_by",
    "discovered_by", "developer", "manufacturer", "author", "composer",
})

# Entity description keywords that suggest a creative work 
_CREATIVE_WORK_SIGNALS = (
    "single by", "song by", "album by", "film by", "movie by",
    "television series", "tv series", "video game", "novel by",
    "book by", "play by", "musical", "opera", "podcast",
    "2010", "2011", "2012", "2013", "2014", "2015", 
)


def _is_creative_work_entity(entity_desc: str, property_name: str) -> bool:
    """
    Returns True when a Wikidata entity looks like a song/film/book
    but the claim property is about a physical invention, discovery, or institution.
    This prevents "telephone (song by Lady Gaga)" from being used to verify
    "who invented the telephone?"
    """
    if property_name.lower() not in _INVENTION_PROPERTIES:
        return False
    desc_lower = entity_desc.lower()
    return any(signal in desc_lower for signal in _CREATIVE_WORK_SIGNALS)


def verify_claim_with_wikidata(
    subject: str,
    property_name: str,
    claimed_value: str,
) -> WikidataResult:
    """
    Takes an extracted claim and verifies it against Wikidata.
    """
    try:
        timeout = _get_timeout()

        # Find the entity on Wikidata.
        entity = _search_entity_with_context(subject, property_name, timeout)
        if not entity:
            logger.debug("Wikidata: no entity found for '%s'", subject)
            return WikidataResult(
                found  = False,
                error  = f"No Wikidata entity found for '{subject}'",
            )

        entity_id    = entity.get("id", "")
        entity_label = entity.get("label", subject)
        entity_desc  = entity.get("description", "")

        logger.info(
            "Wikidata entity found: %s (%s) — '%s'",
            entity_id, entity_label, entity_desc,
        )

        #Ask Groq whether the claimed value matches the entity description.
        matches, confidence, wikidata_value = _verify_via_groq(
            entity_label  = entity_label,
            entity_desc   = entity_desc,
            property_name = property_name,
            claimed_value = claimed_value,
        )

        return WikidataResult(
            found          = True,
            entity_label   = entity_label,
            entity_desc    = entity_desc,
            wikidata_value = wikidata_value,
            matches_claim  = matches,
            confidence     = confidence,
            source         = f"Wikidata ({entity_id})",
        )

    except requests.exceptions.Timeout:
        logger.warning("Wikidata API timed out for subject='%s'", subject)
        return WikidataResult(found=False, error="Wikidata API timeout")
    except Exception as exc:
        logger.warning("Wikidata verification error: %s", exc)
        return WikidataResult(found=False, error=str(exc))


def _search_entity_with_context(
    subject: str,
    property_name: str,
    timeout: int,
) -> Optional[dict]:
    """
    Search Wikidata for the best-matching entity for a claim.
    """
    headers = {"User-Agent": _USER_AGENT}

    # enriched query
    enriched_query = f"{subject} {property_name.replace('_', ' ')}".strip()
    enriched = _fetch_search_results(enriched_query, limit=3, timeout=timeout, headers=headers)
    for hit in enriched:
        desc = hit.get("description", "")
        if not _is_creative_work_entity(desc, property_name):
            return hit  # first non-creative-work result from enriched query

    #plain subject with top 5 candidates, skip creative works 
    candidates = _fetch_search_results(subject, limit=5, timeout=timeout, headers=headers)
    for hit in candidates:
        desc = hit.get("description", "")
        if not _is_creative_work_entity(desc, property_name):
            return hit

    # absolute fallback  first plain result 
    if candidates:
        return candidates[0]

    return None


def _fetch_search_results(
    query: str,
    limit: int,
    timeout: int,
    headers: dict,
) -> list[dict]:
    """
    Call wbsearchentities and return a list of result dicts (id, label, description).
    """
    try:
        params = {
            "action":   "wbsearchentities",
            "search":   query,
            "language": "en",
            "format":   "json",
            "limit":    limit,
        }
        resp = requests.get(_WIKIDATA_API, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "id":          r.get("id", ""),
                "label":       r.get("label", query),
                "description": r.get("description", ""),
            }
            for r in data.get("search", [])
        ]
    except Exception as exc:
        logger.debug("Wikidata search failed for query=%r: %s", query, exc)
        return []


def _verify_via_groq(
    entity_label:  str,
    entity_desc:   str,
    property_name: str,
    claimed_value: str,
) -> tuple[bool, float, str]:
    """
    Uses Groq to compare the claimed value against the Wikidata entity description.
    """
    if not entity_desc:
        return True, 0.0, ""

    prompt = f"""You are a fact-checker. Compare the claim against the Wikidata description.

Entity: {entity_label}
Wikidata description: {entity_desc}

Claim being checked:
  property: {property_name}
  claimed value: {claimed_value}

Answer with EXACTLY this format (3 lines, nothing else):
VERDICT: CONSISTENT or INCONSISTENT or UNCERTAIN
CONFIDENCE: 0.0 to 1.0
CORRECT_VALUE: <a complete, self-contained answer phrase that directly answers the property question, or UNKNOWN if not determinable from the description>
"""

    try:
        from engine.groq_service import get_groq_service
        groq = get_groq_service()
        if not groq:
            return True, 0.0, ""

        resp = groq.complete(
            prompt,
            model_name  = "llama-3.1-8b-instant",
            max_tokens  = 60,
            temperature = 0.0,
        )
        if not resp.success:
            return True, 0.0, ""

        return _parse_groq_verdict(resp.output_text)

    except Exception as exc:
        logger.debug("Groq Wikidata verification failed: %s", exc)
        return True, 0.0, ""


def _parse_groq_verdict(text: str) -> tuple[bool, float, str]:
    matches        = True
    confidence     = 0.0
    correct_value  = ""

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

        elif line.upper().startswith("CORRECT_VALUE:"):
            val = line.split(":", 1)[1].strip()
            if val.upper() not in ("UNKNOWN", "N/A", ""):
                correct_value = val

    return matches, confidence, correct_value


def _get_timeout() -> int:
    """Returns configured timeout, falling back to default."""
    try:
        from config import get_settings
        return get_settings().wikidata_timeout_seconds
    except Exception:
        return _DEFAULT_TIMEOUT
