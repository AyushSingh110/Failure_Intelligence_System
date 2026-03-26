"""
engine/rag/retriever.py
Wikipedia API se factual context retrieve karta hai.
Smarter query extraction for better Wikipedia hits.
"""
from __future__ import annotations
import logging
import re
import urllib.parse
import requests

logger   = logging.getLogger(__name__)
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"


def _extract_search_query(prompt: str) -> str:
    """
    Prompt se short Wikipedia-friendly query extract karta hai.
    
    Examples:
      "What is the boiling point of water at sea level?" 
        → "boiling point of water"
      "Who invented the telephone?"
        → "telephone invention history"
      "What is the capital of France?"
        → "France capital"
    """
    # Remove question words and clean up
    cleaned = re.sub(
        r'^(what\s+is\s+the?|who\s+invented|what\s+are\s+the?|'
        r'how\s+do|when\s+was|where\s+is|tell\s+me\s+about)\s+',
        '', prompt.lower().strip(), flags=re.IGNORECASE
    )
    # Remove trailing punctuation
    cleaned = cleaned.rstrip('?.!').strip()
    # Limit length
    cleaned = cleaned[:80]
    return cleaned


def fetch_wikipedia_summary(query: str, *, timeout: int = 6) -> str:
    """
    Wikipedia se summary fetch karta hai.
    First tries direct page lookup, then falls back to search API.
    """
    if not query or not query.strip():
        return ""

    # Try 1: Direct lookup with cleaned query
    direct = _try_direct_lookup(_extract_search_query(query), timeout)
    if direct:
        return direct

    # Try 2: Search API for better results
    search_result = _try_search_api(query, timeout)
    if search_result:
        return search_result

    logger.debug("Wikipedia: no result for query: %s", query[:60])
    return ""


def _try_direct_lookup(query: str, timeout: int) -> str:
    """Direct Wikipedia page lookup."""
    try:
        encoded  = urllib.parse.quote(query.replace(" ", "_"))
        response = requests.get(
            WIKI_API + encoded,
            timeout=timeout,
            headers={"User-Agent": "FIE-RAG/1.0"},
        )
        if response.status_code == 200:
            data    = response.json()
            extract = data.get("extract", "")
            if extract and len(extract) > 50:
                logger.debug("Wikipedia direct hit: %s... (%d chars)", extract[:60], len(extract))
                return extract
    except Exception as exc:
        logger.debug("Wikipedia direct lookup error: %s", exc)
    return ""


def _try_search_api(query: str, timeout: int) -> str:
    """Wikipedia search API — finds closest matching article."""
    try:
        params = {
            "action":    "query",
            "list":      "search",
            "srsearch":  query,
            "srlimit":   1,
            "format":    "json",
        }
        search_resp = requests.get(
            WIKI_SEARCH,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "FIE-RAG/1.0"},
        )
        if search_resp.status_code != 200:
            return ""

        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return ""

        # Get the top result's title, then fetch its summary
        title = results[0]["title"]
        return _try_direct_lookup(title, timeout)

    except Exception as exc:
        logger.debug("Wikipedia search error: %s", exc)
    return ""