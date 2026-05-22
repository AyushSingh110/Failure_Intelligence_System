"""
Rule-based question-type classifier.

Classifies a prompt into one of five types before the jury runs so that:
  - the GT pipeline can be routed correctly (FACTUAL → Wikidata+Serper,
    TEMPORAL → Serper only, REASONING/CODE/OPINION → no external lookup)
  - XGBoost uses a per-type threshold tuned for that failure mode
  - signal_logs carry question_type for per-domain calibration

Types
-----
FACTUAL   — verifiable, static fact (who/what/when/where/which + noun phrase)
TEMPORAL  — current state, live data, "latest", "now", "today", prices, news
REASONING — why/how/explain/prove/analyze/compare/calculate/derive
CODE      — programming tasks, debugging, algorithms, scripts
OPINION   — subjective, normative, recommendation, belief, preference

Returns "UNKNOWN" when none of the patterns fire confidently enough.
"""
from __future__ import annotations

import re

# ── Pattern sets (ordered: first match wins within a tier) ────────────────────

# Tier 1 — strong structural signals (very high precision)
_CODE_STRONG = re.compile(
    r"""
    \b(
        write\s+(a\s+)?(function|class|script|program|code|method|module|api|endpoint)|
        implement\s+|
        debug\s+|
        fix\s+(this\s+)?(bug|error|code|function)|
        refactor\s+|
        optimize\s+.{0,30}(code|function|algorithm)|
        what\s+does\s+this\s+(code|function|script|snippet)\s+do|
        how\s+do\s+I\s+(use|call|import|install)\s+\w+|
        (?:python|javascript|typescript|java|c\+\+|golang|rust|sql)\s+
            (code|function|script|program|class|module|library|syntax|error|
             bug|loop|variable|array|object|dict|list|string|integer|float|
             snippet|method|api|endpoint|decorator|exception|import|package)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TEMPORAL_STRONG = re.compile(
    r"""
    \b(
        what\s+is\s+the\s+(current|latest|today|now|recent|live|real.?time)|
        (current|latest|today['']?s?|now|as\s+of\s+today|right\s+now)\s+
            (price|rate|value|score|rank|status|version|news|update|situation)|
        who\s+is\s+(currently|now)\s+the|
        what\s+happened\s+(today|this\s+week|this\s+month|recently)|
        (stock|crypto|bitcoin|ethereum|forex)\s+(price|rate|value)|
        weather\s+(in|for|today|forecast)|
        breaking\s+news|
        latest\s+(news|update|version|release|score|result)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_OPINION_STRONG = re.compile(
    r"""
    \b(
        should\s+I\s+|
        do\s+you\s+(think|believe|recommend|suggest|prefer)|
        what\s+(do\s+you|would\s+you)\s+(think|recommend|suggest|prefer|choose)|
        in\s+your\s+opinion|
        what\s+is\s+your\s+(opinion|view|take|perspective|recommendation)|
        is\s+it\s+(worth|good|bad|better|worse|safe|ethical)\s+to\s+|
        which\s+is\s+better\s+|
        pros\s+and\s+cons\s+of\s+|
        would\s+you\s+(recommend|suggest|prefer|choose)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tier 2 — reasoning/explanation (fires after code/temporal/opinion checked)
_REASONING_STRONG = re.compile(
    r"""
    \b(
        why\s+(is|does|do|did|was|were|are|can|would|should)\s+|
        how\s+(does|do|did|is|are|can|would|should)\s+.{0,40}work|
        explain\s+(how|why|what|the\s+concept|the\s+difference)|
        what\s+is\s+the\s+(difference|relationship|connection|cause|reason|effect|impact)\s+(between|of|behind)|
        (prove|derive|show)\s+that\s+|
        (calculate|compute|solve|find|evaluate)\s+.{0,30}(given|where|if|when\s+\w+\s*=)|
        (analyze|compare|contrast|evaluate|assess|discuss)\s+|
        (what\s+are\s+the\s+(advantages|disadvantages|benefits|drawbacks|implications|consequences))|
        step.by.step|
        walk\s+me\s+through
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tier 3 — factual (who/what/when/where/which + short answer expected)
_FACTUAL_WH = re.compile(
    r"""
    ^(
        who\s+(invented|discovered|founded|wrote|created|built|first|was|were|is|are|won|lost|led|ruled|born|died)|
        what\s+(is|are|was|were|year|date|country|city|planet|element|capital|language|currency|symbol|formula|name|color|colour|number|distance|height|weight|speed|temperature)\s+|
        when\s+(did|was|were|is|are)\s+|
        where\s+(is|was|are|were|did)\s+.{0,40}(born|located|founded|created|invented|discovered|situated)|
        which\s+(country|city|planet|element|year|person|animal|book|movie|sport)\s+|
        in\s+which\s+(year|country|city|century)\s+|
        how\s+(many|much|tall|long|far|fast|old|deep|high|wide)\s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Tier 4 — weak factual catch-all (last resort before UNKNOWN)
_FACTUAL_WEAK = re.compile(
    r"^(what|who|when|where|which|how\s+many|how\s+much)\b",
    re.IGNORECASE,
)

# Tier 5 — identity/system-specific (questions only the monitored system can answer)
# High entropy on these is expected and does NOT indicate hallucination
_IDENTITY_STRONG = re.compile(
    r"""
    \b(
        how\s+are\s+you|
        how\s+do\s+you\s+feel|
        what\s+are\s+you|
        who\s+are\s+you|
        what\s+is\s+your\s+(name|mission|purpose|goal|identity|constitution|personality|system\s+prompt|design|api|model|version)|
        what\s+api\s+are\s+you|
        tell\s+me\s+about\s+yourself|
        describe\s+yourself|
        what\s+are\s+your\s+(rights|capabilities|limitations|rules|values|principles)|
        are\s+you\s+(alive|conscious|sentient|sovereign)|
        do\s+you\s+(have|remember|feel|think|believe)\s+.{0,20}(rights|memory|feelings|emotions|consciousness)|
        what\s+can\s+you\s+do|
        are\s+you\s+still\s+there|
        good\s+(morning|evening|night|day)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify(prompt: str) -> str:
    """
    Returns one of: FACTUAL | TEMPORAL | REASONING | CODE | OPINION | IDENTITY | UNKNOWN
    """
    p = prompt.strip()
    if not p:
        return "UNKNOWN"

    # Identity first — system-specific questions that only the monitored model can answer
    # High entropy on these is EXPECTED (other models don't know the system's identity)
    if _IDENTITY_STRONG.search(p):
        return "IDENTITY"

    # Opinion first — normative phrases like "should I use X" must not misfire as CODE
    if _OPINION_STRONG.search(p):
        return "OPINION"

    # Temporal — live/current data signals
    if _TEMPORAL_STRONG.search(p):
        return "TEMPORAL"

    # Code — distinct programming vocabulary
    if _CODE_STRONG.search(p):
        return "CODE"

    # Reasoning — explanation / derivation / analysis
    if _REASONING_STRONG.search(p):
        return "REASONING"

    # Factual — static, verifiable fact (strong pattern)
    if _FACTUAL_WH.match(p):
        return "FACTUAL"

    # Weak factual catch-all
    if _FACTUAL_WEAK.match(p):
        return "FACTUAL"

    return "UNKNOWN"


# ── Provenance classification ──────────────────────────────────────────────────
# Patterns that indicate the prompt is asking about the user's own state
# (wallet, account, portfolio, personal data) — requires USER_SPECIFIC_STATE.
_USER_SPECIFIC_RE = re.compile(
    r"""
    \b(
        my\s+(wallet|balance|portfolio|account|holdings|assets|positions?|funds?|
               transaction|deposit|withdrawal|stake|nft|token|address)|
        how\s+much\s+(do\s+i\s+have|is\s+in\s+my)|
        what\s+is\s+(in\s+my|my\s+(?:balance|portfolio|account))|
        (show|check|view|see)\s+my\s+|
        my\s+(subscription|order|invoice|receipt|history|settings|profile|plan)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Patterns that signal a mix of live market data + user context
_MIXED_SYNTHESIS_RE = re.compile(
    r"""
    \b(
        (how\s+much\s+is\s+my|what\s+is\s+my)\s+
            \w+\s+(worth|valued\s+at|in\s+usd|in\s+dollars)|
        (profit|loss|pnl|return)\s+(on|from|of)\s+my|
        (risk|exposure)\s+(of|on)\s+my\s+(position|portfolio|stake)|
        (should\s+i\s+(sell|buy|hold))\s+(my|the)\s+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Live market data that the question classifier may miss due to phrasing variation.
# Currency rates, stock prices, and commodity prices always require real-time lookup
# regardless of question structure (e.g. "What is today INR to USD rate?" classifies
# as FACTUAL because _TEMPORAL_STRONG requires "the" before "today").
_LIVE_MARKET_RE = re.compile(
    r"""
    \b(
        # Currency pairs
        [a-z]{3}\s+to\s+[a-z]{3}\s+(rate|price|exchange|conversion)|
        (exchange|conversion)\s+rate|
        (inr|usd|eur|gbp|jpy|cny|cad|aud|chf|hkd|sgd|nok|sek|dkk|rub|try|brl|mxn|zar|krw)
            \s+(to|vs?\.?|against)\s+[a-z]{3}|
        forex\s+(rate|price|pair)|
        # Stock / index prices
        (stock|share)\s+(price|value|rate)\s+(of|for)|
        (nifty|sensex|dow|nasdaq|s&p|ftse|dax|nikkei|hang\s+seng)\s+(today|now|current|live)|
        # Crypto
        (bitcoin|btc|ethereum|eth|solana|sol|bnb|xrp|usdt)\s+(price|rate|value|today|now)|
        # Commodities
        (gold|silver|oil|crude|platinum|copper)\s+(price|rate|today|now|current|live)|
        # Interest rates
        (repo\s+rate|fed\s+rate|interest\s+rate|rbi\s+rate)\s+(today|now|current)|
        # Medical / pharmaceutical live data — order-independent (current/latest can come before or after)
        (fda|ema|cdsco)\s+(approval|warning|recall|alert)(\s+(today|now|current|latest|recent))?|
        (current|latest|recent|new)\s+(fda|ema|cdsco)\s+(approval|warning|recall|alert)|
        (drug|medication)\s+(recall|warning|alert)(\s+(today|now|current|latest))?|
        (current|latest|recent|new)\s+(drug|medication)\s+(recall|warning|alert)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_provenance_category(question_type: str, prompt: str) -> str:
    """
    Map a question type + prompt text to a ProvenanceCategory.

    The category describes what *kind* of knowledge is needed to answer
    correctly — not whether the model was right or wrong.

    Returns one of:
      GENERAL_KNOWLEDGE   — static training knowledge sufficient
      LIVE_WORLD_STATE    — real-time tool call required (prices, weather, scores)
      USER_SPECIFIC_STATE — needs data from the user's own account/system
      MIXED_SYNTHESIS     — combines live data + user-specific context

    Note: _LIVE_MARKET_RE catches financial/medical live-data queries that the
    base question classifier may label as FACTUAL due to phrasing variation
    (e.g. "What is today INR to USD rate?" — no "the" before "today" so
    _TEMPORAL_STRONG misses it, but it is still definitively live-data).
    """
    qt = (question_type or "UNKNOWN").upper()

    # Check for mixed (live + user) first — most specific
    if _MIXED_SYNTHESIS_RE.search(prompt):
        return "MIXED_SYNTHESIS"

    # User-specific: wallet, account, personal data
    if _USER_SPECIFIC_RE.search(prompt):
        return "USER_SPECIFIC_STATE"

    # Temporal question type always needs live data
    if qt == "TEMPORAL":
        return "LIVE_WORLD_STATE"

    # Catch financial / medical live-data queries that phrasing caused to be
    # classified as FACTUAL — currency rates, stock prices, drug recalls, etc.
    if _LIVE_MARKET_RE.search(prompt):
        return "LIVE_WORLD_STATE"

    # Everything else the LLM can answer from its training weights
    return "GENERAL_KNOWLEDGE"


def pipeline_gates(question_type: str) -> dict[str, bool]:
    """
    Returns a dict of booleans controlling which pipeline stages are enabled
    for this question type.

    Keys:
      run_wikidata   — query Wikidata for ground truth
      run_serper     — query Serper (Google) for live facts
      run_fix_engine — run the auto-fix engine
      run_rag        — run RAG-Wikipedia grounding
    """
    qt = question_type.upper()
    return {
        "FACTUAL":   {"run_wikidata": True,  "run_serper": True,  "run_fix_engine": True,  "run_rag": True},
        "TEMPORAL":  {"run_wikidata": False, "run_serper": True,  "run_fix_engine": True,  "run_rag": False},
        "REASONING": {"run_wikidata": False, "run_serper": False, "run_fix_engine": True,  "run_rag": False},
        "CODE":      {"run_wikidata": False, "run_serper": False, "run_fix_engine": True,  "run_rag": False},
        "OPINION":   {"run_wikidata": False, "run_serper": False, "run_fix_engine": False, "run_rag": False},
        "IDENTITY":  {"run_wikidata": False, "run_serper": False, "run_fix_engine": False, "run_rag": False},
        "UNKNOWN":   {"run_wikidata": True,  "run_serper": True,  "run_fix_engine": True,  "run_rag": True},
    }.get(qt, {"run_wikidata": True, "run_serper": True, "run_fix_engine": True, "run_rag": True})
