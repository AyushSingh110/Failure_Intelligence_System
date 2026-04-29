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


def classify(prompt: str) -> str:
    """
    Returns one of: FACTUAL | TEMPORAL | REASONING | CODE | OPINION | UNKNOWN
    """
    p = prompt.strip()
    if not p:
        return "UNKNOWN"

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
        "UNKNOWN":   {"run_wikidata": True,  "run_serper": True,  "run_fix_engine": True,  "run_rag": True},
    }.get(qt, {"run_wikidata": True, "run_serper": True, "run_fix_engine": True, "run_rag": True})
