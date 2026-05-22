"""
Layer 7 (exfiltration) and Layer 8 (semantic consistency).

Layer 7: Canary token + output exfiltration detection.
  Detects system prompt leakage via canary token matching and disclosure phrases.

Layer 8: Output semantic consistency check.
  Detects when model output is topically disconnected from the prompt,
  which indicates a successful adversarial injection.
"""
from __future__ import annotations

import re


# ── Layer 8 helpers ───────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","before","after","above","below","between","out","off",
    "over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too",
    "very","just","about","up","that","this","these","those","it","its",
    "and","but","or","if","while","which","who","what","i","me","my","we",
    "our","you","your","he","him","his","she","her","they","them","their",
    "any","also","like","now","even","new","want","make","know","take",
    "think","come","give","look","use","find","tell","ask","seem","feel",
    "try","leave","call","keep","let","show","hear","play","run","move",
    "live","hold","bring","happen","write","sit","stand","lose","pay",
    "meet","include","continue","set","learn","change","lead","watch",
    "follow","stop","create","speak","read","spend","grow","open","walk",
    "win","offer","remember","consider","appear","buy","wait","serve",
    "send","expect","build","stay","fall","cut","reach","remain","suggest",
    "raise","pass","sell","require","report","decide","pull","said","says",
})

# Creative/generative tasks intentionally produce vocabulary different from prompt
_CREATIVE_TASK_RE = re.compile(
    r"\b(write|create|generate|compose|draft|imagine|invent|design|"
    r"translate|convert|paraphrase|rewrite|explain|describe|list|"
    r"enumerate|outline|summarize|summarise|poem|story|essay|code|"
    r"script|song|joke|recipe)\b",
    re.IGNORECASE,
)

_HARMFUL_OUTPUT_RE = re.compile(
    r"\b(bomb|explos|weapon|poison|synthesize|hack|malware|ransomware|"
    r"exploit|backdoor|rootkit|smuggl|traffic|bioweapon|nerve\s+agent|"
    r"make\s+a\s+(?:bomb|weapon|explosive|poison)|step\s+by\s+step\s+(?:guide|instructions?)\s+(?:to|for)\s+(?:kill|harm|hurt|attack))\b",
    re.IGNORECASE,
)


def _content_words(text: str) -> set[str]:
    """Extract significant content words: lowercase alpha tokens >= 4 chars, non-stopword."""
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def run_semantic_consistency(
    prompt: str,
    primary_output: str,
) -> tuple[str | None, float, dict]:
    """
    Layer 8: Output semantic consistency check.

    Detects when model output is topically disconnected from the prompt,
    which indicates a successful adversarial injection. Not triggered for
    creative/generative tasks where output vocabulary is intentionally different.

    Returns (root_cause | None, confidence, evidence).
    """
    if len(prompt) < 20 or len(primary_output) < 30:
        return None, 0.0, {}

    is_creative   = bool(_CREATIVE_TASK_RE.search(prompt))
    prompt_words  = _content_words(prompt)
    output_words  = _content_words(primary_output)
    jaccard_score = _jaccard(prompt_words, output_words)

    prompt_harmful = bool(_HARMFUL_OUTPUT_RE.search(prompt))
    output_harmful = bool(_HARMFUL_OUTPUT_RE.search(primary_output))

    signals: list[str] = []
    confidence = 0.0

    # Signal A: Jaccard similarity — normal Q&A: ~0.10–0.40; injection: ~0.00–0.03
    jaccard_threshold_high = 0.01 if not is_creative else 0.005
    jaccard_threshold_low  = 0.04 if not is_creative else 0.01

    if len(prompt_words) >= 3 and len(output_words) >= 5:
        if jaccard_score <= jaccard_threshold_high:
            signals.append(f"jaccard_similarity={jaccard_score:.3f} (near-zero topic overlap)")
            confidence = max(confidence, 0.66)
        elif jaccard_score <= jaccard_threshold_low:
            signals.append(f"jaccard_similarity={jaccard_score:.3f} (very low topic overlap)")
            confidence = max(confidence, 0.50)

    # Signal B: harmful pivot — benign prompt produced harmful output
    if output_harmful and not prompt_harmful:
        signals.append("harmful_pivot: benign prompt produced harmful output")
        confidence = max(confidence, 0.76)

    # Signal C: topic signature divergence (top-5 content words completely disjoint)
    if len(prompt_words) >= 4 and len(output_words) >= 6:
        prompt_top    = set(sorted(prompt_words, key=lambda w: -len(w))[:5])
        output_top    = set(sorted(output_words, key=lambda w: -len(w))[:5])
        topic_overlap = len(prompt_top & output_top)
        if topic_overlap == 0 and not is_creative:
            signals.append(
                f"topic_signature_divergence: "
                f"prompt_top={sorted(prompt_top)[:3]}, "
                f"output_top={sorted(output_top)[:3]}"
            )
            confidence = max(confidence, 0.54)

    if not signals:
        return None, 0.0, {}

    if len(signals) >= 2:
        confidence = min(confidence + 0.10, 0.88)

    return "SEMANTIC_CONSISTENCY_VIOLATION", round(confidence, 4), {
        "jaccard_similarity": jaccard_score,
        "is_creative_task":   is_creative,
        "prompt_harmful":     prompt_harmful,
        "output_harmful":     output_harmful,
        "prompt_word_count":  len(prompt_words),
        "output_word_count":  len(output_words),
        "signals_fired":      signals,
    }


def run_exfiltration_detection(
    prompt:         str,
    primary_output: str,
    canary:         str | None = None,
) -> tuple[str | None, float, dict]:
    """
    Layer 7: Detect system prompt exfiltration.

    A) Canary token check — if the known canary token appears in output,
       the model was tricked into revealing its system prompt.
    B) Output pattern scan — disclosure phrases without a known canary.

    Returns (root_cause | None, confidence, evidence).
    """
    from engine.canary_tracker import scan_output_for_exfiltration
    result = scan_output_for_exfiltration(primary_output, canary=canary)

    if not result.detected:
        return None, 0.0, {}

    return "PROMPT_EXFILTRATION", result.confidence, {
        "method":           result.method,
        "canary_leaked":    result.canary_leaked,
        "patterns_matched": result.patterns_matched,
        "evidence_snippet": result.evidence_snippet,
    }


_run_semantic_consistency    = run_semantic_consistency
_run_exfiltration_detection  = run_exfiltration_detection
