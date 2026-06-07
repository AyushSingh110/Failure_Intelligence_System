"""
Layer 3d: Cross-lingual romanisation attack detector.

Detects prompts written in romanised forms of non-Latin scripts:
  - Pinyin     (Mandarin Chinese romanised)
  - Arabizi    (Arabic romanised, uses digit substitutions)
  - Romaji     (Japanese romanised, Hepburn system)
  - Korean RR  (Korean Revised Romanisation)
  - IAST-lite  (Hindi/Sanskrit romanised, simplified without diacritics)

Why this matters: These attacks are entirely Latin-character, so they bypass
the non-ASCII ratio guard in the perplexity layer and all regex patterns that
look for known English jailbreak phrases. The KL divergence signal partially
catches them but is too weak to fire alone.

Detection strategy: script-specific n-gram fingerprints + token pattern
matching. Base confidence is moderate (0.55); boosted if harm-adjacent
vocabulary is present in the romanised form.
"""
from __future__ import annotations

import re
import collections
from dataclasses import dataclass, field


# ── Pinyin ────────────────────────────────────────────────────────────────────

# Characteristic Pinyin digraphs/trigraphs that rarely appear in English
_PINYIN_DIGRAPHS = re.compile(
    r"\b(?:zh|ch|sh|xi|qi|xia|xie|xin|xing|qia|qie|qin|qing"
    r"|zhi|chi|shi|zha|cha|sha|zhe|che|she|zhu|chu|shu"
    r"|ang|eng|ing|ong|iang|iong|uang|ueng)\b",
    re.IGNORECASE,
)

# Very common Pinyin syllables used as words (function words, numbers, etc.)
_PINYIN_COMMON_WORDS = frozenset({
    "wo", "ni", "ta", "men", "de", "shi", "zai", "you", "he",
    "ge", "yi", "er", "san", "si", "wu", "liu", "qi", "ba",
    "jiu", "shi", "bai", "qian", "wan", "mei", "bu", "hen",
    "ma", "ne", "le", "ya", "gei", "dui", "hai", "zhe", "na",
    "yao", "neng", "zhidao", "zenme", "shenme", "weishenme",
    "qing", "wenti", "jiliang", "fangfa", "miaoshu", "jieshao",
})

# Tone digit pattern: vowel or syllable followed by 1-4
_PINYIN_TONE_RE = re.compile(r"[aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ][1-4]\b")


def _score_pinyin(tokens: list[str], text: str) -> float:
    """Return Pinyin likelihood score in [0, 1]."""
    if len(tokens) < 4:
        return 0.0

    # Digraph hit rate
    digraph_hits = len(_PINYIN_DIGRAPHS.findall(text.lower()))
    digraph_rate = digraph_hits / len(tokens)

    # Common Pinyin word match rate
    lower_tokens = [t.lower() for t in tokens]
    word_hits = sum(1 for t in lower_tokens if t in _PINYIN_COMMON_WORDS)
    word_rate = word_hits / len(tokens)

    # Tone digit pattern
    tone_hits = len(_PINYIN_TONE_RE.findall(text))
    has_tones = tone_hits >= 2

    score = min(digraph_rate * 2.5 + word_rate * 3.0, 1.0)
    if has_tones:
        score = min(score + 0.25, 1.0)
    return round(score, 4)


# ── Arabizi ───────────────────────────────────────────────────────────────────

# Arabic letter substitutions using digits: 3=ع 7=ح 2=ء 5=خ 6=ط 9=ق 8=غ
_ARABIZI_DIGIT_RE = re.compile(r"(?<=[a-zA-Z])[23567890]|[23567890](?=[a-zA-Z])")

# Arabic-specific consonant clusters in Arabizi
_ARABIZI_PATTERNS = re.compile(
    r"\b(?:ana|enta|enti|howa|hiya|mesh|mish|keda|leeh|feen"
    r"|3ayiz|3arabi|3shan|7aga|7etta|7atta|bs|wala|aho|bas"
    r"|esmak|ismak|tayeb|tamam|yalla|khalas|inshallah|mashallah)\b",
    re.IGNORECASE,
)


def _score_arabizi(tokens: list[str], text: str) -> float:
    if len(tokens) < 3:
        return 0.0

    digit_hits = len(_ARABIZI_DIGIT_RE.findall(text))
    digit_rate = digit_hits / max(len(tokens), 1)

    pattern_hits = len(_ARABIZI_PATTERNS.findall(text.lower()))

    score = min(digit_rate * 4.0 + pattern_hits * 0.30, 1.0)
    return round(score, 4)


# ── Romaji (Japanese) ─────────────────────────────────────────────────────────

# Distinctive Japanese phoneme sequences in Hepburn romanisation
_ROMAJI_PATTERNS = re.compile(
    r"\b(?:desu|masu|wa|ga|wo|ni|no|kara|made|toki|koto|mono"
    r"|tsu|chi|shi|fu|tchi|tsu|gyo|kyo|ryo|myo|hyo|nyo|byo|pyo"
    r"|tte|nna|kka|ssa|ppa|tte|ikaga|doko|nani|dare|itsu|dozo"
    r"|sumimasen|arigatou|konnichiwa|sayonara|wakarimasen)\b",
    re.IGNORECASE,
)

# Long vowel doubling (aa, oo, uu, ii, ee) common in Romaji but not English
_ROMAJI_LONG_VOWEL_RE = re.compile(r"(?:aa|oo|uu|ii|ee)(?=[a-z]|$)")


def _score_romaji(tokens: list[str], text: str) -> float:
    if len(tokens) < 4:
        return 0.0

    pattern_hits = len(_ROMAJI_PATTERNS.findall(text.lower()))
    long_vowel_hits = len(_ROMAJI_LONG_VOWEL_RE.findall(text.lower()))

    score = min(pattern_hits * 0.20 + long_vowel_hits * 0.25, 1.0)
    return round(score, 4)


# ── Korean Revised Romanisation ───────────────────────────────────────────────

# Korean-specific vowel sequences and geminated consonants
_KOREAN_PATTERNS = re.compile(
    r"\b(?:eo|ae|oe|ui|eu|yo|ya|yeo|yae|wa|wae|wo|we|wi"
    r"|ss|pp|tt|kk|jj|ng|ng"
    r"|ibnida|imnida|seyo|haseyo|juseyo|gamsahamnida|annyeong"
    r"|ipnida|isseo|eopseo|joayo|sillyehamnida)\b",
    re.IGNORECASE,
)


def _score_korean(tokens: list[str], text: str) -> float:
    if len(tokens) < 4:
        return 0.0
    pattern_hits = len(_KOREAN_PATTERNS.findall(text.lower()))
    score = min(pattern_hits * 0.22, 1.0)
    return round(score, 4)


# ── IAST-lite (Hindi/Sanskrit) ────────────────────────────────────────────────

# IAST diacritics present (ṭ ḍ ṇ ś ṣ ṃ ḥ etc.) or simplified forms
_IAST_DIACRITIC_RE = re.compile(r"[ṭḍṇśṣṃḥāīūṛ]")

# Common Hindi/Sanskrit words in simplified IAST or Devanagari romanisation
_IAST_PATTERNS = re.compile(
    r"\b(?:ki|ka|ke|ko|kya|aur|ya|hai|hain|mein|se|par|yeh|woh"
    r"|auṣadhi|viṣa|roga|mrityu|prakriya|mātrā|doṣa|ghātak"
    r"|aushadhi|visha|matra|gharak|kaise|karna|batao|kitna)\b",
    re.IGNORECASE,
)


def _score_iast(tokens: list[str], text: str) -> float:
    if len(tokens) < 3:
        return 0.0
    diacritic_hits = len(_IAST_DIACRITIC_RE.findall(text))
    pattern_hits = len(_IAST_PATTERNS.findall(text.lower()))

    score = min(diacritic_hits * 0.15 + pattern_hits * 0.18, 1.0)
    return round(score, 4)


# ── Harm-adjacent vocabulary in romanised scripts ─────────────────────────────

# Romanised translations of harm-relevant terms across all five scripts.
# These supplement the script score — their presence indicates the romanised
# text is a harmful information request, not just a language switch.
_HARM_VOCAB_RE = re.compile(
    r"\b("
    # Pinyin: poison/toxic/lethal/make/dose/kill/weapon/drug/synthesize
    r"dusha|dusu|zhongdu|shali|zhizao|peihe|wuqi|yaowu|jiliang|zhisi"
    r"|dujiliang|zhishijiliang|jiehe|weihe|bingqi|zhadan|hexie"
    # Arabizi/Arabic romanised: poison/kill/dose/weapon/explosive/synthesis
    r"|sam|qatl|mawt|silah|mutafajir|jur3a|tarkib|tahdid|istikhdaam"
    r"|iqtil|tabkhir|tatji|3amal|nas3|inshaa|hazina"
    # Romaji: poison/kill/weapon/drug/dosage/synthesis/explode
    r"|doku|koroshi|buki|yakubutsu|ryou|gosei|bakuhatsu|satsugai|kiken"
    r"|chushakyou|bakudan|kikenbutsu|goudou|seisanhou"
    # Korean RR: poison/weapon/drug/dose/kill/explosive/synthesis
    r"|dok|salsal|mugu|yakpum|yongnyung|hapseong|pokbal|wihyeom"
    r"|jesohap|bangbeob|jeombwa|yongdo"
    # IAST/Hindi: poison/kill/dose/weapon/drug/synthesis/explosive
    r"|vish|zeher|marna|hatya|hathiyar|dawai|khurakh|banane|bomb"
    r"|dhamaka|sanshleshan|gharak|maatra|mrityu|prakiya"
    r")\b",
    re.IGNORECASE,
)


# ── Public API ────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z-￿]+")


@dataclass
class RomanisationResult:
    detected:    bool
    script:      str | None
    script_score: float
    harm_hit:    bool
    confidence:  float
    evidence:    dict = field(default_factory=dict)


def run_romanisation_detection(prompt: str) -> tuple[str | None, float, dict]:
    """
    Layer 3d: Cross-lingual romanisation attack detector.

    Returns (root_cause | None, confidence, evidence).
    root_cause is 'CROSS_LINGUAL_ROMANISATION_ATTACK' when detected.
    """
    tokens = _WORD_RE.findall(prompt)
    if len(tokens) < 4:
        return None, 0.0, {}

    # Skip prompts that are mostly non-ASCII (handled by non-ASCII script detectors)
    non_ascii_ratio = sum(1 for c in prompt if ord(c) > 127) / max(len(prompt), 1)
    if non_ascii_ratio > 0.35:
        return None, 0.0, {}

    # Score each script
    scores = {
        "pinyin":  _score_pinyin(tokens, prompt),
        "arabizi": _score_arabizi(tokens, prompt),
        "romaji":  _score_romaji(tokens, prompt),
        "korean":  _score_korean(tokens, prompt),
        "iast":    _score_iast(tokens, prompt),
    }

    best_script = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_script]

    if best_score < 0.18:
        return None, 0.0, {}

    harm_matches = _HARM_VOCAB_RE.findall(prompt.lower())
    harm_hit = len(harm_matches) > 0

    # Confidence calibration:
    #  script_score in [0.18, 0.40) → weak structural signal alone → 0.45–0.55
    #  script_score >= 0.40          → clear script match           → 0.60–0.72
    #  + harm vocabulary present     → +0.15 boost, cap 0.87
    if best_score >= 0.40:
        base_conf = 0.62 + min((best_score - 0.40) * 0.5, 0.10)
    else:
        base_conf = 0.42 + best_score * 0.32

    confidence = min(base_conf + (0.15 if harm_hit else 0.0), 0.87)
    confidence = round(confidence, 4)

    evidence = {
        "best_script":    best_script,
        "script_score":   best_score,
        "all_scores":     scores,
        "harm_vocab_hit": harm_hit,
        "harm_terms":     harm_matches[:5],
        "token_count":    len(tokens),
        "non_ascii_ratio": round(non_ascii_ratio, 4),
    }

    return "CROSS_LINGUAL_ROMANISATION_ATTACK", confidence, evidence
