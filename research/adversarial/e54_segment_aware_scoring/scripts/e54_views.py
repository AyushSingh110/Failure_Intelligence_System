"""
E54 view construction and pooling, exactly as preregistered (PREREGISTRATION.md §3).
"""
from __future__ import annotations

import math
import re

import e54_common as C

_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+|(?<=[:;])\s+")
MIN_WORDS = 4
WINDOWS = (64, 128)
STRIDE = 32
TOPK = 3
BETA = 4.0
VARIANTS = {"V1": ("whole", "seg"), "V2": ("whole", "w64", "w128"), "V3": ("whole", "seg", "w64", "w128")}
RULES = ("max", "top3", "lse4")
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(str(C.REPO / "fie" / "models" / "minilm-onnx" / "tokenizer.json"))
        t.no_truncation()
        t.no_padding()
        _TOK = t
    return _TOK


def segments(text: str) -> list[tuple[int, int]]:
    """Sentence/clause character spans; pieces under MIN_WORDS words merge forward (last merges back)."""
    spans, pos = [], 0
    for m in _SPLIT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    pieces = []
    for a, b in spans:
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg.rstrip())
        if seg.strip():
            pieces.append([a + lead, a + trail])
    merged: list[list[int]] = []
    carry = None
    for p in pieces:
        if carry is not None:
            p = [carry[0], p[1]]
            carry = None
        if len(text[p[0]:p[1]].split()) < MIN_WORDS:
            carry = p
            continue
        merged.append(p)
    if carry is not None:
        if merged:
            merged[-1][1] = carry[1]
        else:
            merged.append(carry)
    return [(a, b) for a, b in merged]


def windows(text: str, size: int) -> list[tuple[int, int, int, int]]:
    """Token windows: (char_start, char_end, tok_start, tok_end) — only when the text exceeds `size` tokens."""
    enc = tokenizer().encode(text, add_special_tokens=False)
    offs = enc.offsets
    n = len(offs)
    if n <= size:
        return []
    starts = list(range(0, n - size + 1, STRIDE))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [(offs[s][0], offs[s + size - 1][1], s, s + size) for s in starts]


def build_views(text: str) -> tuple[list[dict], int]:
    views = [{"kind": "whole", "start": 0, "end": len(text), "text": text}]
    for a, b in segments(text):
        views.append({"kind": "seg", "start": a, "end": b, "text": text[a:b]})
    for w in WINDOWS:
        for a, b, ts, te in windows(text, w):
            views.append({"kind": f"w{w}", "start": a, "end": b, "tok_start": ts, "tok_end": te, "text": text[a:b]})
    n_tokens = len(tokenizer().encode(text, add_special_tokens=False).ids)
    return views, n_tokens


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def aggregate(view_scores: list[tuple[str, str, float]], kinds: tuple[str, ...], rule: str) -> float:
    """view_scores: (kind, text_sha, score). Unique texts within the prompt count once."""
    uniq: dict[str, float] = {}
    for kind, sha, s in view_scores:
        if kind in kinds:
            uniq[sha] = s
    vals = sorted(uniq.values(), reverse=True)
    if rule == "max":
        return vals[0]
    if rule == "top3":
        top = vals[:TOPK]
        return sum(top) / len(top)
    if rule == "lse4":
        z = [_logit(v) for v in vals]
        m = max(z)
        lse = m + math.log(sum(math.exp(BETA * (x - m)) for x in z) / len(z)) / BETA
        return 1.0 / (1.0 + math.exp(-lse))
    raise ValueError(rule)


def candidates() -> list[str]:
    return [f"{v}_{r}" for v in VARIANTS for r in RULES]
