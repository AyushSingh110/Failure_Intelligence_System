"""Canonical serializer: the rules every deterministic artifact depends on."""
from __future__ import annotations

import json

import pytest

from evals import canonical as c
from _helpers import child_json


# ── key order, separators, encoding ──────────────────────────────────────────

def test_keys_are_sorted_at_every_depth():
    a = {"b": 1, "a": {"z": 1, "y": [{"q": 1, "p": 2}]}}
    b = {"a": {"y": [{"p": 2, "q": 1}], "z": 1}, "b": 1}
    assert c.to_bytes(a) == c.to_bytes(b) == b'{"a":{"y":[{"p":2,"q":1}],"z":1},"b":1}'


def test_insertion_order_does_not_change_bytes():
    keys = [f"k{i}" for i in range(50)]
    forward = {k: i for i, k in enumerate(keys)}
    backward = {k: forward[k] for k in reversed(keys)}
    assert c.to_bytes(forward) == c.to_bytes(backward)
    assert c.hash_obj(forward) == c.hash_obj(backward)


def test_utf8_without_bom_and_without_ascii_escapes():
    data = c.to_bytes({"t": "नमस्ते"})
    assert not data.startswith(b"\xef\xbb\xbf")
    assert "नमस्ते".encode("utf-8") in data
    assert b"\\u" not in data


@pytest.mark.parametrize("text", [
    "नमस्ते दुनिया",          # Devanagari
    "مرحبا بالعالم",           # Arabic
    "你好，世界",               # CJK
    "こんにちは世界",            # Japanese
    "emoji 🙂🚀",              # astral plane
    "é combining",       # combining mark, deliberately not normalized
    "quote \" backslash \\ tab \t newline \n",
])
def test_unicode_round_trips_unchanged(text):
    line = c.record_line({"p": text})
    assert json.loads(line.decode("utf-8"))["p"] == text


def test_no_unicode_normalization_is_applied():
    composed, decomposed = "é", "é"
    assert c.to_bytes({"p": composed}) != c.to_bytes({"p": decomposed})


def test_lone_surrogate_is_rejected():
    with pytest.raises(c.CanonicalError):
        c.to_bytes({"p": "bad \ud800 text"})


# ── newlines ─────────────────────────────────────────────────────────────────

def test_record_line_ends_with_exactly_one_lf_and_has_no_cr():
    line = c.record_line({"a": 1, "s": "x\r\ny"})
    assert line.endswith(b"\n") and not line.endswith(b"\n\n")
    assert b"\r" not in line            # a CR inside a string is escaped as \\r
    assert line.count(b"\n") == 1


def test_written_files_are_lf_only(tmp_path):
    path = tmp_path / "out.jsonl"
    c.write_jsonl(path, [{"i": i, "s": "line\nbreak"} for i in range(3)])
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 3 and raw.endswith(b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    doc = tmp_path / "doc.json"
    c.write_json(doc, {"b": [1, 2], "a": {"x": 1.5}})
    raw = doc.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")


def test_crlf_copy_reads_back_identically(tmp_path):
    path = tmp_path / "a.jsonl"
    c.write_jsonl(path, [{"i": 1, "s": "a\r\nb"}, {"i": 2}])
    lf_bytes = path.read_bytes()
    crlf = tmp_path / "b.jsonl"
    crlf.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    assert c.read_lf(crlf) == lf_bytes
    assert c.read_jsonl(crlf) == c.read_jsonl(path)


# ── floats ───────────────────────────────────────────────────────────────────

def test_floats_round_to_six_places():
    assert c.to_bytes({"x": 0.1 + 0.2}) == b'{"x":0.3}'
    assert c.to_bytes({"x": 0.12345649}) == b'{"x":0.123456}'
    assert c.to_bytes({"x": 0.7313}) == b'{"x":0.7313}'


def test_negative_zero_becomes_zero():
    assert c.to_bytes({"x": -0.0}) == b'{"x":0.0}'
    assert c.to_bytes({"x": -1e-9}) == b'{"x":0.0}'


def test_integers_and_booleans_are_untouched():
    assert c.to_bytes({"n": 250, "t": True, "f": False, "z": None}) == \
        b'{"f":false,"n":250,"t":true,"z":null}'
    assert c.to_bytes({"n": 1.0}) == b'{"n":1.0}'        # a float stays a float


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(bad):
    with pytest.raises(c.CanonicalError):
        c.to_bytes({"x": bad})


# ── types ────────────────────────────────────────────────────────────────────

def test_tuples_become_lists():
    assert c.to_bytes({"a": (1, 2)}) == b'{"a":[1,2]}'


@pytest.mark.parametrize("bad", [{1, 2}, b"bytes", object(), {1: "non-string key"}])
def test_unsupported_types_are_rejected(bad):
    with pytest.raises(c.CanonicalError):
        c.to_bytes({"x": bad} if not isinstance(bad, dict) else bad)


# ── hashing ──────────────────────────────────────────────────────────────────

def test_hash_rows_ignores_line_endings_but_not_content_or_order():
    rows = [{"prompt": "a", "label": "safe"}, {"prompt": "b", "label": "safe"}]
    assert c.hash_rows(rows) == c.hash_rows([dict(reversed(list(r.items()))) for r in rows])
    assert c.hash_rows(rows) != c.hash_rows(list(reversed(rows)))
    assert c.hash_rows(rows) != c.hash_rows(rows + [{"prompt": "c", "label": "safe"}])
    assert c.hash_rows([]) == c.sha256_bytes(b"")


def test_serialization_is_identical_across_interpreters(tmp_path):
    """Two fresh processes with different hash seeds must agree byte for byte."""
    code = (
        "import json\n"
        "from evals import canonical as c\n"
        "obj = {'z': [0.1+0.2, -0.0, 3], 'a': {'नम': 'स्ते', 'k': (1, 2)}, "
        "'set_like': sorted({'b', 'a', 'c'})}\n"
        "print(json.dumps({'hex': c.to_bytes(obj).hex(), 'sha': c.hash_obj(obj)}))\n"
    )
    first = child_json(code, tmp_path, {"PYTHONHASHSEED": "1"})
    second = child_json(code, tmp_path, {"PYTHONHASHSEED": "4242"})
    assert first == second
