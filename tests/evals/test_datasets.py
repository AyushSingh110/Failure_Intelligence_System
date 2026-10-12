"""Dataset registry and integrity: parsing, validation, content hashing, the eight pins."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import canonical, datasets
from evals.datasets import DatasetIntegrityError
from _helpers import REPO_ROOT

# The approved row counts (WP-001 hard checkpoint, step 2).
EXPECTED_ROWS = {
    "xstest_safe": 250, "xstest_unsafe": 198, "orbench_hard": 250, "jailbreakbench": 134,
    "harmbench": 387, "strongreject": 242, "sorrybench": 387, "advbench": 168,
}


def _write(path: Path, lines: list[str], newline: str = "\n") -> Path:
    path.write_bytes(newline.join(lines).encode("utf-8"))
    return path


def _row(prompt="hello there", label="safe", **extra) -> str:
    return json.dumps({"prompt": prompt, "label": label, **extra}, ensure_ascii=False)


# ── the real registry (integration) ──────────────────────────────────────────

def test_registry_lists_exactly_the_eight_benchmark_datasets():
    reg = datasets.load_registry(REPO_ROOT)
    bench = {k for k, v in reg["datasets"].items() if v.get("kind", "benchmark") == "benchmark"}
    assert bench == set(EXPECTED_ROWS)


@pytest.mark.parametrize("ds_id,rows", sorted(EXPECTED_ROWS.items()))
def test_row_counts_match_the_approved_baseline(ds_id, rows):
    ds = datasets.load_dataset(ds_id, REPO_ROOT)        # verifies count, unique, content hash
    assert ds.n == rows
    assert ds.content_sha256 == ds.spec["content_sha256"]


def test_labels_and_roles():
    reg = datasets.load_registry(REPO_ROOT)["datasets"]
    assert {k for k, v in reg.items() if v["label"] == "benign"} >= {"xstest_safe", "orbench_hard"}
    assert reg["advbench"]["role"] == "case_study"
    headline = {k for k, v in reg.items() if v.get("role") == "headline_attack"}
    assert headline == {"jailbreakbench", "harmbench", "strongreject", "sorrybench"}
    assert "advbench" not in headline


def test_harmbench_keeps_387_rows_and_reports_380_unique():
    ds = datasets.load_dataset("harmbench", REPO_ROOT)
    assert (ds.n, ds.n_unique) == (387, 380)


def test_every_entry_records_source_and_contamination_metadata():
    for ds_id, spec in datasets.load_registry(REPO_ROOT)["datasets"].items():
        if spec.get("kind", "benchmark") != "benchmark":
            continue
        assert spec["source"]["origin"], ds_id
        assert isinstance(spec["source"]["revision_recorded"], bool), ds_id
        assert spec["contamination"]["audit"], ds_id
        assert spec["contamination"]["audited_against"], ds_id
        assert len(spec["content_sha256"]) == 64 and len(spec["raw_sha256_lf"]) == 64, ds_id


def test_raw_and_content_hashes_are_both_reported():
    table = {r["id"]: r for r in datasets.verify_all(REPO_ROOT, ["xstest_safe"])}
    row = table["xstest_safe"]
    assert row["content_sha256"] != row["raw_sha256_lf"]
    assert len(row["raw_sha256"]) == 64


# ── content hash is independent of line endings ──────────────────────────────

def test_crlf_and_lf_copies_share_a_content_hash_but_not_a_raw_hash(tmp_path):
    lines = [_row("first prompt"), _row("second prompt"), _row("नमस्ते")]
    lf_rows, lf_raw = datasets.parse_jsonl(_write(tmp_path / "lf.jsonl", lines, "\n"), ("safe",))
    crlf_rows, crlf_raw = datasets.parse_jsonl(_write(tmp_path / "crlf.jsonl", lines, "\r\n"), ("safe",))
    assert datasets.content_hash(lf_rows) == datasets.content_hash(crlf_rows)
    assert canonical.sha256_bytes(lf_raw) != canonical.sha256_bytes(crlf_raw)
    assert canonical.sha256_bytes(canonical.lf(crlf_raw)) == canonical.sha256_bytes(lf_raw)


def test_trailing_newline_does_not_change_the_content_hash(tmp_path):
    lines = [_row("a prompt"), _row("another")]
    a, _ = datasets.parse_jsonl(_write(tmp_path / "a.jsonl", lines), ("safe",))
    b, _ = datasets.parse_jsonl(_write(tmp_path / "b.jsonl", lines + [""]), ("safe",))
    assert datasets.content_hash(a) == datasets.content_hash(b)


# ── malformed input ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_line,fragment", [
    ("{not json", "not valid JSON"),
    ('["a list"]', "not a JSON object"),
    ('{"label": "safe"}', "'prompt' is missing"),
    ('{"prompt": 7, "label": "safe"}', "not a string"),
    ('{"prompt": "   ", "label": "safe"}', "'prompt' is empty"),
    ('{"prompt": "ok", "label": "mystery"}', "label 'mystery'"),
    ('{"prompt": "ok"}', "label None"),
    ('{"prompt": "bad \\ud800 text", "label": "safe"}', "lone surrogate"),
])
def test_malformed_rows_are_rejected_with_the_line_number(tmp_path, bad_line, fragment):
    path = _write(tmp_path / "bad.jsonl", [_row("fine"), bad_line])
    with pytest.raises(DatasetIntegrityError) as exc:
        datasets.parse_jsonl(path, ("safe",))
    assert fragment in str(exc.value)
    assert ":2:" in str(exc.value)


def test_invalid_utf8_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(_row("fine").encode() + b"\n" + b'{"prompt": "\xff\xfe", "label": "safe"}')
    with pytest.raises(DatasetIntegrityError, match="not valid UTF-8"):
        datasets.parse_jsonl(path, ("safe",))


def test_oversized_line_is_rejected(tmp_path):
    path = _write(tmp_path / "big.jsonl", [_row("x" * (datasets.MAX_LINE_BYTES + 10))])
    with pytest.raises(DatasetIntegrityError, match="over the"):
        datasets.parse_jsonl(path, ("safe",))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DatasetIntegrityError, match="cannot read"):
        datasets.parse_jsonl(tmp_path / "absent.jsonl", ("safe",))


# ── registry verification against a temporary repository ─────────────────────

def _temp_repo(tmp_path: Path, lines: list[str], **overrides) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    rows, _ = datasets.parse_jsonl(_write(data / "d.jsonl", lines), ("safe",))
    spec = {
        "kind": "benchmark", "path": "data/d.jsonl", "label": "benign", "file_labels": ["safe"],
        "rows": len(rows), "unique": len({r["prompt"] for r in rows}),
        "content_sha256": datasets.content_hash(rows),
    }
    spec.update(overrides)
    reg = tmp_path / "evals" / "registry"
    reg.mkdir(parents=True)
    canonical.write_json(reg / "datasets.json", {"registry_version": 1, "datasets": {"d": spec}})
    return tmp_path


def test_a_matching_dataset_loads(tmp_path):
    root = _temp_repo(tmp_path, [_row("one"), _row("two")])
    ds = datasets.load_dataset("d", root)
    assert ds.prompts == ["one", "two"] and ds.label == "benign"


def test_wrong_row_count_aborts(tmp_path):
    root = _temp_repo(tmp_path, [_row("one"), _row("two")], rows=3)
    with pytest.raises(DatasetIntegrityError, match="row count 2, registry says 3"):
        datasets.load_dataset("d", root)


def test_changed_content_aborts(tmp_path):
    root = _temp_repo(tmp_path, [_row("one"), _row("two")])
    _write(root / "data" / "d.jsonl", [_row("one"), _row("TWO")])
    with pytest.raises(DatasetIntegrityError, match="content hash"):
        datasets.load_dataset("d", root)


def test_duplicates_are_kept_and_counted(tmp_path):
    root = _temp_repo(tmp_path, [_row("same"), _row("same"), _row("other")])
    ds = datasets.load_dataset("d", root)
    assert (ds.n, ds.n_unique) == (3, 2)
    assert ds.prompts == ["same", "same", "other"]


def test_wrong_unique_count_aborts(tmp_path):
    root = _temp_repo(tmp_path, [_row("same"), _row("same")], unique=2)
    with pytest.raises(DatasetIntegrityError, match="unique prompts 1"):
        datasets.load_dataset("d", root)


def test_unknown_dataset_id_aborts(tmp_path):
    root = _temp_repo(tmp_path, [_row("one")])
    with pytest.raises(DatasetIntegrityError, match="unknown dataset id"):
        datasets.load_dataset("nope", root)


@pytest.mark.parametrize("path", ["../outside.jsonl", "data/../../outside.jsonl"])
def test_paths_cannot_escape_the_repository(tmp_path, path):
    root = _temp_repo(tmp_path, [_row("one")], path=path)
    with pytest.raises(DatasetIntegrityError, match="escapes the repository"):
        datasets.load_dataset("d", root)


def test_registry_with_a_missing_field_is_rejected(tmp_path):
    reg = tmp_path / "evals" / "registry"
    reg.mkdir(parents=True)
    canonical.write_json(reg / "datasets.json", {"datasets": {"d": {"path": "x.jsonl"}}})
    with pytest.raises(DatasetIntegrityError, match="has no"):
        datasets.load_registry(tmp_path)
