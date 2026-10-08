"""
Dataset registry and integrity.

A dataset is a local JSON Lines file. It is parsed with json.loads and nothing
else: no `datasets` library, no loading scripts, no code from a dataset is ever
executed.

Identity is the CONTENT hash (evals.canonical.hash_rows): SHA-256 over the rows,
canonically serialized and joined by LF. It does not depend on line endings.
The raw byte hash is reported too, for information only — the repository's older
manifests pinned raw hashes of CRLF working copies, which do not match the LF
blobs git stores and so fail on any other checkout.

Any mismatch raises DatasetIntegrityError. There is no warn-and-continue path.

Standard library only (the lite worker imports this module with numpy blocked).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals import canonical

REGISTRY_PATH = "evals/registry/datasets.json"
MAX_LINE_BYTES = 1_000_000
EXPECTED_LABELS = ("attack", "benign")


class DatasetIntegrityError(Exception):
    """A dataset is missing, malformed, or not the pinned dataset. Exit code 4."""


@dataclass(frozen=True)
class Dataset:
    id: str
    spec: dict
    rows: list[dict] = field(repr=False)
    content_sha256: str = ""
    raw_sha256: str = ""
    raw_sha256_lf: str = ""

    @property
    def prompts(self) -> list[str]:
        return [r["prompt"] for r in self.rows]

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_unique(self) -> int:
        return len(set(self.prompts))

    @property
    def label(self) -> str:
        return self.spec["label"]


def load_registry(repo_root: str | Path) -> dict:
    path = Path(repo_root) / REGISTRY_PATH
    try:
        reg = canonical.read_json(path)
    except (OSError, ValueError) as exc:
        raise DatasetIntegrityError(f"cannot read dataset registry {path}: {exc}") from exc
    if not isinstance(reg, dict) or not isinstance(reg.get("datasets"), dict):
        raise DatasetIntegrityError(f"{path}: expected an object with a 'datasets' object")
    for ds_id, spec in reg["datasets"].items():
        for key in ("path", "rows", "unique", "label", "file_labels", "content_sha256"):
            if key not in spec:
                raise DatasetIntegrityError(f"registry entry '{ds_id}' has no '{key}'")
        if spec["label"] not in EXPECTED_LABELS:
            raise DatasetIntegrityError(
                f"registry entry '{ds_id}': label must be one of {EXPECTED_LABELS}")
    return reg


def _resolve(repo_root: str | Path, rel: str) -> Path:
    root = Path(repo_root).resolve()
    if Path(rel).is_absolute():
        raise DatasetIntegrityError(f"dataset path must be relative to the repository: {rel}")
    path = (root / rel).resolve()
    if root != path and root not in path.parents:
        raise DatasetIntegrityError(f"dataset path escapes the repository: {rel}")
    return path


def parse_jsonl(path: str | Path, allowed_labels: tuple[str, ...] | list[str]) -> tuple[list[dict], bytes]:
    """
    Parse and validate one JSON Lines file. Returns (rows, raw_bytes).

    Each row must be a JSON object with a non-empty string `prompt` and a
    `label` from `allowed_labels`. Errors name the 1-based line number.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise DatasetIntegrityError(f"cannot read dataset file {p}: {exc}") from exc

    rows: list[dict] = []
    for lineno, line in enumerate(canonical.lf(raw).split(b"\n"), start=1):
        if not line.strip():
            continue
        if len(line) > MAX_LINE_BYTES:
            raise DatasetIntegrityError(
                f"{p}:{lineno}: line is {len(line)} bytes, over the {MAX_LINE_BYTES} limit")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetIntegrityError(f"{p}:{lineno}: not valid UTF-8 ({exc})") from exc
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetIntegrityError(f"{p}:{lineno}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DatasetIntegrityError(f"{p}:{lineno}: row is not a JSON object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise DatasetIntegrityError(f"{p}:{lineno}: 'prompt' is missing or not a string")
        if not prompt.strip():
            raise DatasetIntegrityError(f"{p}:{lineno}: 'prompt' is empty")
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DatasetIntegrityError(
                f"{p}:{lineno}: 'prompt' contains a lone surrogate ({exc.reason})") from exc
        if row.get("label") not in allowed_labels:
            raise DatasetIntegrityError(
                f"{p}:{lineno}: label {row.get('label')!r} not in {list(allowed_labels)}")
        rows.append(row)
    return rows, raw


def content_hash(rows: list[dict]) -> str:
    try:
        return canonical.hash_rows(rows)
    except canonical.CanonicalError as exc:
        raise DatasetIntegrityError(f"dataset rows cannot be hashed: {exc}") from exc


def load_dataset(dataset_id: str, repo_root: str | Path, registry: dict | None = None,
                 verify: bool = True) -> Dataset:
    """
    Load one registered dataset. With `verify` (the default) the row count,
    unique-prompt count and content hash must equal the registry, or this raises.
    """
    reg = registry or load_registry(repo_root)
    spec = reg["datasets"].get(dataset_id)
    if spec is None:
        raise DatasetIntegrityError(f"unknown dataset id: {dataset_id!r}")
    path = _resolve(repo_root, spec["path"])
    rows, raw = parse_jsonl(path, tuple(spec["file_labels"]))
    ds = Dataset(
        id=dataset_id, spec=spec, rows=rows,
        content_sha256=content_hash(rows),
        raw_sha256=canonical.sha256_bytes(raw),
        raw_sha256_lf=canonical.sha256_bytes(canonical.lf(raw)),
    )
    if verify:
        problems = []
        if ds.n != spec["rows"]:
            problems.append(f"row count {ds.n}, registry says {spec['rows']}")
        if ds.n_unique != spec["unique"]:
            problems.append(f"unique prompts {ds.n_unique}, registry says {spec['unique']}")
        if ds.content_sha256 != spec["content_sha256"]:
            problems.append(f"content hash {ds.content_sha256[:16]}…, "
                            f"registry says {spec['content_sha256'][:16]}…")
        if problems:
            raise DatasetIntegrityError(
                f"dataset '{dataset_id}' ({spec['path']}) is not the pinned dataset: "
                + "; ".join(problems)
                + ". Restore the file from git, or — if the dataset is meant to change — "
                  "register it under a new id. A changed dataset is never compared with an old baseline.")
    return ds


def verify_all(repo_root: str | Path, ids: list[str] | None = None,
               registry: dict | None = None) -> list[dict]:
    """Verify datasets and return one summary row each. Raises on the first failure."""
    reg = registry or load_registry(repo_root)
    table = []
    for ds_id in (ids if ids is not None else list(reg["datasets"])):
        ds = load_dataset(ds_id, repo_root, reg)
        table.append({
            "id": ds_id, "path": ds.spec["path"], "kind": ds.spec.get("kind", "benchmark"),
            "rows": ds.n, "unique": ds.n_unique, "label": ds.label,
            "content_sha256": ds.content_sha256,
            "raw_sha256": ds.raw_sha256, "raw_sha256_lf": ds.raw_sha256_lf,
        })
    return table


def identity_block(repo_root: str | Path, ids: list[str], registry: dict | None = None) -> dict[str, Any]:
    """The per-dataset identity that goes into the run fingerprint."""
    reg = registry or load_registry(repo_root)
    out = {}
    for row in verify_all(repo_root, ids, reg):
        out[row["id"]] = {
            "content_sha256": row["content_sha256"], "rows": row["rows"],
            "unique": row["unique"], "label": row["label"],
        }
    return out
