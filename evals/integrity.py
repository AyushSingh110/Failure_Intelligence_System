"""
Model integrity: know exactly which model files are about to be measured.

The expectation comes from scripts/model_manifest.json at the checked-out commit
— the same published list that CI, the Dockerfile and the golden test use.

Verification happens three times per run:

  1. by the orchestrator, before a worker is started
  2. by the worker, BEFORE `fie` is imported — so nothing unverified is unpickled
  3. by the worker, after the last suite — so a file changed mid-run is caught

A mismatch, a missing file, a missing role, or a role with no manifest entry
raises ModelIntegrityError (exit code 3). There is no warn-and-continue path.

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals import canonical

MANIFEST_PATH = "scripts/model_manifest.json"
CHUNK = 1 << 20   # 1 MiB


class ModelIntegrityError(Exception):
    """The model files are not exactly the expected ones. Exit code 3."""


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    """Streaming SHA-256 over raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(repo_root: str | Path, manifest_path: str | None = None) -> dict:
    """Return {"release_tag": str, "artifacts": {relative_path: {"sha256", "size"}}}."""
    path = Path(repo_root) / (manifest_path or MANIFEST_PATH)
    try:
        raw = json.loads(canonical.read_lf(path).decode("utf-8"))
        artifacts = {a["path"]: {"sha256": a["sha256"], "size": a.get("size")}
                     for a in raw["artifacts"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelIntegrityError(f"cannot read model manifest {path}: {exc}") from exc
    return {"release_tag": raw.get("release_tag"), "artifacts": artifacts}


def _resolve(repo_root: str | Path, rel: str) -> Path:
    root = Path(repo_root).resolve()
    if Path(rel).is_absolute():
        raise ModelIntegrityError(f"model path must be relative to the repository: {rel}")
    path = (root / rel).resolve()
    if root != path and root not in path.parents:
        raise ModelIntegrityError(f"model path escapes the repository: {rel}")
    return path


_REMEDY = (
    "\n  This is not a detection result. Either:\n"
    "    * restore the published models:  python scripts/download_models.py --strict\n"
    "    * or, if a new model is meant to be measured, publish it, add it to\n"
    "      scripts/model_manifest.json, and name it in an evaluation profile.\n"
    "  The run was aborted before any model was loaded."
)


def verify_roles(roles: dict[str, str], repo_root: str | Path, manifest: dict | None = None,
                 required: list[str] | tuple[str, ...] = (),
                 overrides: dict[str, str] | None = None) -> dict[str, dict]:
    """
    Verify every file a profile will load. Returns {role: {...}} or raises.

    roles      role name -> repository-relative path
    required   roles that must be present in `roles`
    overrides  role -> expected SHA-256 given explicitly on the command line, for
               measuring an unpublished model. Such a run is never canonical.
    """
    manifest = manifest or load_manifest(repo_root)
    overrides = overrides or {}
    problems: list[str] = []
    table: dict[str, dict] = {}

    for role in required:
        if role not in roles:
            problems.append(f"role '{role}' is required by the profile but no file is assigned to it")

    for role in sorted(roles):
        rel = roles[role]
        entry = manifest["artifacts"].get(rel)
        if role in overrides:
            expected, source = overrides[role], "command line"
        elif entry is None:
            problems.append(f"role '{role}' ({rel}) has no entry in {MANIFEST_PATH}")
            continue
        else:
            expected, source = entry["sha256"], "manifest"
        try:
            path = _resolve(repo_root, rel)
        except ModelIntegrityError as exc:
            problems.append(str(exc))
            continue
        if not path.is_file():
            problems.append(f"role '{role}': file is missing: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(
                f"role '{role}': {rel}\n"
                f"      expected ({source}): {expected}\n"
                f"      actual:              {actual}")
            continue
        table[role] = {
            "path": rel, "file": path.name, "sha256": actual, "size": path.stat().st_size,
            "expected_from": source, "in_manifest": entry is not None,
        }

    if problems:
        raise ModelIntegrityError(
            "model integrity check failed:\n  - " + "\n  - ".join(problems) + _REMEDY)
    return table


def reverify(table: dict[str, dict], repo_root: str | Path) -> None:
    """Re-hash the verified files. Raises if any changed since `table` was built."""
    changed = []
    for role in sorted(table):
        path = _resolve(repo_root, table[role]["path"])
        if not path.is_file():
            changed.append(f"role '{role}': {table[role]['path']} disappeared during the run")
        elif sha256_file(path) != table[role]["sha256"]:
            changed.append(f"role '{role}': {table[role]['path']} changed during the run")
    if changed:
        raise ModelIntegrityError(
            "model files changed while the evaluation was running; results are invalid:\n  - "
            + "\n  - ".join(changed))


def declared_metadata(roles: dict[str, str], repo_root: str | Path) -> dict:
    """
    What the verified metadata files declare. Read only after verify_roles(), so
    these values come from hash-checked files.
    """
    out: dict = {}
    if "pair_meta" in roles:
        meta = json.loads(_resolve(repo_root, roles["pair_meta"]).read_text(encoding="utf-8"))
        out["pair"] = {
            "declared_version": meta.get("version"),
            "threshold": meta.get("threshold"),
            "embed_model": meta.get("embed_model"),
            "model_type": meta.get("model_type"),
        }
    if "meta_classifier_meta" in roles:
        meta = json.loads(_resolve(repo_root, roles["meta_classifier_meta"]).read_text(encoding="utf-8"))
        out["meta_classifier"] = {
            "threshold": meta.get("threshold"),
            "features": meta.get("layer_names", []),
            "model_type": meta.get("model_type"),
        }
    return out


def unmanifested_model_files(repo_root: str | Path, manifest: dict | None = None) -> list[str]:
    """
    Pickled model files on the loader's search path that the manifest does not
    list. They cannot be selected while a profile forces a version, but their
    presence is reported in run metadata.
    """
    manifest = manifest or load_manifest(repo_root)
    root = Path(repo_root)
    found = []
    for folder in ("fie/models", "models"):
        for path in sorted((root / folder).glob("*.pkl")):
            rel = f"{folder}/{path.name}"
            if rel not in manifest["artifacts"]:
                found.append(rel)
    return found
