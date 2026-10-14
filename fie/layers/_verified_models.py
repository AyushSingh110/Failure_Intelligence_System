"""
fie.layers._verified_models — the only place in fie/ that turns model bytes into objects.

Every model artifact the SDK loads is first looked up in the package's own manifest,
`fie/models/MANIFEST.json`, then read once and checked against the size and SHA-256 listed
there. The bytes that passed the check are the bytes that are deserialized; the file is not
opened again. A file the manifest does not list is never opened.

    manifest            = package_manifest()
    classifier, sidecar = select_pair(manifest, requested)    # or select_meta(manifest)
    data   = read_artifact(models_dir, classifier)            # verified bytes; nothing deserialized
    meta   = parse_json(sidecar, read_artifact(models_dir, sidecar))
    model  = unpickle(classifier, data)                       # only verified bytes reach the deserializer

The manifest is derived from the tracked inventory, `scripts/model_manifest.json`, by
`python scripts/download_models.py`, and is never edited by hand. It is looked for in one
place only: the `models` directory of the installed `fie` package. There is no fallback to
another directory and no fallback to another model: a version that is not listed, or a listed
file that is missing or altered, makes the model unavailable.

Failures are raised as `LoadRefused`, which carries
    reason   the public layer reason: "model" (nothing to load) or "load_failed"
    detail   a fixed code for logs and health(): manifest_missing, manifest_invalid,
             not_listed, missing, mismatch, read_failed, path_violation, deserialize_failed
    artifact the manifest-relative path, or None
Neither the message nor any attribute contains an absolute path. ImportError raised while
unpickling (a package the model needs is not installed) is not a LoadRefused: it propagates
unchanged, so the caller can report a missing dependency as it always has.

Out of scope here: the ONNX encoder and the sentence-transformers fallback (WP-022), and
session data in fie/session_tracker.py, which is not a model.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import threading
from pathlib import Path, PurePosixPath

from fie import _model_integrity as _mi

# The package's own manifest. Nothing else is ever read as a manifest at run time.
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "models" / _mi.PACKAGE_MANIFEST_NAME
_MANIFEST_MAX_BYTES = 1 << 20

# Detail codes, and the public reason each one is reported under.
_MODEL_DETAILS = frozenset({"manifest_missing", "missing"})
DETAILS = frozenset({"manifest_missing", "manifest_invalid", "not_listed", "missing", "mismatch",
                     "read_failed", "path_violation", "deserialize_failed"})


class LoadRefused(Exception):
    """A model was not loaded, for a reason that is not a programming error."""

    def __init__(self, detail: str, message: str, *, artifact: str | None = None):
        if detail not in DETAILS:            # a typo here would silently mislabel a refusal
            raise ValueError(f"unknown detail code {detail!r}")
        super().__init__(message)
        self.detail = detail
        self.artifact = artifact
        self.reason = "model" if detail in _MODEL_DETAILS else "load_failed"


def _refused(exc: _mi.ModelIntegrityError, artifact: str | None) -> LoadRefused:
    """An integrity error, as a refusal. Its message carries manifest-relative paths only."""
    return LoadRefused(exc.code, str(exc), artifact=artifact if artifact is not None else exc.path)


# ── the manifest ─────────────────────────────────────────────────────────────
# Read and validated once per process and location. A failure is cached too: a manifest that
# was missing or invalid at the first load stays so for the life of the process, just as the
# loaders themselves try once.

_manifest_lock = threading.Lock()
_manifest_cache: dict[str, object] = {}


def _read_manifest(path: Path):
    """The parsed manifest, or a (detail, message) pair saying why there is none."""
    try:
        with open(path, "rb") as f:
            data = f.read(_MANIFEST_MAX_BYTES + 1)
    except (FileNotFoundError, NotADirectoryError):
        return ("manifest_missing",
                f"the package model manifest {_mi.PACKAGE_MANIFEST_NAME} is not present; "
                "run `python scripts/download_models.py`")
    except OSError as exc:                   # the OS message names the absolute path: not kept
        return ("read_failed", f"could not read the package model manifest ({type(exc).__name__})")
    if len(data) > _MANIFEST_MAX_BYTES:
        return ("manifest_invalid", "the package model manifest is larger than any valid manifest")
    try:
        return _mi.parse_manifest(data)
    except _mi.ManifestError as exc:
        return ("manifest_invalid", f"the package model manifest is invalid: {exc}")


def package_manifest() -> _mi.Manifest:
    """The validated package manifest. Raises LoadRefused if it is missing, unreadable or invalid."""
    key = os.fspath(MANIFEST_PATH)
    with _manifest_lock:
        if key not in _manifest_cache:
            _manifest_cache[key] = _read_manifest(Path(key))
        cached = _manifest_cache[key]
    if isinstance(cached, tuple):
        detail, message = cached
        raise LoadRefused(detail, message, artifact=_mi.PACKAGE_MANIFEST_NAME)
    return cached


# ── selection: exact, never a substitute ─────────────────────────────────────

def select_pair(manifest: _mi.Manifest, requested: str | None) -> tuple[_mi.Entry, _mi.Entry]:
    """
    The PAIR classifier and its side-car.

    `requested` empty or None selects the entry the manifest marks as the default. Otherwise
    it selects exactly that version; a version the manifest does not list is refused, never
    replaced by another one.
    """
    try:
        if requested:
            classifier = _mi.select_entry(manifest, "pair_classifier", requested)
        else:
            classifier = _mi.default_entry(manifest, "pair_classifier")
        sidecar = _mi.select_entry(manifest, "pair_meta", classifier.version)
    except _mi.ModelIntegrityError as exc:
        raise _refused(exc, None) from exc
    return classifier, sidecar


def select_meta(manifest: _mi.Manifest) -> tuple[_mi.Entry, _mi.Entry]:
    """The meta-classifier and its side-car, each the sole entry of its role. Several is refused."""
    try:
        return _mi.sole_entry(manifest, "meta_classifier"), _mi.sole_entry(manifest, "meta_classifier_meta")
    except _mi.ModelIntegrityError as exc:
        raise _refused(exc, None) from exc


# ── reading and deserializing ────────────────────────────────────────────────

def read_artifact(models_dir, entry: _mi.Entry) -> bytes:
    """The entry's bytes, read once and verified. Raises LoadRefused; deserializes nothing."""
    try:
        return _mi.read_verified(models_dir, entry)
    except _mi.ModelIntegrityError as exc:
        raise _refused(exc, entry.path) from exc


def digest(data: bytes) -> str:
    """SHA-256 of the bytes, as hex. Used for the identity a scan result reports."""
    return hashlib.sha256(data).hexdigest()


def unpickle(entry: _mi.Entry, data: bytes):
    """
    Deserialize verified bytes from an in-memory buffer. `data` must come from read_artifact().

    The bytes are checked against the entry once more, so that bytes from anywhere else can
    never reach the deserializer through this function. ImportError propagates unchanged: the
    model needs a package that is not installed. Any other failure inside the deserializer is
    a LoadRefused with detail "deserialize_failed", chained to its cause.
    """
    try:
        _mi.verify_bytes(entry, data)
    except _mi.ModelIntegrityError as exc:
        raise _refused(exc, entry.path) from exc
    import joblib

    try:
        return joblib.load(io.BytesIO(data))
    except ImportError:
        raise
    except Exception as exc:
        raise LoadRefused("deserialize_failed",
                          f"{entry.path}: could not be deserialized ({type(exc).__name__})",
                          artifact=entry.path) from exc


def parse_json(entry: _mi.Entry, data: bytes) -> dict:
    """Parse verified side-car bytes as a JSON object. `data` must come from read_artifact()."""
    try:
        _mi.verify_bytes(entry, data)
    except _mi.ModelIntegrityError as exc:
        raise _refused(exc, entry.path) from exc
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LoadRefused("deserialize_failed", f"{entry.path}: not valid UTF-8 JSON ({type(exc).__name__})",
                          artifact=entry.path) from exc
    if not isinstance(value, dict):
        raise LoadRefused("deserialize_failed", f"{entry.path}: must be a JSON object", artifact=entry.path)
    return value


def required_threshold(entry: _mi.Entry, meta: dict) -> float:
    """The side-car's `threshold`: a number in [0, 1]. Never a default."""
    value = meta.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0.0 <= value <= 1.0:
        raise LoadRefused("deserialize_failed", f"{entry.path}: 'threshold' must be a number in [0, 1]",
                          artifact=entry.path)
    return float(value)


def required_text(entry: _mi.Entry, meta: dict, field: str) -> str:
    """A non-empty string field of the side-car. Never a default."""
    value = meta.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LoadRefused("deserialize_failed", f"{entry.path}: {field!r} must be a non-empty string",
                          artifact=entry.path)
    return value


def required_names(entry: _mi.Entry, meta: dict, field: str) -> list[str]:
    """A non-empty list of strings in the side-car. Never a default."""
    value = meta.get(field)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise LoadRefused("deserialize_failed", f"{entry.path}: {field!r} must be a non-empty list of names",
                          artifact=entry.path)
    return list(value)


def file_name(entry: _mi.Entry) -> str:
    """The entry's file name, for the readiness log line."""
    return PurePosixPath(entry.path).name
