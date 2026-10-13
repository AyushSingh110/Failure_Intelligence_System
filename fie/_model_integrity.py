"""
fie._model_integrity — verify a model artifact against a manifest before anything reads it.

A model file is only as trustworthy as the check made on its bytes. This module does one
thing: given a manifest and a model directory, it hands back the exact bytes of a listed
artifact after confirming their size and SHA-256, or it raises. It never deserializes
anything, and it imports only the standard library.

    manifest = parse_manifest(text)                       # validate the whole manifest first
    entry    = default_entry(manifest, "pair_classifier")  # or select_entry(manifest, role, version)
    data     = read_verified(models_dir, entry)            # the bytes that were checked

The caller must use the returned bytes. Verifying a path and then opening that path again
would check one file and load another.

Selection is exact. A role or version that the manifest does not list is an error, never a
reason to pick something else. What a detector layer should report when verification fails is
decided by the caller, not here.

What this protects against: a wrong, stale, unreviewed or corrupted file being used in place
of the one the project released. What it does not: someone who can already write to the
installation can change the manifest too.

Status: the primitives only. The model loader does not use this module yet.

Manifest format (schema 1). Paths are POSIX-style and relative to the model directory.

    {"schema": 1,
     "artifacts": [
       {"path": "pair_intent_classifier_v6_3b.pkl", "sha256": "<64 lowercase hex>", "size": 11349,
        "role": "pair_classifier", "version": "v6_3b", "in_wheel": true, "default": true},
       ...]}

    path      required  relative, no "..", no backslash, no drive, unique (case-insensitively)
    sha256    required  64 lowercase hexadecimal characters
    size      required  the file's exact size in bytes
    role      required  what the artifact is for; a short token
    version   optional  a short token; (role, version) is unique; required on a default
    in_wheel  required  true: shipped inside the wheel; false: known, with a pinned digest, not shipped
    default   optional  true on at most one entry per role
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

__all__ = [
    "SCHEMA", "Entry", "Manifest",
    "ModelIntegrityError", "ManifestError", "PathViolation", "NotListed", "ArtifactMissing", "IntegrityMismatch",
    "ReadFailed",
    "parse_manifest", "select_entry", "default_entry", "resolve_path", "verify_bytes", "read_verified",
]

SCHEMA = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_REQUIRED = ("path", "sha256", "size", "role", "in_wheel")


# ── errors ──────────────────────────────────────────────────────────────────
# Each carries a fixed `code`, so a caller can branch without reading the message, and the
# manifest-relative `path` where one applies. No message contains an absolute path.

class ModelIntegrityError(Exception):
    """Base class. `code` is one of the six codes below."""
    code = "integrity_error"

    def __init__(self, message: str, *, path: str | None = None):
        super().__init__(message)
        self.path = path


class ManifestError(ModelIntegrityError):
    """The manifest itself cannot be trusted: not JSON, wrong shape, or an invalid entry."""
    code = "manifest_invalid"


class PathViolation(ModelIntegrityError):
    """An artifact path would leave the model directory, or is not a regular file."""
    code = "path_violation"


class NotListed(ModelIntegrityError):
    """The requested role or version is not in the manifest. Nothing else is offered in its place."""
    code = "not_listed"

    def __init__(self, message: str, *, role: str, version: str | None = None):
        super().__init__(message)
        self.role, self.version = role, version


class ArtifactMissing(ModelIntegrityError):
    """The manifest lists the artifact; the file is not there."""
    code = "missing"


class IntegrityMismatch(ModelIntegrityError):
    """The file is there and is not the listed one. `what` is "size" or "sha256"."""
    code = "mismatch"

    def __init__(self, message: str, *, path: str, what: str, expected, actual):
        super().__init__(message, path=path)
        self.what, self.expected, self.actual = what, expected, actual


class ReadFailed(ModelIntegrityError):
    """
    The operating system would not let the artifact be examined or read: permission denied,
    an I/O error, a sharing violation. Not "missing" (the file may well be there) and not
    "mismatch" (its content was never seen).

    `operation` is "resolve", "open" or "read"; `reason` is the exception class name, such as
    "PermissionError"; `errno` is the system error number, or None. The operating system's own
    message is deliberately not kept: it contains the absolute path.
    """
    code = "read_failed"

    def __init__(self, message: str, *, path: str, operation: str, reason: str, errno: int | None):
        super().__init__(message, path=path)
        self.operation, self.reason, self.errno = operation, reason, errno


def _read_failed(entry: "Entry", exc: OSError, operation: str) -> ReadFailed:
    """Classify an OSError without carrying its text, which names the absolute location."""
    return ReadFailed(f"{entry.path}: could not {operation} the artifact ({type(exc).__name__})",
                      path=entry.path, operation=operation, reason=type(exc).__name__,
                      errno=exc.errno if isinstance(exc.errno, int) else None)


# ── manifest ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entry:
    path: str
    sha256: str
    size: int
    role: str
    in_wheel: bool
    version: str | None = None
    default: bool = False


@dataclass(frozen=True)
class Manifest:
    schema: int
    entries: tuple[Entry, ...]

    def by_path(self, path: str) -> Entry | None:
        return next((e for e in self.entries if e.path == path), None)

    def for_role(self, role: str) -> tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.role == role)


def _check_path(value, where: str) -> str:
    """A manifest path: relative, forward slashes, plain segments. Returns it unchanged."""
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: 'path' must be a non-empty string")
    if "\\" in value or "\x00" in value or ":" in value:
        raise ManifestError(f"{where}: path contains a backslash, a colon or a NUL character", path=value)
    if value.startswith("/") or value.endswith("/"):
        raise ManifestError(f"{where}: path must be relative and name a file", path=value)
    for segment in value.split("/"):
        if segment in ("", ".", "..") or not _SEGMENT_RE.match(segment):
            raise ManifestError(f"{where}: path segment {segment!r} is not allowed", path=value)
    return value


def _entry(raw, index: int) -> Entry:
    where = f"artifacts[{index}]"
    if not isinstance(raw, dict):
        raise ManifestError(f"{where}: must be an object")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ManifestError(f"{where}: missing required field(s) {missing}")
    path = _check_path(raw["path"], where)
    sha = raw["sha256"]
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise ManifestError(f"{where}: 'sha256' must be 64 lowercase hexadecimal characters", path=path)
    size = raw["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ManifestError(f"{where}: 'size' must be a non-negative integer", path=path)
    role = raw["role"]
    if not isinstance(role, str) or not _TOKEN_RE.match(role):
        raise ManifestError(f"{where}: 'role' must be a short token", path=path)
    in_wheel = raw["in_wheel"]
    if not isinstance(in_wheel, bool):
        raise ManifestError(f"{where}: 'in_wheel' must be true or false", path=path)
    version = raw.get("version")
    if version is not None and (not isinstance(version, str) or not _TOKEN_RE.match(version)):
        raise ManifestError(f"{where}: 'version' must be a short token or absent", path=path)
    default = raw.get("default", False)
    if not isinstance(default, bool):
        raise ManifestError(f"{where}: 'default' must be true or false", path=path)
    if default and version is None:
        raise ManifestError(f"{where}: a default entry must name its version", path=path)
    return Entry(path=path, sha256=sha, size=size, role=role, in_wheel=in_wheel, version=version, default=default)


def parse_manifest(source) -> Manifest:
    """
    Validate a manifest given as JSON text, bytes, or an already-decoded object.

    The whole manifest is checked before any entry is used: one invalid entry makes the
    manifest invalid. Raises ManifestError.
    """
    if isinstance(source, (bytes, bytearray)):
        try:
            source = bytes(source).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError("manifest is not UTF-8") from exc
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except ValueError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(source, dict):
        raise ManifestError("manifest must be a JSON object")
    schema = source.get("schema")
    if isinstance(schema, bool) or schema != SCHEMA:
        raise ManifestError(f"manifest schema must be {SCHEMA}, found {schema!r}")
    raw_entries = source.get("artifacts")
    if not isinstance(raw_entries, list):
        raise ManifestError("manifest 'artifacts' must be a list")
    entries = tuple(_entry(raw, i) for i, raw in enumerate(raw_entries))

    seen_paths: dict[str, str] = {}
    seen_versions: set[tuple[str, str]] = set()
    defaults: dict[str, str] = {}
    for e in entries:
        folded = e.path.casefold()                  # two names one filesystem cannot tell apart
        if folded in seen_paths:
            raise ManifestError(f"duplicate path: {e.path!r} and {seen_paths[folded]!r}", path=e.path)
        seen_paths[folded] = e.path
        if e.version is not None:
            if (e.role, e.version) in seen_versions:
                raise ManifestError(f"role {e.role!r} lists version {e.version!r} more than once", path=e.path)
            seen_versions.add((e.role, e.version))
        if e.default:
            if e.role in defaults:
                raise ManifestError(
                    f"role {e.role!r} has more than one default: {defaults[e.role]!r} and {e.path!r}", path=e.path)
            defaults[e.role] = e.path
    return Manifest(schema=SCHEMA, entries=entries)


# ── selection: exact, never a substitute ────────────────────────────────────

def select_entry(manifest: Manifest, role: str, version: str) -> Entry:
    """The entry for exactly this role and version. Raises NotListed if the manifest has none."""
    for e in manifest.entries:
        if e.role == role and e.version == version:
            return e
    raise NotListed(f"the manifest lists no {role!r} with version {version!r}", role=role, version=version)


def default_entry(manifest: Manifest, role: str) -> Entry:
    """The entry the manifest marks as the default for this role. Raises NotListed if none is marked."""
    for e in manifest.entries:
        if e.role == role and e.default:
            return e
    raise NotListed(f"the manifest marks no default for role {role!r}", role=role)


# ── reading ─────────────────────────────────────────────────────────────────

def resolve_path(models_dir, entry: Entry) -> Path:
    """
    Where the entry's file is, inside `models_dir`, with symbolic links resolved.

    Raises PathViolation if the resolved location is outside the directory or is not a regular
    file, ArtifactMissing if nothing is there, and ReadFailed if the operating system refuses
    to say. The entry's path was validated when the manifest was parsed; it is checked again
    here because an Entry can be built by hand.

    Only OSError is classified. Anything else (a wrong argument type, for instance) is a
    programming error and is left to propagate.
    """
    try:
        _check_path(entry.path, "entry")
    except ManifestError as exc:
        raise PathViolation(str(exc), path=entry.path) from exc
    directory = os.fspath(models_dir)
    try:
        root = os.path.realpath(directory)
        root_is_dir = stat.S_ISDIR(os.stat(root).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        root_is_dir = False
    except OSError as exc:
        raise _read_failed(entry, exc, "resolve") from None
    if not root_is_dir:
        raise ArtifactMissing("the model directory does not exist", path=entry.path)
    try:
        candidate = os.path.realpath(os.path.join(root, *PurePosixPath(entry.path).parts))
    except OSError as exc:
        raise _read_failed(entry, exc, "resolve") from None
    try:
        inside = os.path.commonpath([root, candidate]) == root and candidate != root
    except ValueError:                               # different drives
        inside = False
    if not inside:
        raise PathViolation("the artifact resolves to a location outside the model directory", path=entry.path)
    try:
        os.lstat(candidate)                          # is anything there at all?
    except (FileNotFoundError, NotADirectoryError):
        raise ArtifactMissing(f"listed artifact is not present: {entry.path}", path=entry.path) from None
    except OSError as exc:
        raise _read_failed(entry, exc, "resolve") from None
    try:
        regular = stat.S_ISREG(os.stat(candidate).st_mode)
    except (FileNotFoundError, NotADirectoryError):  # a link to nowhere
        regular = False
    except OSError as exc:
        raise _read_failed(entry, exc, "resolve") from None
    if not regular:
        raise PathViolation("the artifact is not a regular file", path=entry.path)
    return Path(candidate)


def verify_bytes(entry: Entry, data: bytes) -> None:
    """Raise IntegrityMismatch unless `data` has the entry's size and SHA-256."""
    if len(data) != entry.size:
        raise IntegrityMismatch(f"{entry.path}: size {len(data)} does not match the manifest's {entry.size}",
                                path=entry.path, what="size", expected=entry.size, actual=len(data))
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.sha256:
        raise IntegrityMismatch(f"{entry.path}: SHA-256 does not match the manifest",
                                path=entry.path, what="sha256", expected=entry.sha256, actual=digest)


def read_verified(models_dir, entry: Entry) -> bytes:
    """
    Read the entry's file once and return its bytes, after checking them.

    The bytes returned are the bytes that were hashed: the file is opened once and is not
    consulted again. Never more than one byte beyond the listed size is read, so a file that
    was swapped for a larger one is rejected without being loaded into memory.

    Raises PathViolation, ArtifactMissing, ReadFailed or IntegrityMismatch. Nothing is returned
    on failure. An OSError from the operating system is always one of those; no other kind of
    exception is caught.
    """
    path = resolve_path(models_dir, entry)
    try:
        f = open(path, "rb")
    except FileNotFoundError:                        # removed between the check and the open
        raise ArtifactMissing(f"listed artifact is not present: {entry.path}", path=entry.path) from None
    except OSError as exc:
        raise _read_failed(entry, exc, "open") from None
    with f:
        try:
            data = f.read(entry.size + 1)
            on_disk = os.fstat(f.fileno()).st_size if len(data) > entry.size else len(data)
        except OSError as exc:
            raise _read_failed(entry, exc, "read") from None
    if len(data) > entry.size:
        raise IntegrityMismatch(f"{entry.path}: size {on_disk} does not match the manifest's {entry.size}",
                                path=entry.path, what="size", expected=entry.size, actual=on_disk)
    verify_bytes(entry, data)
    return data
