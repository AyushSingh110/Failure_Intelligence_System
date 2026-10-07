"""
The one serializer and the one hash used for every deterministic artifact.

Every identity claim the harness makes ("these two runs are byte-identical",
"this dataset is the pinned dataset") rests on this module, so its rules are
few and fixed:

  * UTF-8, no byte-order mark, non-ASCII written as-is (ensure_ascii=False)
  * "\\n" only, files written in binary mode so the platform cannot translate
  * keys sorted at every depth
  * floats rounded to FLOAT_DIGITS places, then Python's shortest round-trip repr
  * -0.0 becomes 0.0; NaN and infinities are rejected
  * tuples become lists; any other type is rejected
  * no timestamps, host names or random ids — callers keep those in run metadata

Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

FLOAT_DIGITS = 6


class CanonicalError(ValueError):
    """Raised when a value cannot be serialized deterministically."""


def normalize(obj: Any, _path: str = "$") -> Any:
    """Return a copy of `obj` that json.dumps renders identically on every run."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise CanonicalError(f"non-finite float at {_path}: {obj!r}")
        rounded = round(obj, FLOAT_DIGITS)
        return 0.0 if rounded == 0 else rounded      # folds -0.0 into 0.0
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalError(f"non-string key at {_path}: {key!r}")
            out[key] = normalize(value, f"{_path}.{key}")
        return out
    if isinstance(obj, (list, tuple)):
        return [normalize(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    raise CanonicalError(f"unsupported type at {_path}: {type(obj).__name__}")


def dumps(obj: Any, *, pretty: bool = False) -> str:
    """Canonical JSON text. `pretty` is for single documents meant to be read."""
    data = normalize(obj)
    if pretty:
        return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          indent=2, separators=(",", ": "))
    return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def to_bytes(obj: Any, *, pretty: bool = False) -> bytes:
    """Canonical UTF-8 bytes, without a trailing newline."""
    text = dumps(obj, pretty=pretty)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:        # lone surrogate
        raise CanonicalError(f"text is not valid Unicode: {exc}") from exc


def record_line(obj: Any) -> bytes:
    """One JSON Lines record: compact canonical bytes plus exactly one LF."""
    return to_bytes(obj) + b"\n"


def document_bytes(obj: Any) -> bytes:
    """A single JSON document: pretty canonical bytes plus exactly one LF."""
    return to_bytes(obj, pretty=True) + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """SHA-256 of the compact canonical serialization of `obj`."""
    return sha256_bytes(to_bytes(obj))


def hash_rows(rows: Iterable[Any]) -> str:
    """
    Content hash of an ordered sequence of rows.

    Rows are serialized canonically and joined by a single LF. The result does
    not depend on the file's line endings or on a trailing newline, which is the
    point: the repository's older SHA-256 pins were taken over CRLF working
    copies and do not match the LF blobs git stores.
    """
    h = hashlib.sha256()
    first = True
    for row in rows:
        if not first:
            h.update(b"\n")
        h.update(to_bytes(row))
        first = False
    return h.hexdigest()


def lf(data: bytes) -> bytes:
    """Normalize CRLF to LF. Lossless for JSON: raw CR never appears inside it."""
    return data.replace(b"\r\n", b"\n")


def read_lf(path: str | os.PathLike) -> bytes:
    """Read a file with CRLF folded to LF (a checkout may have converted it)."""
    return lf(Path(path).read_bytes())


def write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Write bytes exactly. Binary mode, so no newline translation happens."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    write_bytes(path, document_bytes(obj))


def write_jsonl(path: str | os.PathLike, records: Iterable[Any]) -> None:
    write_bytes(path, b"".join(record_line(r) for r in records))


def read_jsonl(path: str | os.PathLike) -> list[Any]:
    out = []
    for line in read_lf(path).decode("utf-8").split("\n"):
        if line:
            out.append(json.loads(line))
    return out


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(read_lf(path).decode("utf-8"))
