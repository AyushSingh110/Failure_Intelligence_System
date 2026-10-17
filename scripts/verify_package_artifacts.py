#!/usr/bin/env python3
"""
Verify the model content of built distributions against the tracked inventory (WP-021 Stage 3.3c).

    python scripts/verify_package_artifacts.py --sdist dist/fie_sdk-X.tar.gz --wheel dist/fie_sdk-X-py3-none-any.whl
                                               [--report out.json]

The expected content is derived from scripts/model_manifest.json and nothing else: under
fie/models/ an archive must contain exactly the `in_wheel: true` artifacts, each with its listed
size and SHA-256, plus fie/models/MANIFEST.json with exactly the bytes derive_package_manifest()
gives for that inventory. Nothing more, nothing less, nothing altered. An sdist must also carry
the inputs a wheel build from it needs: pyproject.toml, hatch_build.py, fie/_model_integrity.py
and an inventory that derives the same package manifest.

What is read: archive member names and member bytes, which are hashed. No member is extracted to
disk, imported or deserialized. fie/_model_integrity.py is loaded by file path (standard library
only); the `fie` package is not imported.

Only the model area and the build inputs are judged. The rest of the sdist (about 67 MB of
repository content today) is out of scope here: narrowing it is a separate decision (W21-29).

Exit codes: 0 every archive passed; 1 at least one check failed; 2 the inventory or an argument
could not be used. Diagnostics name archives by file name and members by archive path only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import posixpath
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_PREFIX = "fie/models/"
RUNTIME_MANIFEST = "fie/models/MANIFEST.json"
SDIST_INPUTS = ("pyproject.toml", "hatch_build.py", "fie/_model_integrity.py", "scripts/model_manifest.json")
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


class UsageError(Exception):
    """The inventory or an argument cannot be used; nothing was judged."""


def load_integrity(path: Path):
    name = "_fie_verify_package_model_integrity"
    if name in sys.modules:
        return sys.modules[name]
    if not path.is_file():
        raise UsageError(f"integrity module not found: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def expectations(inventory_bytes: bytes, mi) -> dict:
    """What every archive must hold under fie/models/, derived from the inventory alone."""
    try:
        manifest = mi.validate_production(inventory_bytes)
        derived = mi.derive_package_manifest(inventory_bytes)
    except mi.ModelIntegrityError as exc:
        raise UsageError(f"the inventory is not a valid production inventory: {exc}") from None
    artifacts = {e.path: {"size": e.size, "sha256": e.sha256} for e in manifest.entries if e.in_wheel}
    return {"artifacts": artifacts, "manifest_bytes": derived,
            "manifest": {"size": len(derived), "sha256": hashlib.sha256(derived).hexdigest()}}


# ── archive reading ─────────────────────────────────────────────────────────

def _normalize(name: str) -> str | None:
    """A member path in canonical form, or None if it is not a plain relative path."""
    if not name or "\\" in name or "\x00" in name or name.startswith("/") or ":" in name.split("/", 1)[0]:
        return None
    parts = name.rstrip("/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    return posixpath.join(*parts)


def read_wheel(path: Path) -> tuple[dict, list[str]]:
    """{member path: {"size", "sha256", "kind"}} for every file member, plus structural problems."""
    members, problems = {}, []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            norm = _normalize(info.filename)
            if norm is None:
                problems.append(f"member path is not a plain relative path: {info.filename!r}")
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            kind = "symlink" if stat.S_ISLNK(mode) else "file"
            data = z.read(info)
            if norm in members:
                problems.append(f"duplicate member: {norm}")
                continue
            members[norm] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind,
                             "_bytes": data if norm == RUNTIME_MANIFEST or norm in SDIST_INPUTS else None}
    return members, problems


def read_sdist(path: Path) -> tuple[dict, list[str]]:
    """As read_wheel, with the sdist's single top-level directory removed."""
    members, problems, tops = {}, [], set()
    with tarfile.open(path, "r:*") as t:
        for info in t.getmembers():
            norm = _normalize(info.name)
            if norm is None:
                problems.append(f"member path is not a plain relative path: {info.name!r}")
                continue
            top, _, rest = norm.partition("/")
            tops.add(top)
            if info.isdir():
                continue
            if not rest:
                problems.append(f"file outside the top-level directory: {norm}")
                continue
            if not info.isfile():
                problems.append(f"not a regular file ({'link' if info.issym() or info.islnk() else 'special'}): {rest}")
                continue
            data = t.extractfile(info).read()
            if rest in members:
                problems.append(f"duplicate member: {rest}")
                continue
            members[rest] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": "file",
                             "_bytes": data if rest == RUNTIME_MANIFEST or rest in SDIST_INPUTS else None}
    if len(tops) != 1:
        problems.append(f"an sdist must have exactly one top-level directory, found {len(tops)}")
    return members, problems


# ── checks ──────────────────────────────────────────────────────────────────

def check_models(members: dict, expected: dict) -> list[str]:
    """The fie/models/ area must hold exactly the expected files, byte for byte."""
    failures = []
    folded = {}
    for name in members:
        key = name.casefold()
        if key in folded:
            failures.append(f"members differ only by case: {folded[key]} and {name}")
        folded[key] = name
    model_area = {n: m for n, m in members.items() if n.casefold().startswith(MODELS_PREFIX)}
    allowed = set(expected["artifacts"]) | {RUNTIME_MANIFEST}
    for name in sorted(allowed - set(model_area)):
        failures.append(f"missing required member: {name}")
    for name in sorted(set(model_area) - allowed):
        failures.append(f"unexpected member under {MODELS_PREFIX}: {name}")
    for name, want in sorted(expected["artifacts"].items()):
        got = model_area.get(name)
        if got is None:
            continue
        if got["kind"] != "file":
            failures.append(f"{name}: not a regular file ({got['kind']})")
        if got["size"] != want["size"]:
            failures.append(f"{name}: size {got['size']} does not match the inventory's {want['size']}")
        elif got["sha256"] != want["sha256"]:
            failures.append(f"{name}: SHA-256 does not match the inventory")
    packed = model_area.get(RUNTIME_MANIFEST)
    if packed is not None and packed["_bytes"] != expected["manifest_bytes"]:
        failures.append(f"{RUNTIME_MANIFEST}: bytes differ from the manifest derived from the inventory "
                        f"(sha256 {packed['sha256'][:16]}..., expected {expected['manifest']['sha256'][:16]}...)")
    return failures


def check_sdist_inputs(members: dict, expected: dict, mi) -> list[str]:
    """The files a wheel build from this sdist needs, and an inventory that agrees with ours."""
    failures = [f"missing build input: {name}" for name in SDIST_INPUTS if name not in members]
    inventory = members.get("scripts/model_manifest.json")
    if inventory is not None:
        try:
            if mi.derive_package_manifest(inventory["_bytes"]) != expected["manifest_bytes"]:
                failures.append("scripts/model_manifest.json in the sdist describes a different model set")
        except mi.ModelIntegrityError as exc:
            failures.append(f"scripts/model_manifest.json in the sdist is not a valid inventory: {exc}")
    return failures


def verify(path: Path, kind: str, expected: dict, mi) -> dict:
    """Every check for one archive. Returns a JSON-ready result."""
    members, problems = (read_sdist if kind == "sdist" else read_wheel)(path)
    failures = list(problems) + check_models(members, expected)
    if kind == "sdist":
        failures += check_sdist_inputs(members, expected, mi)
    model_area = {n: {"size": m["size"], "sha256": m["sha256"]} for n, m in sorted(members.items())
                  if n.casefold().startswith(MODELS_PREFIX)}
    data = path.read_bytes()
    return {"archive": path.name, "kind": kind, "archive_sha256": hashlib.sha256(data).hexdigest(),
            "archive_bytes": len(data), "member_count": len(members), "members": sorted(members),
            "fie_models": model_area, "failures": failures, "passed": not failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sdist", action="append", default=[], type=Path, help="an sdist to verify (repeatable)")
    parser.add_argument("--wheel", action="append", default=[], type=Path, help="a wheel to verify (repeatable)")
    parser.add_argument("--inventory", type=Path, default=ROOT / "scripts" / "model_manifest.json",
                        help="the authoritative inventory (default: scripts/model_manifest.json)")
    parser.add_argument("--integrity-module", type=Path, default=ROOT / "fie" / "_model_integrity.py",
                        help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help="write the full result as JSON here")
    args = parser.parse_args(argv)
    try:
        if not args.sdist and not args.wheel:
            raise UsageError("nothing to verify: pass --sdist and/or --wheel")
        for p in args.sdist + args.wheel:
            if not p.is_file():
                raise UsageError(f"archive not found: {p.name}")
        mi = load_integrity(args.integrity_module)
        try:
            inventory_bytes = args.inventory.read_bytes()
        except OSError as exc:
            raise UsageError(f"inventory cannot be read ({type(exc).__name__})") from None
        expected = expectations(inventory_bytes, mi)
    except UsageError as exc:
        print(f"verify_package_artifacts: ERROR {exc}", file=sys.stderr)
        return EXIT_USAGE

    results = [verify(p, "sdist", expected, mi) for p in args.sdist] + \
              [verify(p, "wheel", expected, mi) for p in args.wheel]
    print(f"verify_package_artifacts: expecting {len(expected['artifacts'])} approved artifacts and "
          f"{RUNTIME_MANIFEST} (sha256 {expected['manifest']['sha256'][:16]}...) under {MODELS_PREFIX}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['kind']:5} {r['archive']}  sha256 {r['archive_sha256'][:16]}...  "
              f"{len(r['fie_models'])} member(s) under {MODELS_PREFIX}")
        for name, m in r["fie_models"].items():
            print(f"           {name}  {m['size']}  {m['sha256'][:16]}...")
        for failure in r["failures"]:
            print(f"         - {failure}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "expected": {"artifacts": expected["artifacts"], "manifest": expected["manifest"]},
            "results": results, "passed": all(r["passed"] for r in results)}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")       # the same bytes on every platform, so a checksum holds
    ok = all(r["passed"] for r in results)
    print(f"verify_package_artifacts: {'PASSED' if ok else 'FAILED'} ({len(results)} archive(s))")
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
