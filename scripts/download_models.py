"""Download FIE model artifacts from GitHub Release assets.

The trained model files (.pkl classifiers, FAISS index) are deliberately not
tracked in git — they are distributed as assets on a GitHub Release so that:

  1. A fresh `git clone` can be made fully functional with one command.
  2. CI-built Docker images and PyPI wheels contain the real models instead
     of silently falling back to rule-based detection.
  3. Every artifact is integrity-checked against a pinned SHA-256 before use.

Usage:
    python scripts/download_models.py            # download missing/changed files
    python scripts/download_models.py --check    # verify local files only, no network
    python scripts/download_models.py --strict   # exit 1 if the release is unreachable
    python scripts/download_models.py --tag models-v1.13.0   # override release tag

Exit codes:
    0  all artifacts present and verified (or release unavailable in non-strict mode)
    1  checksum mismatch (always fatal — never run with corrupt models), or
       release unreachable in --strict mode

The release asset for a manifest entry `fie/models/foo.pkl` must be named
`fie__models__foo.pkl` (slashes become double underscores — release assets are
a flat namespace). See docs/OPERATIONS.md for the upload procedure.

Stdlib only — safe to run before `pip install -r requirements.txt`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST_PATH = os.path.join(HERE, "model_manifest.json")


def _load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _asset_name(rel_path: str) -> str:
    # Release assets use the plain filename (no path prefix).
    return os.path.basename(rel_path)


def _download(url: str, dest: str, token: str = "") -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "fie-model-downloader"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    # Write to a temp file first so an interrupted download never leaves a
    # half-written model where the runtime would try to load it.
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, os.fdopen(fd, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify local files against the manifest; no network access")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the release or an asset is unreachable "
                             "(default: warn and continue, so builds without the "
                             "release published yet are not broken)")
    parser.add_argument("--tag", default=os.getenv("FIE_MODELS_RELEASE", ""),
                        help="release tag to download from (default: manifest release_tag, "
                             "override via FIE_MODELS_RELEASE env var)")
    args = parser.parse_args()

    manifest = _load_manifest()
    repo = manifest["repo"]
    tag = args.tag or manifest["release_tag"]
    token = os.getenv("GITHUB_TOKEN", "")
    base_url = f"https://github.com/{repo}/releases/download/{tag}"

    ok, missing, mismatched, unreachable = [], [], [], []

    for art in manifest["artifacts"]:
        rel = art["path"]
        local = os.path.join(ROOT, rel.replace("/", os.sep))

        if os.path.exists(local) and _sha256(local) == art["sha256"]:
            ok.append(rel)
            continue

        if os.path.exists(local):
            print(f"[stale]   {rel} — checksum differs from manifest")
        else:
            print(f"[missing] {rel}")

        if args.check:
            missing.append(rel)
            continue

        url = f"{base_url}/{_asset_name(rel)}"
        try:
            print(f"          downloading {url}")
            _download(url, local, token)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"[warn]    could not download {rel}: {exc}")
            unreachable.append(rel)
            continue

        actual = _sha256(local)
        if actual != art["sha256"]:
            # A bad checksum after a successful download means the published
            # asset does not match the manifest — never trust that file.
            os.unlink(local)
            print(f"[FATAL]   {rel}: downloaded file sha256={actual[:16]}… "
                  f"does not match manifest {art['sha256'][:16]}… — deleted")
            mismatched.append(rel)
            continue
        ok.append(rel)
        print(f"[ok]      {rel} verified")

    total = len(manifest["artifacts"])
    print(f"\nverified {len(ok)}/{total} artifacts "
          f"({len(missing)} missing, {len(unreachable)} unreachable, {len(mismatched)} corrupt)")

    if mismatched:
        return 1
    if args.check and missing:
        return 1
    if unreachable:
        if args.strict:
            return 1
        print("[warn]    some artifacts unavailable — the system will run with "
              "rule-based fallbacks for those layers. Publish the release "
              f"'{tag}' (see docs/OPERATIONS.md) to enable full detection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
