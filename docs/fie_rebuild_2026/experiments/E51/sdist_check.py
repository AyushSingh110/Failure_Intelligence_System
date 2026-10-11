"""
E51 (WP-020), Stage B remediation: verify and inspect a PyPI source archive WITHOUT executing it.

    python sdist_check.py langdetect 1.0.9 --dest <scratch dir> --out <evidence dir>

Downloads the source archive named by PyPI's JSON metadata from files.pythonhosted.org
with the standard library (proxy handling off), checks its SHA-256 against the digest
PyPI publishes, and lists its contents and build configuration by reading the archive.
Nothing in the archive is imported or run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

APPROVED = {"pypi.org", "files.pythonhosted.org"}


def _open(url: str):
    host = re.match(r"https://([^/]+)/", url).group(1)
    if host not in APPROVED:
        raise SystemExit(f"refusing to contact {host}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(urllib.request.Request(url, headers={"User-Agent": "fie-e51-provision"}), timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("version")
    ap.add_argument("--dest", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dest, out = Path(a.dest), Path(a.out)
    dest.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(_open(f"https://pypi.org/pypi/{a.project}/{a.version}/json").read().decode("utf-8"))
    (out / f"pypi_{a.project}_{a.version}.json").write_text(json.dumps(meta, indent=1, sort_keys=True),
                                                           encoding="utf-8", newline="\n")
    files = [{"filename": f["filename"], "packagetype": f["packagetype"], "python_version": f["python_version"],
              "size": f["size"], "sha256": f["digests"]["sha256"], "url": f["url"],
              "upload_time": f["upload_time_iso_8601"], "yanked": f.get("yanked", False)} for f in meta["urls"]]
    sdists = [f for f in files if f["packagetype"] == "sdist"]
    if len(sdists) != 1:
        raise SystemExit(f"expected exactly one source archive, found {len(sdists)}")
    sd = sdists[0]
    target = dest / sd["filename"]
    if target.exists():
        raise SystemExit(f"{target.name} already exists in the destination; not overwriting")
    data = _open(sd["url"]).read()
    sha = hashlib.sha256(data).hexdigest()
    verified = sha == sd["sha256"] and len(data) == sd["size"]
    if not verified:
        raise SystemExit("the downloaded archive does not match PyPI's published digest or size; nothing was saved")
    target.write_bytes(data)

    members, texts = [], {}
    with tarfile.open(target, "r:gz") as tar:
        for m in tar.getmembers():
            members.append({"name": m.name, "size": m.size, "type": "file" if m.isfile() else
                            "dir" if m.isdir() else "link" if (m.issym() or m.islnk()) else "other"})
            base = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if m.isfile() and base in ("setup.py", "setup.cfg", "pyproject.toml", "PKG-INFO", "MANIFEST.in",
                                       "requirements.txt"):
                texts[base] = tar.extractfile(m).read().decode("utf-8", errors="replace")
    names = [m["name"] for m in members]
    unsafe = [n for n in names if n.startswith(("/", "\\")) or ".." in n.split("/")]
    suffixes = sorted({Path(n).suffix for n in names if "." in Path(n).name})
    report = {
        "project": a.project, "version": a.version,
        "release_info": {k: meta["info"].get(k) for k in ("name", "version", "requires_python", "requires_dist",
                                                         "license", "home_page", "author")},
        "files_published": files,
        "source_archive": {**sd, "sha256_downloaded": sha, "bytes_downloaded": len(data),
                           "matches_pypi_digest_and_size": verified},
        "archive_member_count": len(members),
        "archive_links_or_special_members": [m["name"] for m in members if m["type"] in ("link", "other")],
        "archive_unsafe_paths": unsafe,
        "archive_file_suffixes": suffixes,
        "has_pyproject_toml": "pyproject.toml" in texts,
        "has_setup_cfg": "setup.cfg" in texts,
        "has_setup_py": "setup.py" in texts,
        "compiled_or_binary_members": [n for n in names if Path(n).suffix in
                                       (".c", ".cpp", ".pyx", ".so", ".pyd", ".dll", ".exe", ".pyc")],
        "build_files": texts,
    }
    (out / f"sdist_{a.project}_{a.version}.json").write_text(json.dumps(report, indent=1, sort_keys=True),
                                                            encoding="utf-8", newline="\n")
    brief = {k: report[k] for k in ("release_info", "source_archive", "archive_member_count",
                                    "archive_links_or_special_members", "archive_unsafe_paths",
                                    "archive_file_suffixes", "has_pyproject_toml", "has_setup_cfg", "has_setup_py",
                                    "compiled_or_binary_members")}
    print(json.dumps(brief, indent=1))
    print("---- setup.py ----"); print(texts.get("setup.py", "<none>"))
    print("---- setup.cfg ----"); print(texts.get("setup.cfg", "<none>"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
