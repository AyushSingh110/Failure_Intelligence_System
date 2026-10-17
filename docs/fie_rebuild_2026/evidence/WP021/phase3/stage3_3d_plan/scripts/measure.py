"""
WP-021 W21-29 (Stage 3.3d planning): measure today's sdist and probe narrowed sdist configurations.

Everything happens under WORK, outside the repository: a `git archive` export of HEAD (LF), the
25 inventory artifacts copied in after a size and SHA-256 check (never deserialized), offline
builds with `python -m build --no-isolation` under a socket-refusing audit hook. Probes change
only a throw-away copy of pyproject.toml. The archive checker judges every archive.

usage: python measure.py <repo> <work>
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
PY = sys.executable
assert not str(WORK).lower().startswith(str(REPO).lower())
for d in ("src", "trees", "out", "logs", "results", "guard", "reports", "unpacked"):
    (WORK / d).mkdir(parents=True, exist_ok=False)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*a, binary=False):
    p = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True)
    return p.stdout if binary else p.stdout.decode()


def scrub(t):
    for real, label in ((str(WORK), "<work>"), (str(WORK).replace("\\", "/"), "<work>"), (str(REPO), "<repo>"),
                        (str(REPO).replace("\\", "/"), "<repo>"), (PY, "<python>"), (str(Path(PY).parent), "<python-env>"),
                        (str(Path.home()), "<home>")):
        t = t.replace(real, label)
    return t


HEAD = git("rev-parse", "HEAD").strip()
NET = WORK / "logs" / "network_attempts.log"
NET.write_text("", encoding="utf-8")
(WORK / "guard" / "sitecustomize.py").write_text(
    "import os, sys\n_L = os.environ.get('S33D_NET_LOG')\n"
    "def _h(e, a):\n"
    "    if e in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):\n"
    "        if _L:\n            open(_L, 'a', encoding='utf-8').write(e + ' ' + repr(a)[:200] + '\\n')\n"
    "        if e in ('socket.connect', 'urllib.Request'):\n"
    "            raise OSError('network blocked by the stage 3.3d guard')\n"
    "sys.addaudithook(_h)\n", encoding="utf-8")
ENV = {k: v for k, v in os.environ.items() if k not in ("SOURCE_DATE_EPOCH", "PYTHONPATH")}
ENV.update({"PYTHONPATH": str(WORK / "guard"), "S33D_NET_LOG": str(NET), "PIP_NO_INDEX": "1",
            "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "PYTHONDONTWRITEBYTECODE": "1"})

SRC = WORK / "src" / "LF"
with tarfile.open(fileobj=io.BytesIO(git("-c", "core.autocrlf=false", "archive", "--format=tar", HEAD, binary=True))) as t:
    t.extractall(SRC)
inventory = json.loads((SRC / "scripts" / "model_manifest.json").read_text(encoding="utf-8"))
for a in inventory["artifacts"]:
    b = (REPO / a["path"]).read_bytes()
    assert len(b) == a["size"] and sha(b) == a["sha256"], f"not verified: {a['path']}"
    (SRC / a["path"]).parent.mkdir(parents=True, exist_ok=True)
    (SRC / a["path"]).write_bytes(b)
BASE_CFG = (SRC / "pyproject.toml").read_text(encoding="utf-8")
WHEEL_BLOCK = '[tool.hatch.build.targets.wheel]\npackages = ["fie"]'
assert BASE_CFG.count(WHEEL_BLOCK) == 1

CORE = ['"/fie"', '"/hatch_build.py"', '"/scripts/model_manifest.json"', '"/README.md"', '"/LICENSE"']
TESTS_FOR_SDIST = ['"/tests/test_build_artifact_allowlist.py"', '"/tests/test_package_artifact_gate.py"',
                   '"/tests/test_model_integrity.py"', '"/tests/test_model_manifest.py"',
                   '"/scripts/verify_package_artifacts.py"']
CONFIGS = {
    "CURRENT": None,
    "NARROW_MIN": CORE,
    "NARROW_TESTS": CORE + TESTS_FOR_SDIST,
}


def run(cmd, label):
    before = NET.read_text(encoding="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, cwd=str(WORK))
    (WORK / "logs" / f"{label}.txt").write_text(scrub(f"$ {' '.join(map(str, cmd))}\nexit={p.returncode}\n"
                                                      f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"),
                                                encoding="utf-8", newline="\n")
    return {"command": scrub(" ".join(map(str, cmd))), "exit_code": p.returncode,
            "network_attempts": NET.read_text(encoding="utf-8")[len(before):].splitlines()}


def sdist_members(path):
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name.split("/", 1)[1]] = {"size": m.size, "sha256": sha(t.extractfile(m).read())}
    return out


def wheel_members(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: {"size": i.file_size, "sha256": sha(z.read(i))} for i in z.infolist() if not i.is_dir()}


MEDIA = (".mp4", ".mp3", ".wav", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp")


def category(p: str) -> str:
    top = p.split("/", 1)[0]
    if p.startswith("fie/models/"):
        return "model artifacts (fie/models/)"
    if p in ("pyproject.toml", "hatch_build.py", "scripts/model_manifest.json", "PKG-INFO", ".gitignore"):
        return "build inputs"
    if p in ("README.md", "LICENSE"):
        return "package metadata files (README, LICENSE)"
    if top == "fie":
        return "package source (fie/)"
    if top.startswith("brag-output"):
        return "demo media bundles (brag-output*/)"
    if p.lower().endswith(MEDIA):
        return "media files elsewhere (docs/media etc.)"
    if p.startswith("docs/fie_rebuild_2026/"):
        return "rebuild reports and evidence (docs/fie_rebuild_2026/)"
    if p.startswith("evals/baselines/"):
        return "evaluation baselines (evals/baselines/)"
    if top == "evals":
        return "evaluation harness code and registry (evals/)"
    if top == "tests":
        return "tests"
    if top == "data":
        return "datasets (data/)"
    if top == "notebooks":
        return "notebooks"
    if top in ("app", "engine", "storage", "telemetry-server", "deploy", "Frontend") or p in (
            "Dockerfile", "config.py", "requirements.txt", "cloudbuild.yaml", ".dockerignore", ".env.example"):
        return "server, frontend and deployment"
    if top == "scripts":
        return "scripts (research and tooling)"
    if top in ("evaluation", "experiments", "research"):
        return "research evaluation code"
    if top == "docs" or p.endswith(".md"):
        return "documentation"
    if top == ".github":
        return "CI workflows (.github/)"
    return "other"


results = {"head": HEAD, "configs": {}}
for name, only in CONFIGS.items():
    tree = WORK / "trees" / name
    shutil.copytree(SRC, tree)
    if only is not None:
        cfg = BASE_CFG.replace(WHEEL_BLOCK, WHEEL_BLOCK + "\n\n[tool.hatch.build.targets.sdist]\nonly-include = [\n    "
                               + ",\n    ".join(only) + ",\n]")
        (tree / "pyproject.toml").write_text(cfg, encoding="utf-8", newline="\n")
    out = WORK / "out" / name
    rec = {"only_include": only, "build_sdist": run([PY, "-m", "build", "--no-isolation", "--sdist", "--outdir",
                                                      str(out), str(tree)], f"{name}_sdist")}
    sd = next(out.glob("*.tar.gz"), None) if out.exists() else None
    if sd:
        up = WORK / "unpacked" / name
        up.mkdir(parents=True)
        with tarfile.open(sd) as t:
            t.extractall(up)
        root = next(p for p in up.iterdir() if p.is_dir())
        rec["build_wheel_from_sdist"] = run([PY, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(out),
                                             str(root)], f"{name}_wheel_from_sdist")
        wh = next(out.glob("*.whl"), None)
        rec["checker"] = run([PY, "-I", str(REPO / "scripts" / "verify_package_artifacts.py"), "--sdist", str(sd)]
                             + (["--wheel", str(wh)] if wh else [])
                             + ["--report", str(WORK / "reports" / f"{name}.json")], f"{name}_checker")
        sm = sdist_members(sd)
        cats = defaultdict(lambda: {"files": 0, "bytes": 0})
        for p, m in sm.items():
            c = cats[category(p)]
            c["files"] += 1
            c["bytes"] += m["size"]
        dirs = defaultdict(int)
        for p, m in sm.items():
            parts = PurePosixPath(p).parts
            dirs["/".join(parts[:2]) if len(parts) > 2 else parts[0]] += m["size"]
        rec["sdist"] = {"name": sd.name, "sha256": sha(sd.read_bytes()), "compressed_bytes": sd.stat().st_size,
                        "uncompressed_bytes": sum(m["size"] for m in sm.values()), "files": len(sm),
                        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["bytes"])),
                        "largest_files": sorted(([p, m["size"]] for p, m in sm.items()), key=lambda x: -x[1])[:25],
                        "largest_dirs": sorted(([d, b] for d, b in dirs.items()), key=lambda x: -x[1])[:25],
                        "members": {p: m for p, m in sorted(sm.items())}}
        if wh:
            wm = wheel_members(wh)
            rec["wheel"] = {"name": wh.name, "sha256": sha(wh.read_bytes()), "bytes": wh.stat().st_size,
                            "files": len(wm), "members": {p: m for p, m in sorted(wm.items())}}
    results["configs"][name] = rec
    shutil.rmtree(tree)
    print(name, rec["build_sdist"]["exit_code"], (rec.get("build_wheel_from_sdist") or {}).get("exit_code"),
          (rec.get("checker") or {}).get("exit_code"), flush=True)

cur = results["configs"]["CURRENT"]
for name in ("NARROW_MIN", "NARROW_TESTS"):
    rec = results["configs"][name]
    if "wheel" in rec and "wheel" in cur:
        rec["wheel_equals_current_wheel_bytes"] = rec["wheel"]["sha256"] == cur["wheel"]["sha256"]
        rec["wheel_members_equal_current"] = rec["wheel"]["members"] == cur["wheel"]["members"]
        diff = sorted(set(rec["wheel"]["members"]) ^ set(cur["wheel"]["members"]))
        changed = sorted(p for p in set(rec["wheel"]["members"]) & set(cur["wheel"]["members"])
                         if rec["wheel"]["members"][p] != cur["wheel"]["members"][p])
        rec["wheel_member_differences"] = {"only_in_one": diff, "content_differs": changed}
results["network_attempts_total"] = NET.read_text(encoding="utf-8").count(chr(10))
(WORK / "results" / "measure.json").write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8",
                                               newline="\n")
print("network attempts:", results["network_attempts_total"])
print("done")
