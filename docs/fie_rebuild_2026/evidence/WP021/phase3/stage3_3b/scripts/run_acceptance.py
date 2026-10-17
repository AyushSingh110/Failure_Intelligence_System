"""
WP-021 Stage 3.3b: offline packaging acceptance of the manifest-driven allowlist.

Derived from the Stage 3.3a experiment script. Everything happens under WORK (outside the
repository). The repository is only read: `git archive` of HEAD, the uncommitted Stage 3.3b
files (pyproject.toml, hatch_build.py) overlaid byte for byte, `git ls-files --eol`, and the
bytes of inventory/extra files, which are hashed and copied. No model is deserialized. Builds
run with --no-isolation under a socket guard (sitecustomize audit hook) and with
package-index access disabled.

Scenarios: M0, M1, M2 (as in 3.3a) x LF/CRLF x 2 repeats; M1S (stale manifest + unlisted files
the .gitignore does not cover), MT (one approved artifact tampered), MM (one approved side-car
missing), LF, once each.

usage: python run_acceptance.py <repo> <work>
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
PY = sys.executable
assert not str(WORK).lower().startswith(str(REPO).lower()), "work directory must be outside the repository"

RESULTS = WORK / "results"
for d in ("src", "trees", "out", "unpacked", "logs", "results", "guard", "probes"):
    (WORK / d).mkdir(parents=True, exist_ok=False)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(*args, binary=False):
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout if binary else p.stdout.decode("utf-8", errors="replace")


def save(name, obj):
    (RESULTS / name).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


def scrub(text: str) -> str:
    for real, label in ((str(WORK), "<work>"), (str(WORK).replace("\\", "/"), "<work>"),
                        (str(REPO), "<repo>"), (str(REPO).replace("\\", "/"), "<repo>"),
                        (PY, "<python>"), (str(Path(PY).parent), "<python-env>"),
                        (str(Path.home()), "<home>")):
        text = text.replace(real, label)
    return text


HEAD = git("rev-parse", "HEAD").strip()

# ── network guard and offline environment ───────────────────────────────────
NET_LOG = WORK / "logs" / "network_attempts.log"
NET_LOG.write_text("", encoding="utf-8")
(WORK / "guard" / "sitecustomize.py").write_text(
    "import os, sys\n"
    "_LOG = os.environ.get('S33A_NET_LOG')\n"
    "def _hook(event, args):\n"
    "    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname', 'urllib.Request'):\n"
    "        if _LOG:\n"
    "            with open(_LOG, 'a', encoding='utf-8') as f:\n"
    "                f.write(event + ' ' + repr(args)[:200] + '\\n')\n"
    "        if event in ('socket.connect', 'urllib.Request'):\n"
    "            raise OSError('network blocked by the stage 3.3a guard')\n"
    "sys.addaudithook(_hook)\n", encoding="utf-8")
ENV = {k: v for k, v in os.environ.items() if k not in ("SOURCE_DATE_EPOCH", "PYTHONPATH", "PIP_INDEX_URL",
                                                         "PIP_EXTRA_INDEX_URL")}
ENV.update({"PYTHONPATH": str(WORK / "guard"), "S33A_NET_LOG": str(NET_LOG), "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "", "PYTHONDONTWRITEBYTECODE": "1"})

# guard self-test: a connect attempt in a child with this environment must be blocked
gt = subprocess.run([PY, "-c", "import socket\ntry:\n    socket.create_connection(('192.0.2.1', 80), timeout=2)\n"
                     "    print('CONNECTED')\nexcept OSError as e:\n    print('BLOCKED', e)"],
                    capture_output=True, text=True, env=ENV, cwd=str(WORK))
GUARD_SELFTEST = gt.stdout.strip()
NET_LOG.write_text("", encoding="utf-8")       # the self-test entry is not a build attempt

VERSIONS = {"python": platform.python_version(), "platform": platform.platform(),
            "build": None, "hatchling": None, "pathspec": None, "packaging": None, "pyproject_hooks": None}
for name in ("build", "hatchling", "pathspec", "packaging", "pyproject_hooks"):
    p = subprocess.run([PY, "-m", "pip", "show", name], capture_output=True, text=True, env=ENV)
    VERSIONS[name] = next((ln.split(":", 1)[1].strip() for ln in p.stdout.splitlines() if ln.startswith("Version:")), None)

# ── exports: LF from git, CRLF by a documented conversion of git-text files ──
tar_bytes = git("-c", "core.autocrlf=false", "archive", "--format=tar", HEAD, binary=True)
LF = WORK / "src" / "LF"
with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
    t.extractall(LF)
# Stage 3.3b changes are not committed: overlay them, byte for byte, from the working tree.
OVERLAY = ["pyproject.toml", "hatch_build.py"]
overlay_record = {}
for rel in OVERLAY:
    data = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")      # LF form, as git would store it
    (LF / rel).write_bytes(data)
    overlay_record[rel] = {"sha256_lf": sha(data), "bytes": len(data)}
eol = git("ls-files", "--eol", "-z").split("\0")
text_lf = ["hatch_build.py"]          # untracked overlay: text, converted like the tracked files
for rec in eol:
    if not rec:
        continue
    meta, path = rec.split("\t", 1)
    fields = meta.split()
    index_eol, attr = fields[0], fields[2]
    if index_eol == "i/lf" and attr in ("attr/", "attr/text", "attr/text=auto"):
        text_lf.append(path)
PROTECTED_PREFIXES = ("fie/models/",)
BINARY_EXT = (".pkl", ".onnx", ".index", ".bin", ".joblib", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".whl",
              ".gz", ".zip")
CRLF = WORK / "src" / "CRLF"
shutil.copytree(LF, CRLF)
converted, skipped_protected = [], []
for rel in sorted(text_lf):
    f = CRLF / rel
    if not f.is_file():
        continue
    if rel.startswith(PROTECTED_PREFIXES) or rel.lower().endswith(BINARY_EXT):
        skipped_protected.append(rel)
        continue
    data = f.read_bytes()
    if b"\0" in data or b"\r\n" in data or b"\n" not in data:
        continue
    f.write_bytes(data.replace(b"\n", b"\r\n"))
    converted.append(rel)


def tree_index(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha(p.read_bytes())
    return out


lf_index, crlf_index = tree_index(LF), tree_index(CRLF)
changed = sorted(k for k in lf_index if lf_index[k] != crlf_index.get(k))
assert changed == converted, "only the documented conversions may differ"
save("exports.json", {
    "head": HEAD, "archive_command": "git -c core.autocrlf=false archive --format=tar HEAD",
    "archive_sha256": sha(tar_bytes), "archive_bytes": len(tar_bytes), "files_in_export": len(lf_index),
    "fie_models_in_export": sorted(k for k in lf_index if k.startswith("fie/models/")),
    "crlf_procedure": "copy of the LF export; every git-tracked file that `git ls-files --eol` reports as i/lf with no "
                      "eol attribute, is not under fie/models/, has no binary extension, contains no NUL and has at "
                      "least one LF, gets every LF replaced by CRLF. Model artifacts and the generated manifest are "
                      "added later, byte-identical in both variants.",
    "crlf_converted_files": converted, "crlf_converted_count": len(converted),
    "skipped_as_protected_or_binary": skipped_protected,
    "only_converted_files_differ": True,
    "overlay_from_working_tree": overlay_record})

# ── inventory and extras, verified by size and SHA-256 (bytes only) ─────────
inventory = json.loads(git("show", f"{HEAD}:scripts/model_manifest.json"))
spec = importlib.util.spec_from_file_location("_mi_s33a", LF / "fie" / "_model_integrity.py")
mi = importlib.util.module_from_spec(spec)
sys.modules["_mi_s33a"] = mi
spec.loader.exec_module(mi)
tracked_manifest_bytes = git("show", f"{HEAD}:scripts/model_manifest.json", binary=True)
mi.validate_production(tracked_manifest_bytes)
DERIVED = mi.derive_package_manifest(tracked_manifest_bytes)
existing = (REPO / "fie" / "models" / "MANIFEST.json")
INV_BYTES, inv_checks = {}, []
for a in inventory["artifacts"]:
    src = REPO / a["path"]
    data = src.read_bytes() if src.is_file() else None
    ok = data is not None and len(data) == a["size"] and sha(data) == a["sha256"]
    inv_checks.append({"path": a["path"], "role": a["role"], "version": a.get("version"), "in_wheel": a["in_wheel"],
                       "expected_size": a["size"], "expected_sha256": a["sha256"],
                       "actual_size": None if data is None else len(data),
                       "actual_sha256": None if data is None else sha(data), "verified": ok})
    if ok:
        INV_BYTES[a["path"]] = data
EXPERIMENT_NAMES = ("v2", "v3_w3x", "v6_1_contaminated", "v6_noaug", "v64_harmaug", "v64_harmaug_capped", "v65_benignaug")
EXTRA_PATHS = [f"fie/models/pair_intent_classifier_{n}.pkl" for n in EXPERIMENT_NAMES] + \
              [f"fie/models/pair_intent_meta_{n}.json" for n in EXPERIMENT_NAMES] + \
              [f"fie/models/minilm-onnx/{n}" for n in ("config.json", "ort_config.json", "special_tokens_map.json",
                                                         "tokenizer_config.json", "vocab.txt")]
EXTRA_BYTES, extra_checks = {}, []
for rel in EXTRA_PATHS:
    src = REPO / rel
    if src.is_file():
        data = src.read_bytes()
        EXTRA_BYTES[rel] = data
        extra_checks.append({"path": rel, "found": True, "size": len(data), "sha256": sha(data)})
    else:
        extra_checks.append({"path": rel, "found": False})
M1_COMPLETE = len(INV_BYTES) == len(inventory["artifacts"])
M2_COMPLETE = M1_COMPLETE and len(EXTRA_BYTES) == len(EXTRA_PATHS)
save("scenario_inputs.json", {
    "inventory_entries": len(inventory["artifacts"]), "inventory_verified": len(INV_BYTES),
    "inventory_checks": inv_checks, "extras_expected": len(EXTRA_PATHS), "extras_found": len(EXTRA_BYTES),
    "extra_checks": extra_checks,
    "derived_manifest": {"sha256": sha(DERIVED), "bytes": len(DERIVED),
                         "equals_repository_generated_file": existing.is_file() and existing.read_bytes() == DERIVED,
                         "derivation": "fie/_model_integrity.py (from the export) derive_package_manifest(tracked inventory at HEAD)"},
    "m1_complete": M1_COMPLETE, "m2_complete": M2_COMPLETE,
    "root_level_tracked_model_dir_files": sorted(k for k in lf_index if k.startswith("models/")),
    "deserialized": "nothing; files were read as bytes, hashed and copied"})


def populate(tree: Path, scenario: str) -> list[str]:
    added = []
    if scenario == "M0":
        for p in list((tree / "fie" / "models").rglob("*")):
            if p.is_file() and p.name != ".gitkeep":
                raise AssertionError(f"unexpected file in a git export: {p}")
        return added
    for rel, data in INV_BYTES.items():
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        added.append(rel)
    (tree / "fie" / "models" / "MANIFEST.json").write_bytes(DERIVED)
    added.append("fie/models/MANIFEST.json")
    if scenario == "M1S":
        # a stale generated manifest, and unlisted files the .gitignore does NOT cover
        (tree / "fie" / "models" / "MANIFEST.json").write_bytes(b'{"stale": "must not ship"}\n')
        for rel in ("fie/models/sub/nested_unlisted.json", "fie/models/notes.txt"):
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes(b"S33B fixture, not a model, must not ship\n")
            added.append(rel)
    if scenario == "MT":
        # one approved artifact altered by a single byte (synthetic tamper; never deserialized)
        victim = tree / "fie" / "models" / "pair_intent_classifier_v6_3b.pkl"
        data = victim.read_bytes()
        victim.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        added.append("TAMPERED fie/models/pair_intent_classifier_v6_3b.pkl (first byte flipped)")
    if scenario == "MM":
        (tree / "fie" / "models" / "meta_clf.json").unlink()
        added.append("REMOVED fie/models/meta_clf.json")
    if scenario == "M2":
        for rel, data in EXTRA_BYTES.items():
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            added.append(rel)
    return added


def snapshot(tree: Path) -> dict:
    return {p.relative_to(tree).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in tree.rglob("*") if p.is_file()}


def run_build(kind: str, src: Path, outdir: Path, label: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=False)
    cmd = [PY, "-m", "build", "--no-isolation", f"--{kind}", "--outdir", str(outdir), str(src)]
    before = NET_LOG.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, cwd=str(WORK))
    elapsed = round(time.perf_counter() - t0, 2)
    log = WORK / "logs" / f"{label}.log"
    log.write_text(scrub(f"$ {' '.join(cmd)}\nexit={p.returncode}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"),
                   encoding="utf-8", newline="\n")
    outputs = sorted(outdir.iterdir())
    return {"label": label, "kind": kind, "command": scrub(" ".join(cmd)), "exit_code": p.returncode,
            "elapsed_s": elapsed, "log": f"logs/{label}.log",
            "outputs": [{"name": o.name, "sha256": sha(o.read_bytes()), "bytes": o.stat().st_size} for o in outputs],
            "network_attempts": NET_LOG.read_text(encoding="utf-8")[len(before):].splitlines(),
            "_paths": [str(o) for o in outputs]}


def members(archive: Path) -> dict:
    """Every member by name; size and SHA-256 of each regular member (bytes only, never deserialized)."""
    out = {}
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as z:
            for i in z.infolist():
                if i.is_dir():
                    continue
                data = z.read(i)
                out[i.filename] = {"size": len(data), "sha256": sha(data),
                                   "date_time": "%04d-%02d-%02dT%02d:%02d:%02d" % i.date_time}
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                data = t.extractfile(m).read()
                out[m.name.split("/", 1)[1] if "/" in m.name else m.name] = {
                    "size": len(data), "sha256": sha(data), "mtime": m.mtime, "mode": oct(m.mode)}
    return out


SUPPORTED = [a["path"] for a in inventory["artifacts"] if a["in_wheel"]]
CHECK_GROUPS = {
    "supported_pair_and_meta (in_wheel true)": SUPPORTED,
    "runtime_manifest": ["fie/models/MANIFEST.json"],
    "legacy (in_wheel false)": [a["path"] for a in inventory["artifacts"] if a["role"] == "legacy"],
    "gitkeep": ["fie/models/.gitkeep"],
    "experiment_files": [p for p in EXTRA_PATHS if "minilm-onnx" not in p],
    "encoder_by_products": [p for p in EXTRA_PATHS if "minilm-onnx" in p],
    "encoder_tokenizer (in_wheel false)": [a["path"] for a in inventory["artifacts"] if a["role"] in ("encoder", "tokenizer")],
    "server (in_wheel false)": [a["path"] for a in inventory["artifacts"] if a["role"] == "server"],
}


def classify(mem: dict) -> dict:
    names = set(mem)
    res = {}
    for group, paths in CHECK_GROUPS.items():
        present = sorted(p for p in paths if p in names)
        res[group] = {"expected_count": len(paths), "present": present,
                      "absent": sorted(p for p in paths if p not in names)}
    fm = sorted(n for n in names if n.startswith("fie/models/"))
    allow = set(SUPPORTED) | {"fie/models/MANIFEST.json"}
    res["fie_models_members"] = fm
    res["vs_proposed_allowlist"] = {"allowlist_size": len(allow),
                                    "extra_vs_allowlist": sorted(set(fm) - allow),
                                    "missing_vs_allowlist": sorted(allow - set(fm))}
    return res


# ── the matrix ──────────────────────────────────────────────────────────────
runs = []
MATRIX = [(s, v, (1, 2)) for s in ("M0", "M1", "M2") for v in ("LF", "CRLF")] +          [(s, "LF", (1,)) for s in ("M1S", "MT", "MM")]
for scenario, variant, reps in MATRIX:
    feasible = scenario == "M0" or (scenario == "M2" and M2_COMPLETE) or (scenario != "M2" and M1_COMPLETE)
    for _ in (0,):
        for rep in reps:
            setup = f"{scenario}_{variant}_r{rep}"
            if not feasible:
                runs.append({"setup": setup, "blocked": "required inputs not verified"})
                continue
            tree = WORK / "trees" / setup
            shutil.copytree(WORK / "src" / variant, tree)
            added = populate(tree, scenario)
            snap0 = snapshot(tree)
            rec = {"setup": setup, "scenario": scenario, "variant": variant, "repeat": rep, "added_files": added}
            rec["sdist"] = run_build("sdist", tree, WORK / "out" / setup / "sdist", f"{setup}_sdist")
            rec["wheel_direct"] = run_build("wheel", tree, WORK / "out" / setup / "wheel_direct", f"{setup}_wheel_direct")
            snap1 = snapshot(tree)
            rec["source_tree_unchanged_by_builds"] = snap0 == snap1
            rec["source_tree_new_files"] = sorted(set(snap1) - set(snap0))
            sd = [Path(p) for p in rec["sdist"].pop("_paths") if p.endswith(".tar.gz")]
            wd = [Path(p) for p in rec["wheel_direct"].pop("_paths") if p.endswith(".whl")]
            rec["sdist_members"] = members(sd[0]) if sd else None
            rec["wheel_direct_members"] = members(wd[0]) if wd else None
            if sd:
                up = WORK / "unpacked" / setup
                up.mkdir(parents=True)
                with tarfile.open(sd[0]) as t:
                    t.extractall(up)
                root = next(up.iterdir())
                rec["sdist_unpacked_has_gitignore"] = (root / ".gitignore").is_file()
                rec["sdist_unpacked_has_dot_git"] = (root / ".git").exists()
                rec["wheel_from_sdist"] = run_build("wheel", root, WORK / "out" / setup / "wheel_from_sdist",
                                                    f"{setup}_wheel_from_sdist")
                ws = [Path(p) for p in rec["wheel_from_sdist"].pop("_paths") if p.endswith(".whl")]
                rec["wheel_from_sdist_members"] = members(ws[0]) if ws else None
            else:
                rec["wheel_from_sdist"] = {"not_applicable": "no sdist was produced (its build failed), so there is nothing to build a wheel from"}
                rec["wheel_from_sdist_members"] = None
            for key in ("sdist_members", "wheel_direct_members", "wheel_from_sdist_members"):
                rec[key.replace("_members", "_classification")] = classify(rec[key]) if rec[key] else None
            runs.append(rec)
            shutil.rmtree(tree)          # keep archives and logs; drop the 100+ MB source copy
            print(setup, rec["sdist"]["exit_code"], rec["wheel_direct"]["exit_code"],
                  rec["wheel_from_sdist"].get("exit_code"), flush=True)
save("builds.json", {"head": HEAD, "versions": VERSIONS, "guard_selftest": GUARD_SELFTEST, "runs": runs})

print("network attempts:", NET_LOG.read_text(encoding="utf-8").count(chr(10)))
print("done")
