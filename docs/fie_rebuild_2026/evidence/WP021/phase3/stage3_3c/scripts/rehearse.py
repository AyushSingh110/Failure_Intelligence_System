"""
WP-021 Stage 3.3c: offline rehearsal of the CI `package` job and the publish workflow's gate.

For each scenario, in a temporary export of HEAD (never the repository): build the sdist, run the
archive checker on it, unpack that sdist, build the wheel from it, run the checker on both; also
build a direct wheel and check it. Builds use --no-isolation with the installed build tools, under
a socket-refusing audit hook. Negative controls: M0 (no models) must not build; archives from
before Stage 3.3b (Stage 3.3a outputs) must be refused by the checker. No model is deserialized.

usage: python rehearse.py <repo> <work> <stage-3.3a-work>
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
from pathlib import Path

REPO, WORK, OLD = (Path(a).resolve() for a in sys.argv[1:4])
PY = sys.executable
CHECKER = REPO / "scripts" / "verify_package_artifacts.py"
assert not str(WORK).lower().startswith(str(REPO).lower())
for d in ("src", "trees", "out", "unpacked", "logs", "results", "guard", "reports"):
    (WORK / d).mkdir(parents=True, exist_ok=False)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def scrub(t):
    for real, label in ((str(WORK), "<work>"), (str(WORK).replace("\\", "/"), "<work>"), (str(REPO), "<repo>"),
                        (str(REPO).replace("\\", "/"), "<repo>"), (str(OLD), "<3.3a-work>"),
                        (str(OLD).replace("\\", "/"), "<3.3a-work>"), (PY, "<python>"),
                        (str(Path(PY).parent), "<python-env>"), (str(Path.home()), "<home>")):
        t = t.replace(real, label)
    return t


def git(*a, binary=False):
    p = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True)
    return p.stdout if binary else p.stdout.decode()


HEAD = git("rev-parse", "HEAD").strip()
NET = WORK / "logs" / "network_attempts.log"
NET.write_text("", encoding="utf-8")
(WORK / "guard" / "sitecustomize.py").write_text(
    "import os, sys\n_L = os.environ.get('S33C_NET_LOG')\n"
    "def _h(e, a):\n"
    "    if e in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):\n"
    "        if _L:\n            open(_L, 'a', encoding='utf-8').write(e + ' ' + repr(a)[:200] + '\\n')\n"
    "        if e in ('socket.connect', 'urllib.Request'):\n"
    "            raise OSError('network blocked by the stage 3.3c guard')\n"
    "sys.addaudithook(_h)\n", encoding="utf-8")
ENV = {k: v for k, v in os.environ.items() if k not in ("SOURCE_DATE_EPOCH", "PYTHONPATH")}
ENV.update({"PYTHONPATH": str(WORK / "guard"), "S33C_NET_LOG": str(NET), "PIP_NO_INDEX": "1",
            "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "PYTHONDONTWRITEBYTECODE": "1"})

# exports: LF from git; CRLF by converting git-text files (as in 3.3a/3.3b)
LF, CRLF = WORK / "src" / "LF", WORK / "src" / "CRLF"
with tarfile.open(fileobj=io.BytesIO(git("-c", "core.autocrlf=false", "archive", "--format=tar", HEAD, binary=True))) as t:
    t.extractall(LF)
shutil.copytree(LF, CRLF)
converted = []
for rec in git("ls-files", "--eol", "-z").split("\0"):
    if not rec:
        continue
    meta, rel = rec.split("\t", 1)
    f = CRLF / rel
    if meta.split()[0] != "i/lf" or rel.startswith("fie/models/") or not f.is_file():
        continue
    data = f.read_bytes()
    if b"\0" in data or b"\r\n" in data or b"\n" not in data:
        continue
    f.write_bytes(data.replace(b"\n", b"\r\n"))
    converted.append(rel)

inventory = json.loads((LF / "scripts" / "model_manifest.json").read_text(encoding="utf-8"))
INV = {}
for a in inventory["artifacts"]:
    b = (REPO / a["path"]).read_bytes()
    assert len(b) == a["size"] and sha(b) == a["sha256"], f"inventory artifact not verified: {a['path']}"
    INV[a["path"]] = b
EXTRAS = {}
for n in ("v2", "v3_w3x", "v6_1_contaminated", "v6_noaug", "v64_harmaug", "v64_harmaug_capped", "v65_benignaug"):
    for rel in (f"fie/models/pair_intent_classifier_{n}.pkl", f"fie/models/pair_intent_meta_{n}.json"):
        EXTRAS[rel] = (REPO / rel).read_bytes()
for n in ("config.json", "ort_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"):
    EXTRAS[f"fie/models/minilm-onnx/{n}"] = (REPO / "fie/models/minilm-onnx" / n).read_bytes()


def populate(tree, scenario):
    if scenario == "M0":
        return
    for rel, b in INV.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(b)
    (tree / "fie/models/MANIFEST.json").write_bytes(b'{"stale": "must not ship"}\n')   # the hook must ignore it
    if scenario == "M2":
        for rel, b in EXTRAS.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes(b)


def run(cmd, label):
    before = NET.read_text(encoding="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, cwd=str(WORK))
    (WORK / "logs" / f"{label}.log").write_text(scrub(f"$ {' '.join(map(str, cmd))}\nexit={p.returncode}\n"
                                                      f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"),
                                                encoding="utf-8", newline="\n")
    return {"label": label, "command": scrub(" ".join(map(str, cmd))), "exit_code": p.returncode,
            "network_attempts": NET.read_text(encoding="utf-8")[len(before):].splitlines()}


def checker(label, sdist=None, wheel=None):
    args = [PY, "-I", str(CHECKER), "--report", str(WORK / "reports" / f"{label}.json")]
    if sdist:
        args += ["--sdist", str(sdist)]
    if wheel:
        args += ["--wheel", str(wheel)]
    r = run(args, f"check_{label}")
    rep = WORK / "reports" / f"{label}.json"
    r["report"] = f"reports/{label}.json" if rep.exists() else None
    return r


def one(d, pattern):
    found = sorted(d.glob(pattern))
    return found[0] if len(found) == 1 else None


runs = []
for scenario, variant, rep in [("M1", "LF", 1), ("M1", "LF", 2), ("M1", "CRLF", 1), ("M2", "LF", 1),
                               ("M2", "CRLF", 1), ("M0", "LF", 1)]:
    setup = f"{scenario}_{variant}_r{rep}"
    tree = WORK / "trees" / setup
    shutil.copytree(WORK / "src" / variant, tree)
    populate(tree, scenario)
    out = WORK / "out" / setup
    rec = {"setup": setup}
    # 1. the sdist, then the gate on it (the CI/publish order)
    rec["build_sdist"] = run([PY, "-m", "build", "--no-isolation", "--sdist", "--outdir", str(out / "dist"), str(tree)],
                             f"{setup}_build_sdist")
    sd = one(out / "dist", "*.tar.gz") if (out / "dist").exists() else None
    if sd:
        rec["check_sdist"] = checker(f"{setup}_sdist", sdist=sd)
        # 2. the wheel from that very sdist, then the gate on both
        up = WORK / "unpacked" / setup
        up.mkdir(parents=True)
        with tarfile.open(sd) as t:
            t.extractall(up)
        srcs = [p for p in up.iterdir() if p.is_dir()]
        rec["build_wheel_from_sdist"] = run([PY, "-m", "build", "--no-isolation", "--wheel", "--outdir",
                                             str(out / "dist"), str(srcs[0])], f"{setup}_build_wheel_from_sdist")
        wh = one(out / "dist", "*.whl")
        rec["check_sdist_and_wheel"] = checker(f"{setup}_sdist_and_wheel", sdist=sd, wheel=wh) if wh else None
        rec["dist"] = {p.name: {"sha256": sha(p.read_bytes()), "bytes": p.stat().st_size} for p in (out / "dist").iterdir()}
    # 3. a direct wheel from the tree, for comparison
    rec["build_wheel_direct"] = run([PY, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(out / "direct"),
                                     str(tree)], f"{setup}_build_wheel_direct")
    wd = one(out / "direct", "*.whl") if (out / "direct").exists() else None
    if wd:
        rec["check_wheel_direct"] = checker(f"{setup}_wheel_direct", wheel=wd)
        rec["wheel_direct"] = {"sha256": sha(wd.read_bytes()), "bytes": wd.stat().st_size}
    runs.append(rec)
    shutil.rmtree(tree)
    print(setup, rec["build_sdist"]["exit_code"], rec.get("check_sdist", {}).get("exit_code"),
          (rec.get("build_wheel_from_sdist") or {}).get("exit_code"),
          (rec.get("check_sdist_and_wheel") or {}).get("exit_code"), rec["build_wheel_direct"]["exit_code"],
          (rec.get("check_wheel_direct") or {}).get("exit_code"), flush=True)

# negative controls on real archives built before Stage 3.3b (Stage 3.3a outputs, current-config era)
old = []
for setup in ("M1_LF_r1", "M2_LF_r1"):
    sd = one(OLD / "out" / setup / "sdist", "*.tar.gz")
    wh = one(OLD / "out" / setup / "wheel_direct", "*.whl")
    old.append({"source": f"stage 3.3a {setup}", **checker(f"pre33b_{setup}", sdist=sd, wheel=wh)})
    print("pre-3.3b", setup, old[-1]["exit_code"], flush=True)

(WORK / "results" / "rehearsal.json").write_text(json.dumps({
    "head": HEAD, "crlf_converted_count": len(converted), "runs": runs, "pre_3_3b_negative_controls": old,
    "network_attempts_total": NET.read_text(encoding="utf-8").count(chr(10))}, indent=1, sort_keys=True),
    encoding="utf-8", newline="\n")
print("network attempts:", NET.read_text(encoding="utf-8").count(chr(10)))
print("done")
