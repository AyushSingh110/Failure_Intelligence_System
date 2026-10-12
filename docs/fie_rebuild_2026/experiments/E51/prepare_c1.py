"""
E51 Stage C1 - connected provisioning and artifact preparation. Run in the repository
environment, from the repository root, WHILE THE NETWORK IS UP:

    python docs/fie_rebuild_2026/experiments/E51/prepare_c1.py

Scope (owner authorization for Stage C1): BASE dependency set only. No [ml] package is
resolved for download, downloaded or installed here. Nothing is installed into any
measurement environment: installation happens later, offline, in offline_runner.py.

Steps, each stopping the script on failure:
  1  re-resolve the base set; it must equal the approved pins (no silent refresh)
  2  wheelhouse-B  : PyPI wheels (hash-required download) + the locally built langdetect wheel
  3  wheelhouse-A  : the same PyPI wheels + the langdetect SOURCE ARCHIVE + build tools;
                     no langdetect wheel
  4  the published fie-sdk 1.18.0 wheel from PyPI, digest-checked
  5  A-clean and A-release built twice each from `git archive <commit>`, --no-isolation
  6  static inspection of the four wheels (PyPI, clean, release, dist/)
  7  provenance ledgers and hashed pin files per installation mode
  8  measurement environments (launchers only; nothing installed)
  9  positive connectivity control through every launcher and the base interpreter
 10  input export

Everything temporary goes under the scratch directory. The repository gains only evidence
files under docs/fie_rebuild_2026/evidence/E51/stageC1/.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path.cwd()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import provision  # noqa: E402  (shared flags, neutralized pip environment, redaction)

COMMIT = "2f62ecd"
SCRATCH = Path(r"C:\Users\ASUS\fie-e51-scratch")
EVID = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageC1"
APPROVED_PINS = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageB_remediation" / "pins" / "base.txt"
LOCAL_WHEEL = SCRATCH / "local-wheels" / "langdetect" / "langdetect-1.0.9-py3-none-any.whl"
SDIST = SCRATCH / "sdist" / "langdetect-1.0.9.tar.gz"
BUILD_CACHE = SCRATCH / "build-cache"
FIE_PYPI_SHA = "be504dc748822699df5dc5353ad0038617ef639a4c1840ed18786bc8579e963d"
LOCAL_WHEEL_SHA = "9acdfda7ee22ffdaaaf87f993cec0eb186d772fe0f3b93469832d83be6e9c8a4"
SDIST_SHA = "cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0"
BASE_REQUIREMENTS = ["requests>=2.28.0", "deep-translator>=1.11.4", "langdetect>=1.0.9"]

CELLS = {   # name -> (artifact, install mode, venv created with its bundled pip?)
    "modeA-pypi-base-r1": ("A-pypi-1.18.0", "A-source-build", True),
    "modeA-pypi-base-r2": ("A-pypi-1.18.0", "A-source-build", True),
    "modeB-pypi-base":    ("A-pypi-1.18.0", "B-prebuilt-local-wheel", False),
    "modeB-clean-base":   ("A-clean", "B-prebuilt-local-wheel", False),
    "modeB-release-base": ("A-release", "B-prebuilt-local-wheel", False),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(message: str) -> None:
    print("STOP:", message)
    provision.save(EVID / "provisioning" / "STOPPED.json", {"reason": message})
    sys.exit(2)


def say(step: str, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", **kwargs)


def pypi_digest(name: str, version: str, filename: str) -> dict:
    meta = provision.fetch_json(f"{provision.JSON_API}/{name}/{version}/json")
    for f in meta["urls"]:
        if f["filename"] == filename:
            return {"sha256": f["digests"]["sha256"], "size": f["size"], "url": f["url"],
                    "host": re.match(r"https://([^/]+)/", f["url"]).group(1), "yanked": f.get("yanked", False)}
    return {}


def fresh_dir(path: Path) -> Path:
    if path.exists():
        stop(f"{path} already exists; this script does not overwrite or delete existing scratch content")
    path.mkdir(parents=True)
    return path


def main() -> int:
    prov = EVID / "provisioning"
    prov.mkdir(parents=True, exist_ok=True)
    if not SCRATCH.is_dir():
        stop("the scratch directory does not exist")
    for needed in (LOCAL_WHEEL, SDIST, BUILD_CACHE, APPROVED_PINS):
        if not needed.exists():
            stop(f"missing input from Stage B: {needed.name}")
    if sha256(LOCAL_WHEEL) != LOCAL_WHEEL_SHA or sha256(SDIST) != SDIST_SHA:
        stop("the langdetect wheel or source archive in the scratch directory is not the one recorded in Stage B")
    py = sys.executable
    provision_env = SCRATCH / "env" / "_provision" / "Scripts" / "python.exe"

    # ── 1. re-resolve; must equal the approved pins ──────────────────────────
    r = run([py, "-B", HERE / "provision.py", "resolve", "--name", "base", "--python", provision_env,
             "--out", prov, "--local-wheel", LOCAL_WHEEL, *BASE_REQUIREMENTS])
    if r.returncode != 0:
        stop("base resolution failed: " + r.stdout[-400:])
    new_pins = (prov / "pins" / "base.txt").read_text(encoding="utf-8")
    if new_pins != APPROVED_PINS.read_text(encoding="utf-8"):
        stop("the base resolution no longer equals the approved pins; pins are not refreshed silently")
    base = json.loads((prov / "resolve" / "base.summary.json").read_text(encoding="utf-8"))
    say("1", f"base set re-resolved: {base['distributions']} distributions, identical to the approved pins")

    # ── 2. wheelhouse-B ──────────────────────────────────────────────────────
    wh_b = fresh_dir(SCRATCH / "wheelhouse-B")
    r = run([py, "-B", HERE / "provision.py", "download", "--name", "base", "--python", provision_env,
             "--out", prov, "--cache", wh_b, "--local-wheel", LOCAL_WHEEL])
    if r.returncode != 0:
        stop("hash-required download of the base set failed: " + r.stdout[-400:])
    say("2", f"wheelhouse-B: {len(list(wh_b.iterdir()))} files, all hash-verified")

    # ── 3. wheelhouse-A ──────────────────────────────────────────────────────
    wh_a = fresh_dir(SCRATCH / "wheelhouse-A")
    ledger_a, ledger_b = [], []
    for item in base["items"]:
        src = wh_b / item["filename"]
        entry = {"filename": item["filename"], "name": item["name"], "version": item["version"],
                 "sha256": sha256(src), "size": src.stat().st_size}
        if item.get("origin") == "pypi":
            entry.update({"origin": "pypi-wheel", "pypi_sha256": item["pypi_sha256"], "url": item["url"],
                          "matches_pypi_digest": entry["sha256"] == item["pypi_sha256"]})
            shutil.copyfile(src, wh_a / item["filename"])
            ledger_a.append(dict(entry))
        else:
            entry.update({"origin": "local-build", "built_from_source_sha256": SDIST_SHA, "pypi_sha256": None,
                          "note": "built in Stage B remediation from the digest-verified PyPI source archive; "
                                  "not a PyPI-published wheel"})
        ledger_b.append(entry)
    sd = pypi_digest("langdetect", "1.0.9", SDIST.name)
    if not sd or sd["sha256"] != SDIST_SHA or sd["host"] != provision.FILE_HOST:
        stop("the langdetect source archive no longer matches PyPI's published digest")
    shutil.copyfile(SDIST, wh_a / SDIST.name)
    ledger_a.append({"filename": SDIST.name, "name": "langdetect", "version": "1.0.9", "sha256": sha256(wh_a / SDIST.name),
                     "size": SDIST.stat().st_size, "origin": "pypi-sdist", "pypi_sha256": sd["sha256"],
                     "url": sd["url"], "matches_pypi_digest": True})
    for tool in sorted(BUILD_CACHE.iterdir()):
        m = re.match(r"([A-Za-z0-9_.]+?)-(\d[^-]*)-", tool.name)
        digest = pypi_digest(m.group(1), m.group(2), tool.name)
        if not digest or digest["sha256"] != sha256(tool) or digest["host"] != provision.FILE_HOST:
            stop(f"build tool {tool.name} does not match PyPI's published digest")
        shutil.copyfile(tool, wh_a / tool.name)
        ledger_a.append({"filename": tool.name, "name": m.group(1), "version": m.group(2), "sha256": sha256(tool),
                         "size": tool.stat().st_size, "origin": "pypi-wheel", "role": "build tool",
                         "pypi_sha256": digest["sha256"], "url": digest["url"], "matches_pypi_digest": True})
    say("3", f"wheelhouse-A: {len(list(wh_a.iterdir()))} files (source archive + build tools, no langdetect wheel)")

    # ── 4. the published wheel ───────────────────────────────────────────────
    wheels = SCRATCH / "wheels"
    pypi_dir = fresh_dir(wheels / "pypi")
    cmd = [py, "-m", "pip", "--python", provision_env, "download", "fie-sdk==1.18.0", "--no-deps",
           *provision.PIP_FLAGS, "-d", pypi_dir]
    r = run(cmd, env=provision.pip_env())
    provision.save(prov / "download" / "fie-sdk.pip.log", "COMMAND: " + " ".join(map(str, cmd)) +
                   f"\nEXIT: {r.returncode}\n\n" + r.stdout + "\n--- stderr ---\n" + r.stderr)
    hosts = sorted(set(re.findall(r"https?://([A-Za-z0-9.\-]+)", r.stdout + r.stderr)))
    fie_pypi = pypi_dir / "fie_sdk-1.18.0-py3-none-any.whl"
    if r.returncode != 0 or not fie_pypi.exists() or sha256(fie_pypi) != FIE_PYPI_SHA:
        stop("the published fie-sdk wheel could not be downloaded with the expected digest")
    if [h for h in hosts if h not in provision.APPROVED_HOSTS]:
        stop(f"unexpected host while downloading the published wheel: {hosts}")
    say("4", f"published wheel downloaded and digest-checked ({fie_pypi.stat().st_size} bytes)")

    # ── 5. local builds, twice each ──────────────────────────────────────────
    epoch = run(["git", "log", "-1", "--format=%ct", COMMIT]).stdout.strip()
    full = run(["git", "rev-parse", COMMIT]).stdout.strip()
    archive = subprocess.run(["git", "archive", "--format=tar", COMMIT], capture_output=True, check=True).stdout
    src_clean = fresh_dir(SCRATCH / "src-clean")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(src_clean)
    src_release = SCRATCH / "src-release"
    if src_release.exists():
        stop("src-release already exists")
    shutil.copytree(src_clean, src_release)
    manifest = json.loads((REPO / "scripts" / "model_manifest.json").read_text(encoding="utf-8"))
    copied = []
    for art in manifest["artifacts"]:
        rel = Path(art["path"])
        if rel.parent.as_posix() != "fie/models":          # only what the wheel's build rule can pick up
            continue
        src = REPO / rel
        if not src.exists() or sha256(src) != art["sha256"]:
            stop(f"model file {art['path']} is missing or does not match the manifest")
        shutil.copyfile(src, src_release / rel)
        copied.append({"path": art["path"], "sha256": art["sha256"]})
    build_env = {**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull}
    builds = {}
    for label, src in (("clean", src_clean), ("release", src_release)):
        hashes = []
        for n in (1, 2):
            outdir = fresh_dir(wheels / f"{label}-build{n}")
            cmd = [py, "-m", "build", "--wheel", "--no-isolation", "--outdir", outdir, src]
            r = run(cmd, env=build_env)
            provision.save(prov / "build" / f"{label}-build{n}.log", "COMMAND: " + " ".join(map(str, cmd)) +
                           f"\nSOURCE_DATE_EPOCH={epoch}\nEXIT: {r.returncode}\n\n" + r.stdout + "\n--- stderr ---\n"
                           + r.stderr)
            built = list(outdir.glob("*.whl"))
            if r.returncode != 0 or len(built) != 1:
                stop(f"building the {label} wheel failed (run {n})")
            hashes.append({"file": built[0].name, "sha256": sha256(built[0]), "size": built[0].stat().st_size})
            shutil.rmtree(src / "dist", ignore_errors=True)
        builds[label] = {"runs": hashes, "identical": hashes[0]["sha256"] == hashes[1]["sha256"]}
        say("5", f"{label}: {hashes[0]['sha256'][:16]} / {hashes[1]['sha256'][:16]}  identical={builds[label]['identical']}")
    provision.save(prov / "build" / "local_builds.json", {
        "commit": full, "source": f"git archive {COMMIT}", "source_date_epoch": epoch,
        "build_command": "python -m build --wheel --no-isolation", "builds": builds,
        "release_model_files_copied": copied, "release_model_files_count": len(copied)})
    fie_wheels = {"A-pypi-1.18.0": fie_pypi,
                  "A-clean": next((wheels / "clean-build1").glob("*.whl")),
                  "A-release": next((wheels / "release-build1").glob("*.whl"))}

    # ── 6. static inspection ─────────────────────────────────────────────────
    dist_wheel = REPO / "dist" / "fie_sdk-1.18.0-py3-none-any.whl"
    dist_before = sha256(dist_wheel)
    r = run([py, "-B", HERE / "inspect_wheel.py", "--manifest", REPO / "scripts" / "model_manifest.json",
             "--out", EVID / "artifacts" / "wheels.json",
             *[f"{label}={path}" for label, path in fie_wheels.items()], f"A-dist={dist_wheel}"])
    if r.returncode != 0 or sha256(dist_wheel) != dist_before:
        stop("static wheel inspection failed: " + r.stderr[-300:])
    inspected = json.loads((EVID / "artifacts" / "wheels.json").read_text(encoding="utf-8"))
    say("6", "inspected: " + ", ".join(f"{k} {v['sha256'][:12]}" for k, v in inspected.items()))

    # ── 7. ledgers and pins ──────────────────────────────────────────────────
    pin_lines = {i["name"].lower(): f"{i['name']}=={i['version']} --hash=sha256:{i['sha256']}" for i in base["items"]}
    pins_dir = SCRATCH / "pins"
    pins_dir.mkdir(exist_ok=True)

    def write_pins(name: str, fie_sha: str, langdetect_sha: str) -> Path:
        lines = dict(pin_lines)
        lines["langdetect"] = f"langdetect==1.0.9 --hash=sha256:{langdetect_sha}"
        lines["fie-sdk"] = f"fie-sdk==1.18.0 --hash=sha256:{fie_sha}"
        text = "".join(lines[k] + "\n" for k in sorted(lines))
        (pins_dir / name).write_text(text, encoding="utf-8", newline="\n")
        provision.save(EVID / "pins" / name, text)
        return pins_dir / name

    cells = {}
    for cell, (artifact, mode, with_pip) in CELLS.items():
        wheel = fie_wheels[artifact]
        if mode == "A-source-build":
            pins = write_pins(f"pins-A-{cell}.txt", sha256(wheel), SDIST_SHA)
            houses = [wh_a, wheel.parent]
        else:
            pins = write_pins(f"pins-B-{cell}.txt", sha256(wheel), LOCAL_WHEEL_SHA)
            houses = [wh_b, wheel.parent]
        cells[cell] = {"artifact": artifact, "install_mode": mode, "bundled_pip": with_pip,
                       "fie_wheel": str(wheel), "fie_wheel_sha256": sha256(wheel), "pins": str(pins),
                       "find_links": [str(h) for h in houses], "env": str(SCRATCH / "env" / cell)}
    if any(p.name.startswith("langdetect") and p.suffix == ".whl" for p in wh_a.iterdir()):
        stop("wheelhouse-A contains a langdetect wheel")
    if any(p.name.endswith(".tar.gz") for p in wh_b.iterdir()):
        stop("wheelhouse-B contains a source archive")
    provision.save(EVID / "provenance" / "wheelhouse-A.ledger.json", {
        "install_mode": "A-source-build", "files": sorted(ledger_a, key=lambda d: d["filename"]),
        "contains_langdetect_wheel": False, "file_count": len(ledger_a),
        "listing": sorted(p.name for p in wh_a.iterdir())})
    provision.save(EVID / "provenance" / "wheelhouse-B.ledger.json", {
        "install_mode": "B-prebuilt-local-wheel", "files": sorted(ledger_b, key=lambda d: d["filename"]),
        "contains_source_archive": False, "file_count": len(ledger_b),
        "listing": sorted(p.name for p in wh_b.iterdir())})
    provision.save(EVID / "provenance" / "fie_wheels.ledger.json", {
        label: {"sha256": sha256(path), "size": path.stat().st_size,
                "origin": "pypi-wheel" if label.startswith("A-pypi") else f"local-build from git archive {COMMIT}"}
        for label, path in fie_wheels.items()})
    say("7", "ledgers and pin files written")

    # ── 8. measurement environments: launchers only ──────────────────────────
    for cell, info in cells.items():
        env = Path(info["env"])
        if env.exists():
            stop(f"environment {cell} already exists")
        r = run([py, "-m", "venv", *([] if info["bundled_pip"] else ["--without-pip"]), env])
        if r.returncode != 0:
            stop(f"could not create environment {cell}: {r.stderr[-300:]}")
        launcher = env / "Scripts" / "python.exe"
        probe = run([launcher, "-I", "-c",
                     "import sys,json,importlib.util as u;print(json.dumps({'version':sys.version.split()[0],"
                     "'executable':sys.executable,'base':sys._base_executable,"
                     "'importable':[m for m in ('pip','setuptools','wheel','fie','langdetect','requests') if u.find_spec(m)]}))"])
        facts = json.loads(probe.stdout)
        info.update({"launcher": str(launcher), "launcher_sha256": sha256(launcher), "python": facts["version"],
                     "base_executable": facts["base"], "importable_before_install": facts["importable"]})
        if info["bundled_pip"]:
            info["bundled_pip_version"] = run([launcher, "-I", "-m", "pip", "--version"]).stdout.split()[1]
            listing = run([launcher, "-I", "-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
            info["packages_before_install"] = json.loads(listing.stdout)
        if facts["version"] != "3.10.19" or "fie" in facts["importable"]:
            stop(f"environment {cell} is not a clean Python 3.10.19 environment")
    say("8", "environments: " + ", ".join(f"{c} ({i.get('bundled_pip_version', 'no pip')})" for c, i in cells.items()))

    # ── 9. positive connectivity control, through every launcher ─────────────
    targets = []
    for host in ("pypi.org", "files.pythonhosted.org"):
        addr = sorted({a[4][0] for a in socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)})[0]
        targets.append(f"{addr}:443")
    controls = {}
    launchers = {"base-interpreter": py, **{c: i["launcher"] for c, i in cells.items()}}
    for name, launcher in launchers.items():
        r = run([launcher, "-I", "-S", "-B", HERE / "egress_selftest.py", "--expect", "open",
                 *sum((["--target", t] for t in targets), [])])
        try:
            controls[name] = json.loads(r.stdout)
        except ValueError:
            stop(f"positive control produced no result for {name}: {r.stderr[-200:]}")
        if r.returncode != 0 or not controls[name]["ok"]:
            stop(f"positive connectivity control FAILED for {name}; without it a later denial proves nothing")
    provision.save(EVID / "isolation" / "positive_controls.json", {
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": targets,
        "target_note": "IPv4 addresses that pypi.org and files.pythonhosted.org resolved to at this time",
        "flags": "-I -S -B", "controls": controls})
    say("9", f"positive control passed through {len(controls)} executables; targets {targets}")

    # ── 10. inputs ───────────────────────────────────────────────────────────
    r = run([py, "-B", HERE / "export_inputs.py", "--out", SCRATCH / "inputs"])
    if r.returncode != 0:
        stop("input export failed: " + r.stderr[-300:])
    inputs = json.loads((SCRATCH / "inputs" / "inputs_summary.json").read_text(encoding="utf-8"))
    provision.save(EVID / "inputs_summary.json", inputs)
    verified = sorted({a["sha256"] for a in manifest["artifacts"]})
    (SCRATCH / "verified_model_sha256.json").write_text(json.dumps(verified), encoding="utf-8")

    plan = {"scratch": str(SCRATCH), "repo": str(REPO), "base_interpreter": py, "commit": full,
            "targets": targets, "cells": cells, "launchers": launchers,
            "inputs_pc": str(SCRATCH / "inputs" / "inputs_pc.jsonl"),
            "inputs_s": str(SCRATCH / "inputs" / "inputs_s.jsonl"),
            "verified_models": str(SCRATCH / "verified_model_sha256.json"),
            "manifest": str(REPO / "scripts" / "model_manifest.json"),
            "not_provisioned": {
                "A-pypi-1.18.0 [ml]": "not provisioned: awaiting explicit owner approval (O-22)",
                "A-clean [ml]": "not provisioned: awaiting explicit owner approval (O-21)",
                "A-release [ml]": "not provisioned: awaiting explicit owner approval (O-21)"},
            "wheel_inspection": {k: {"sha256": v["sha256"], "model_counts": v["model_counts"],
                                     "default_load_set_all_verified": v["loader"]["default_load_set_all_verified"],
                                     "default_classifier": v["loader"]["default_classifier"]}
                                 for k, v in inspected.items()}}
    (SCRATCH / "c1_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    provision.save(EVID / "c1_plan.json", plan)
    say("10", f"inputs exported ({inputs['inputs_pc']['count']} + {inputs['inputs_s']['count']}); plan written")
    print("PREPARED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
