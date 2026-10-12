"""
E51 Stage C1 - connected preparation of ONE fresh mode A environment for the retry after
offline attempt 4. Run in the repository environment, from the repository root, WHILE THE
NETWORK IS UP:

    python docs/fie_rebuild_2026/experiments/E51/prepare_retry.py

Creates exactly one environment, `env/modeA-pypi-base-r3`, empty, and takes its positive
connectivity control. Downloads no package and installs nothing into it: the install happens
offline, in offline_runner.py. Reuses the mode A wheelhouse only after re-hashing every file
against its ledger. Touches no other environment and deletes nothing.

Stops, changing nothing further, if: the retry directories already exist; PyPI's latest
release is not 1.18.0; pip configuration, a proxy or a hosts redirection is found; the base
set no longer resolves to the approved pins; any wheelhouse file differs from its ledger; the
wheelhouse holds a langdetect wheel; less than 5 GB is free; or a positive control fails.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path.cwd()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import provision  # noqa: E402

CELL = "modeA-pypi-base-r3"
PINS_FROM = "pins-A-modeA-pypi-base-r1.txt"            # identical content; the approved mode A pins
SCRATCH = Path(r"C:\Users\ASUS\fie-e51-scratch")
STAGE = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageC1"
EVID = STAGE / "retry1"
APPROVED_PINS = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageB_remediation" / "pins" / "base.txt"
LOCAL_WHEEL = SCRATCH / "local-wheels" / "langdetect" / "langdetect-1.0.9-py3-none-any.whl"
FIE_PYPI_SHA = "be504dc748822699df5dc5353ad0038617ef639a4c1840ed18786bc8579e963d"
SDIST_SHA = "cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0"
BASE_REQUIREMENTS = ["requests>=2.28.0", "deep-translator>=1.11.4", "langdetect>=1.0.9"]
MIN_FREE_GB = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(message: str) -> None:
    print("STOP:", message)
    if EVID.is_dir():
        provision.save(EVID / "STOPPED.json", {"reason": message})
    sys.exit(2)


def say(step: str, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", **kwargs)


def main() -> int:
    env = SCRATCH / "env" / CELL
    retry_plan = SCRATCH / "c1_retry_plan.json"
    for existing in (EVID, env, retry_plan, SCRATCH / "pins" / f"pins-A-{CELL}.txt"):
        if existing.exists():
            print(f"STOP: {existing.name} already exists; nothing is overwritten")
            return 2
    plan = json.loads((SCRATCH / "c1_plan.json").read_text(encoding="utf-8"))
    py = sys.executable
    if py != plan["base_interpreter"]:
        print("STOP: this is not the interpreter the plan was made with")
        return 2
    EVID.mkdir(parents=True)
    prov = EVID / "provisioning"

    # ── 1. repository state (recorded, not changed) ──────────────────────────
    git = {"head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
           "branch": run(["git", "branch", "--show-current"]).stdout.strip(),
           "staged": run(["git", "diff", "--cached", "--name-only"]).stdout.split(),
           "status": run(["git", "status", "--short"]).stdout.splitlines(),
           "canonical": (REPO / "evals" / "baselines" / "CANONICAL").read_text(encoding="utf-8").strip()}
    provision.save(EVID / "git_state.json", git)
    if git["staged"]:
        stop("there are staged changes")
    say("1", f"HEAD {git['head'][:7]} on {git['branch']}; CANONICAL {git['canonical']}")

    # ── 2. G0: release, pip configuration, proxy, hosts ──────────────────────
    r = run([py, "-B", HERE / "provision.py", "preflight", "--out", prov])
    if r.returncode != 0:
        stop("provisioning preflight failed (pip configuration, proxy or hosts): " + r.stdout[-300:])
    r = run([py, "-B", HERE / "provision.py", "index", "fie-sdk", "--expect", "1.18.0", "--out", prov])
    if r.returncode != 0:
        stop("the latest PyPI release of fie-sdk is no longer 1.18.0, or its record could not be read")
    meta = provision.fetch_json(f"{provision.JSON_API}/fie-sdk/1.18.0/json")
    published = next((f for f in meta["urls"] if f["filename"] == "fie_sdk-1.18.0-py3-none-any.whl"), None)
    if not published or published["digests"]["sha256"] != FIE_PYPI_SHA or published.get("yanked"):
        stop("PyPI no longer publishes the approved digest for the fie-sdk 1.18.0 wheel")
    say("2", "PyPI latest is 1.18.0 with the approved wheel digest; no pip configuration, proxy or hosts redirection")

    # ── 3. G2: the base set must still resolve to the approved pins ──────────
    r = run([py, "-B", HERE / "provision.py", "resolve", "--name", "base", "--python",
             SCRATCH / "env" / "_provision" / "Scripts" / "python.exe", "--out", prov,
             "--local-wheel", LOCAL_WHEEL, *BASE_REQUIREMENTS])
    if r.returncode != 0:
        stop("base resolution failed: " + r.stdout[-300:])
    if (prov / "pins" / "base.txt").read_text(encoding="utf-8") != APPROVED_PINS.read_text(encoding="utf-8"):
        stop("the base resolution no longer equals the approved pins; pins are not refreshed")
    say("3", "base set re-resolved: identical to the approved pins (nothing downloaded)")

    # ── 4. the mode A wheelhouse, against its ledger ─────────────────────────
    ledger = json.loads((STAGE / "provenance" / "wheelhouse-A.ledger.json").read_text(encoding="utf-8"))
    wh_a = SCRATCH / "wheelhouse-A"
    on_disk = sorted(p.name for p in wh_a.iterdir())
    checks = [{"filename": f["filename"], "origin": f["origin"], "sha256": f["sha256"],
               "matches_ledger": (wh_a / f["filename"]).is_file() and sha256(wh_a / f["filename"]) == f["sha256"]}
              for f in ledger["files"]]
    problems = [c["filename"] for c in checks if not c["matches_ledger"]]
    if on_disk != sorted(f["filename"] for f in ledger["files"]):
        problems.append("the file set differs from the ledger")
    if any(n.startswith("langdetect") and n.endswith(".whl") for n in on_disk):
        problems.append("a langdetect wheel is present")
    sdist = wh_a / "langdetect-1.0.9.tar.gz"
    if not sdist.is_file() or sha256(sdist) != SDIST_SHA:
        problems.append("the langdetect source archive is missing or has the wrong digest")
    pypi_dir = SCRATCH / "wheels" / "pypi"
    fie_wheel = pypi_dir / "fie_sdk-1.18.0-py3-none-any.whl"
    if sorted(p.name for p in pypi_dir.iterdir()) != [fie_wheel.name] or sha256(fie_wheel) != FIE_PYPI_SHA:
        problems.append("the published fie-sdk wheel directory is not exactly the approved wheel")
    provision.save(EVID / "wheelhouse_revalidation.json", {
        "wheelhouse": "wheelhouse-A", "files": checks, "listing": on_disk, "file_count": len(on_disk),
        "contains_langdetect_wheel": any(n.startswith("langdetect") and n.endswith(".whl") for n in on_disk),
        "langdetect_source_archive_sha256": sha256(sdist) if sdist.is_file() else None,
        "fie_wheel_sha256": sha256(fie_wheel), "fie_wheel_matches_pypi_digest": sha256(fie_wheel) == FIE_PYPI_SHA,
        "downloaded_for_this_retry": [], "problems": problems})
    if problems:
        stop(f"wheelhouse revalidation failed: {problems}")
    say("4", f"wheelhouse-A: {len(on_disk)} files equal to the ledger; source archive present; no langdetect wheel")

    # ── 5. pins: the approved mode A pins, unchanged ─────────────────────────
    source_pins = SCRATCH / "pins" / PINS_FROM
    if source_pins.read_text(encoding="utf-8") != (STAGE / "pins" / PINS_FROM).read_text(encoding="utf-8"):
        stop("the mode A pin file in the scratch directory differs from the one recorded as evidence")
    pins = SCRATCH / "pins" / f"pins-A-{CELL}.txt"
    shutil.copyfile(source_pins, pins)
    provision.save(EVID / "pins" / pins.name, pins.read_text(encoding="utf-8"))
    say("5", f"pins: {sum(1 for ln in pins.read_text().splitlines() if ln.strip())} hashed lines, identical to the approved mode A pins")

    # ── 6. disk ──────────────────────────────────────────────────────────────
    free_gb = shutil.disk_usage(SCRATCH).free / 1e9
    if free_gb < MIN_FREE_GB:
        stop(f"only {free_gb:.1f} GB free")

    # ── 7. the environment: a launcher and its bundled pip; nothing installed ─
    r = run([py, "-m", "venv", env])
    if r.returncode != 0:
        stop("could not create the environment: " + r.stderr[-300:])
    launcher = env / "Scripts" / "python.exe"
    facts = json.loads(run([launcher, "-I", "-c",
                            "import sys,json,importlib.util as u;print(json.dumps({'version':sys.version.split()[0],"
                            "'base':sys._base_executable,'importable':[m for m in ('pip','setuptools','wheel','fie',"
                            "'langdetect','requests') if u.find_spec(m)]}))"]).stdout)
    listing = run([launcher, "-I", "-X", "utf8", "-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
    if listing.returncode != 0:
        stop("pip list failed in the new environment: " + listing.stderr[-300:])
    before = json.loads(listing.stdout)
    pip_version = next(p["version"] for p in before if p["name"] == "pip")
    same_base = Path(facts["base"]).resolve() == Path(py).resolve()
    if facts["version"] != "3.10.19" or not same_base or set(facts["importable"]) != {"pip", "setuptools"}:
        stop(f"the new environment is not a clean Python 3.10.19 venv of the base interpreter: {facts}")
    cell = {"artifact": "A-pypi-1.18.0", "install_mode": "A-source-build", "bundled_pip": True,
            "fie_wheel": str(fie_wheel), "fie_wheel_sha256": FIE_PYPI_SHA, "pins": str(pins),
            "find_links": [str(wh_a), str(pypi_dir)], "env": str(env), "launcher": str(launcher),
            "launcher_sha256": sha256(launcher), "python": facts["version"], "base_executable": facts["base"],
            "importable_before_install": facts["importable"], "bundled_pip_version": pip_version,
            "packages_before_install": before, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    say("7", f"{CELL}: Python {facts['version']}, launcher {cell['launcher_sha256'][:16]}, before install {before}")

    # ── 8. positive control: this launcher and the base interpreter ──────────
    targets = []
    for host in ("pypi.org", "files.pythonhosted.org"):
        addr = sorted({a[4][0] for a in socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)})[0]
        targets.append(f"{addr}:443")
    launchers = {"base-interpreter": py, CELL: str(launcher)}
    controls = {}
    for name, exe in launchers.items():
        r = run([exe, "-I", "-S", "-B", HERE / "egress_selftest.py", "--expect", "open",
                 *sum((["--target", t] for t in targets), [])])
        try:
            controls[name] = json.loads(r.stdout)
        except ValueError:
            stop(f"the positive control produced no result for {name}")
        if r.returncode != 0 or not controls[name]["ok"]:
            stop(f"the positive connectivity control FAILED for {name}")
    provision.save(EVID / "isolation" / "positive_controls.json", {
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": targets,
        "target_note": "IPv4 addresses that pypi.org and files.pythonhosted.org resolved to at this time",
        "flags": "-I -S -B", "controls": controls})
    say("8", f"positive control passed through {len(controls)} executables; targets {targets}")

    # ── 9. the retry plan ────────────────────────────────────────────────────
    new_plan = {k: plan[k] for k in ("scratch", "repo", "base_interpreter", "commit", "inputs_pc", "inputs_s",
                                     "verified_models", "manifest", "not_provisioned")}
    new_plan.update({
        "retry_of": "offline_attempt4 (modeA-pypi-base-r1: verification tool failed)", "commit": git["head"],
        "targets": targets, "cells": {CELL: cell}, "launchers": launchers, "free_gb_at_preparation": round(free_gb, 1),
        "not_in_this_retry": {"modeA-pypi-base-r1": "installed in attempt 4; kept, not reused",
                              "modeA-pypi-base-r2": "not used", "mode B base cells": "not authorized for this retry"}})
    for key in ("inputs_pc", "inputs_s", "verified_models", "manifest"):
        if not Path(new_plan[key]).is_file():
            stop(f"missing input: {key}")
    retry_plan.write_text(json.dumps(new_plan, indent=1), encoding="utf-8")
    provision.save(EVID / "c1_retry_plan.json", new_plan)
    say("9", "retry plan written")
    print("PREPARED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
