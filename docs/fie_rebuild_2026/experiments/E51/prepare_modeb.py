"""
E51 Stage C1 - connected preparation of the three MODE B BASE cells. Run in the repository
environment, from the repository root, WHILE THE NETWORK IS UP:

    python docs/fie_rebuild_2026/experiments/E51/prepare_modeb.py

Creates three fresh, empty environments (no pip inside; mode B installs with the repository's
pip through --python) and takes their positive connectivity controls. Downloads no package,
builds nothing, installs nothing: it reuses the verified base wheelhouse, the published wheel
and the two local wheels built from `git archive 2f62ecd`, each only after re-hashing it
against its ledger. No [ml] package is involved. No other environment is touched.

Stops, changing nothing further, if: its directories already exist; fie/, pyproject.toml or the
model manifest differ from the commit the local wheels were built from; PyPI's latest release
is not 1.18.0; pip configuration, a proxy or a hosts redirection is found; the base set no
longer resolves to the approved pins; any artifact differs from its ledger; the mode B
wheelhouse holds a source archive; a wheel contains a model file the manifest does not verify
where that matters (A-release); less than 5 GB is free; or a positive control fails.
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

BUILD_COMMIT = "2f62ecd"
SUFFIX = "m1"
SCRATCH = Path(r"C:\Users\ASUS\fie-e51-scratch")
STAGE = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageC1"
EVID = STAGE / "modeb1"
APPROVED_PINS = REPO / "docs" / "fie_rebuild_2026" / "evidence" / "E51" / "stageB_remediation" / "pins" / "base.txt"
LOCAL_WHEEL = SCRATCH / "local-wheels" / "langdetect" / "langdetect-1.0.9-py3-none-any.whl"
LOCAL_WHEEL_SHA = "9acdfda7ee22ffdaaaf87f993cec0eb186d772fe0f3b93469832d83be6e9c8a4"
FIE_PYPI_SHA = "be504dc748822699df5dc5353ad0038617ef639a4c1840ed18786bc8579e963d"
BASE_REQUIREMENTS = ["requests>=2.28.0", "deep-translator>=1.11.4", "langdetect>=1.0.9"]
MIN_FREE_GB = 5
# cell -> (artifact, directory holding exactly that wheel, second build to compare with, approved pin file)
CELLS = {
    f"modeB-pypi-base-{SUFFIX}": ("A-pypi-1.18.0", "pypi", None, "pins-B-modeB-pypi-base.txt"),
    f"modeB-clean-base-{SUFFIX}": ("A-clean", "clean-build1", "clean-build2", "pins-B-modeB-clean-base.txt"),
    f"modeB-release-base-{SUFFIX}": ("A-release", "release-build1", "release-build2", "pins-B-modeB-release-base.txt"),
}


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


def only_wheel(directory: Path) -> Path:
    files = sorted(directory.iterdir())
    if len(files) != 1 or files[0].suffix != ".whl":
        stop(f"{directory.name} does not hold exactly one wheel")
    return files[0]


def main() -> int:
    modeb_plan = SCRATCH / "c1_modeb_plan.json"
    for existing in (EVID, modeb_plan, *[SCRATCH / "env" / c for c in CELLS],
                     *[SCRATCH / "pins" / f"pins-B-{c}.txt" for c in CELLS]):
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

    # ── 1. repository state; the local wheels must still represent the tree ──
    git = {"head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
           "branch": run(["git", "branch", "--show-current"]).stdout.strip(),
           "staged": run(["git", "diff", "--cached", "--name-only"]).stdout.split(),
           "status": run(["git", "status", "--short"]).stdout.splitlines(),
           "canonical": (REPO / "evals" / "baselines" / "CANONICAL").read_text(encoding="utf-8").strip(),
           "packaged_paths_changed_since_build_commit": run(
               ["git", "diff", "--name-only", BUILD_COMMIT, "--", "fie", "pyproject.toml",
                "scripts/model_manifest.json"]).stdout.split()}
    provision.save(EVID / "git_state.json", git)
    if git["staged"]:
        stop("there are staged changes")
    if git["packaged_paths_changed_since_build_commit"]:
        stop(f"fie/, pyproject.toml or the manifest differ from {BUILD_COMMIT}; the local wheels are stale")
    say("1", f"HEAD {git['head'][:7]} on {git['branch']}; packaged paths identical to {BUILD_COMMIT}")

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
    if sha256(LOCAL_WHEEL) != LOCAL_WHEEL_SHA:
        stop("the local langdetect wheel is not the one built and recorded in Stage B remediation")
    r = run([py, "-B", HERE / "provision.py", "resolve", "--name", "base", "--python",
             SCRATCH / "env" / "_provision" / "Scripts" / "python.exe", "--out", prov,
             "--local-wheel", LOCAL_WHEEL, *BASE_REQUIREMENTS])
    if r.returncode != 0:
        stop("base resolution failed: " + r.stdout[-300:])
    if (prov / "pins" / "base.txt").read_text(encoding="utf-8") != APPROVED_PINS.read_text(encoding="utf-8"):
        stop("the base resolution no longer equals the approved pins; pins are not refreshed")
    say("3", "base set re-resolved: identical to the approved pins (nothing downloaded)")

    # ── 4. the mode B wheelhouse and the three fie wheels, against ledgers ───
    ledger = json.loads((STAGE / "provenance" / "wheelhouse-B.ledger.json").read_text(encoding="utf-8"))
    wh_b = SCRATCH / "wheelhouse-B"
    on_disk = sorted(p.name for p in wh_b.iterdir())
    checks = [{"filename": f["filename"], "origin": f["origin"], "sha256": f["sha256"],
               "matches_ledger": (wh_b / f["filename"]).is_file() and sha256(wh_b / f["filename"]) == f["sha256"]}
              for f in ledger["files"]]
    problems = [c["filename"] for c in checks if not c["matches_ledger"]]
    if on_disk != sorted(f["filename"] for f in ledger["files"]):
        problems.append("the file set differs from the ledger")
    if any(not n.endswith(".whl") for n in on_disk):
        problems.append("a file that is not a wheel is present (mode B must hold no source archive)")
    if sha256(wh_b / LOCAL_WHEEL.name) != LOCAL_WHEEL_SHA:
        problems.append("the langdetect wheel in the wheelhouse is not the recorded local build")
    wheel_ledger = json.loads((STAGE / "provenance" / "fie_wheels.ledger.json").read_text(encoding="utf-8"))
    fie_wheels, wheel_checks = {}, {}
    for cell, (artifact, directory, second, _) in CELLS.items():
        wheel = only_wheel(SCRATCH / "wheels" / directory)
        fie_wheels[cell] = wheel
        entry = {"artifact": artifact, "filename": wheel.name, "sha256": sha256(wheel),
                 "origin": wheel_ledger[artifact]["origin"],
                 "matches_ledger": sha256(wheel) == wheel_ledger[artifact]["sha256"]}
        if second:
            entry["second_build_identical"] = sha256(only_wheel(SCRATCH / "wheels" / second)) == entry["sha256"]
            if not entry["second_build_identical"]:
                problems.append(f"{artifact}: the two local builds differ")
        if not entry["matches_ledger"]:
            problems.append(f"{artifact}: wheel differs from the ledger")
        wheel_checks[cell] = entry
    if wheel_checks[f"modeB-pypi-base-{SUFFIX}"]["sha256"] != FIE_PYPI_SHA:
        problems.append("the published wheel is not the approved digest")
    provision.save(EVID / "wheelhouse_revalidation.json", {
        "wheelhouse": "wheelhouse-B", "files": checks, "listing": on_disk, "file_count": len(on_disk),
        "contains_source_archive": any(not n.endswith(".whl") for n in on_disk),
        "langdetect_local_wheel_sha256": sha256(wh_b / LOCAL_WHEEL.name), "fie_wheels": wheel_checks,
        "downloaded_or_built_for_this_run": [], "problems": problems})
    if problems:
        stop(f"artifact revalidation failed: {problems}")
    say("4", f"wheelhouse-B: {len(on_disk)} wheels equal to the ledger, no source archive; three fie wheels equal to the ledger")

    # ── 5. static inspection, before anything is installed or imported ───────
    r = run([py, "-B", HERE / "inspect_wheel.py", "--manifest", REPO / "scripts" / "model_manifest.json",
             "--out", EVID / "artifacts" / "wheels.json",
             *[f"{CELLS[c][0]}={w}" for c, w in fie_wheels.items()]])
    if r.returncode != 0:
        stop("static wheel inspection failed: " + r.stderr[-300:])
    inspected = json.loads((EVID / "artifacts" / "wheels.json").read_text(encoding="utf-8"))
    release = inspected["A-release"]["model_counts"]
    if release["unmanifested"] or release["hash_mismatch"]:
        stop(f"A-release contains a model file the manifest does not verify: {release}")
    say("5", "inspected: " + "; ".join(f"{k} models {v['model_counts']}" for k, v in inspected.items()))

    # ── 6. pins: the approved mode B pins for each artifact, unchanged ───────
    pins = {}
    for cell, (_, _, _, approved) in CELLS.items():
        source = SCRATCH / "pins" / approved
        if source.read_text(encoding="utf-8") != (STAGE / "pins" / approved).read_text(encoding="utf-8"):
            stop(f"{approved} in the scratch directory differs from the one recorded as evidence")
        text = source.read_text(encoding="utf-8")
        if f"fie-sdk==1.18.0 --hash=sha256:{sha256(fie_wheels[cell])}" not in text \
                or f"langdetect==1.0.9 --hash=sha256:{LOCAL_WHEEL_SHA}" not in text:
            stop(f"{approved} does not pin this cell's wheel and the local langdetect wheel")
        pins[cell] = SCRATCH / "pins" / f"pins-B-{cell}.txt"
        shutil.copyfile(source, pins[cell])
        provision.save(EVID / "pins" / pins[cell].name, text)
    say("6", "pins: three files, each identical to its approved mode B pin file")

    # ── 7. disk ──────────────────────────────────────────────────────────────
    free_gb = shutil.disk_usage(SCRATCH).free / 1e9
    if free_gb < MIN_FREE_GB:
        stop(f"only {free_gb:.1f} GB free")

    # ── 8. environments: launchers only; no pip inside; nothing installed ────
    cells = {}
    for cell, (artifact, _, _, _) in CELLS.items():
        env = SCRATCH / "env" / cell
        r = run([py, "-m", "venv", "--without-pip", env])
        if r.returncode != 0:
            stop(f"could not create {cell}: {r.stderr[-300:]}")
        launcher = env / "Scripts" / "python.exe"
        facts = json.loads(run([launcher, "-I", "-c",
                                "import sys,json,importlib.util as u;print(json.dumps({'version':sys.version.split()[0],"
                                "'base':sys._base_executable,'importable':[m for m in ('pip','setuptools','wheel','fie',"
                                "'langdetect','requests','starlette') if u.find_spec(m)]}))"]).stdout)
        site = sorted(p.name for p in (env / "Lib" / "site-packages").iterdir())
        if facts["version"] != "3.10.19" or Path(facts["base"]).resolve() != Path(py).resolve() \
                or facts["importable"] or site:
            stop(f"{cell} is not an empty Python 3.10.19 venv of the base interpreter: {facts}, {site}")
        cells[cell] = {"artifact": artifact, "install_mode": "B-prebuilt-local-wheel", "bundled_pip": False,
                       "fie_wheel": str(fie_wheels[cell]), "fie_wheel_sha256": sha256(fie_wheels[cell]),
                       "pins": str(pins[cell]), "find_links": [str(wh_b), str(fie_wheels[cell].parent)],
                       "env": str(env), "launcher": str(launcher), "launcher_sha256": sha256(launcher),
                       "python": facts["version"], "base_executable": facts["base"],
                       "importable_before_install": facts["importable"], "site_packages_before_install": site,
                       "packages_before_install": [],
                       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    say("8", "environments: " + ", ".join(f"{c} ({i['launcher_sha256'][:12]}, empty)" for c, i in cells.items()))

    # ── 9. positive control: every launcher and the base interpreter ─────────
    targets = []
    for host in ("pypi.org", "files.pythonhosted.org"):
        addr = sorted({a[4][0] for a in socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)})[0]
        targets.append(f"{addr}:443")
    launchers = {"base-interpreter": py, **{c: i["launcher"] for c, i in cells.items()}}
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
    say("9", f"positive control passed through {len(controls)} executables; targets {targets}")

    # ── 10. the plan ─────────────────────────────────────────────────────────
    new_plan = {k: plan[k] for k in ("scratch", "repo", "base_interpreter", "inputs_pc", "inputs_s",
                                     "verified_models", "manifest", "not_provisioned")}
    new_plan.update({
        "scope": "mode B base cells only", "commit": git["head"], "wheels_built_from": BUILD_COMMIT,
        "targets": targets, "cells": cells, "launchers": launchers, "free_gb_at_preparation": round(free_gb, 1),
        "wheel_inspection": {k: {"sha256": v["sha256"], "model_counts": v["model_counts"],
                                 "default_load_set": v["loader"]["default_load_set"], "has": v["has"]}
                             for k, v in inspected.items()},
        "not_in_this_run": {"modeA-pypi-base-r1, -r3": "measured earlier; kept, not reused",
                            "modeB-*-base (created 2026-10-11)": "never installed; not used",
                            "[ml] cells": "not provisioned: awaiting explicit owner approval"}})
    for key in ("inputs_pc", "inputs_s", "verified_models", "manifest"):
        if not Path(new_plan[key]).is_file():
            stop(f"missing input: {key}")
    modeb_plan.write_text(json.dumps(new_plan, indent=1), encoding="utf-8")
    provision.save(EVID / "c1_modeb_plan.json", new_plan)
    say("10", "mode B plan written")
    print("PREPARED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
