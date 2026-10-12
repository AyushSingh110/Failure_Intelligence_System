"""
Release check: install a BUILT WHEEL into a fresh environment and test what was installed.

    python scripts/release_check.py --wheel <path/to/fie_sdk-X.Y.Z-py3-none-any.whl> --out <evidence dir>
        [--wheelhouse <dir>]   install dependencies from this directory only (--no-index);
                               without it pip uses its configured index
        [--workdir <dir>]      where the temporary environment is created (default: the system
                               temp directory); must be outside the repository
        [--keep]               do not remove the temporary environment afterwards
    python scripts/release_check.py --selftest

It never imports the working tree. Every check runs in the new environment, through that
environment's interpreter with `-I`, from an empty working directory outside the repository,
and the installed copy is proven by `fie.__file__` lying inside that environment's
site-packages.

Scope (PLAN_021 section 5.2). Implemented here: Level 1, the package installs (AC-1 to AC-4),
and Level 2, the SDK imports (AC-5 to AC-8). AC-9 is a repository test and is not run here.
Levels 3 to 5 are NOT implemented. A pass says that the wheel installs and that `import fie`
and the command line start. It says nothing about whether detection works.

This tool does not repair anything. It installs the wheel with no extra, adds no package to
make a check pass, downloads no model and contacts no service of its own. A failed import is
reported as a failed criterion and the exit code is non-zero.

Exit codes:  0  every implemented criterion passed
             1  at least one criterion FAILED (the report names it)
             2  the check itself could not run (usage, missing file, tool failure)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCOPE = "PLAN_021 section 5.2: Level 1 (AC-1 to AC-4) and Level 2 (AC-5 to AC-8)"
NOT_IMPLEMENTED = ["AC-9 (repository test)", "Level 3: AC-10 to AC-12", "Level 4: AC-13 to AC-18",
                   "Level 5: AC-19, AC-20", "release hygiene: AC-21 to AC-24"]
# Set for every command that runs the installed package. Recorded in the report.
FIE_CONTROLS = {"FIE_NO_TELEMETRY": "1", "FIE_NO_AUTO_DOWNLOAD": "1"}
# AC-8: none of these may be imported by a bare `import fie`.
OPTIONAL_MODULES = ["starlette", "fastapi", "openai", "anthropic", "joblib", "sklearn", "xgboost", "onnxruntime"]
PASS, FAIL, NOT_EVALUATED, TOOL_FAILED = "PASS", "FAIL", "NOT_EVALUATED", "TOOL_FAILED"

IMPORT_SNIPPET = r"""
import json, os, sys
out = {"cwd": os.getcwd(), "prefix": sys.prefix, "isolated": bool(sys.flags.isolated)}
try:
    import fie
except BaseException as exc:
    out.update({"import_ok": False, "exception": type(exc).__name__,
                "missing_module": getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None,
                "message": str(exc)})
    print("RELEASE_CHECK_JSON " + json.dumps(out))
    raise
site = os.path.normcase(os.path.join(sys.prefix, "Lib", "site-packages")) if os.name == "nt" else None
fie_file = os.path.normcase(os.path.abspath(fie.__file__))
out.update({"import_ok": True, "fie_file": fie.__file__, "fie_version": getattr(fie, "__version__", None),
            "fie_under_prefix": fie_file.startswith(os.path.normcase(os.path.abspath(sys.prefix))),
            "fie_in_site_packages": "site-packages" in fie_file,
            "optional_modules_imported": sorted(m for m in %(optional)r if m in sys.modules),
            "all_names": list(getattr(fie, "__all__", [])), "unresolved_names": []})
for name in out["all_names"]:
    try:
        getattr(fie, name)
    except BaseException as exc:
        out["unresolved_names"].append([name, type(exc).__name__])
print("RELEASE_CHECK_JSON " + json.dumps(out))
"""


# ── pure functions (covered by --selftest) ──────────────────────────────────

def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    return norm(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement).group(1))


def requirement_kind(requirement: str) -> str:
    """'extra' if only wanted for an extra, 'conditional' for any other marker, else 'required'."""
    marker = requirement.partition(";")[2]
    if not marker.strip():
        return "required"
    return "extra" if re.search(r"\bextra\s*==", marker) else "conditional"


def classify_installed_set(exit_code: int, stdout: str, root: str, preexisting: dict) -> dict:
    """
    AC-3 from one `pip inspect` run. A tool that failed, or whose output cannot be used, has
    said nothing about what is installed: that is TOOL_FAILED with the lists not evaluated,
    never a package finding (the E51 attempt 4 mistake).
    """
    blank = {"missing": NOT_EVALUATED, "undeclared": NOT_EVALUATED, "distributions": NOT_EVALUATED}
    if exit_code != 0:
        return {"status": TOOL_FAILED, "detail": f"pip inspect exited {exit_code}", **blank}
    try:
        entries = json.loads(stdout)["installed"]
        installed = {norm(d["metadata"]["name"]): d["metadata"] for d in entries}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": TOOL_FAILED, "detail": f"pip inspect output unusable: {type(exc).__name__}", **blank}
    root = norm(root)
    if root not in installed:
        return {"status": FAIL, "detail": f"{root} is not installed", "missing": [root], "undeclared": [],
                "distributions": {k: v.get("version") for k, v in sorted(installed.items())}}
    required, allowed, queue, missing = {root}, {root}, [root], []
    while queue:
        for req in installed[queue.pop()].get("requires_dist") or []:
            name, kind = requirement_name(req), requirement_kind(req)
            if kind == "extra":
                continue
            allowed.add(name)
            if kind == "required":
                if name not in installed:
                    missing.append(name)
                elif name not in required:
                    required.add(name)
                    queue.append(name)
            elif name in installed and name not in required:      # conditional and present: follow it
                required.add(name)
                queue.append(name)
    undeclared = sorted(n for n in installed if n not in allowed and n not in {norm(k) for k in preexisting})
    missing = sorted(set(missing))
    return {"status": PASS if not missing and not undeclared else FAIL,
            "detail": "installed set equals the declared base closure" if not missing and not undeclared
                      else "installed set differs from the declared base closure",
            "missing": missing, "undeclared": undeclared,
            "distributions": {k: v.get("version") for k, v in sorted(installed.items())}}


def classify_import(exit_code: int, payload: dict | None) -> dict:
    """AC-5. PASS needs a completed import of the installed copy. Nothing else passes."""
    if payload is None:
        return {"status": TOOL_FAILED, "detail": f"the import check produced no result (exit {exit_code})"}
    if payload.get("import_ok") is not True:
        missing = payload.get("missing_module")
        return {"status": FAIL, "exception": payload.get("exception"), "missing_module": missing,
                "message": payload.get("message"),
                "detail": f"import fie raised {payload.get('exception')}"
                          + (f": missing module {missing!r}" if missing else "")}
    if exit_code != 0:
        return {"status": TOOL_FAILED, "detail": f"import reported success but the process exited {exit_code}"}
    if not (payload.get("fie_under_prefix") and payload.get("fie_in_site_packages")):
        return {"status": FAIL, "detail": "fie was not imported from the environment's site-packages"}
    return {"status": PASS, "detail": "import fie succeeded from the environment's site-packages",
            "fie_version": payload.get("fie_version")}


def overall(criteria: dict) -> tuple:
    """(exit code, verdict). Zero only when every implemented criterion passed."""
    statuses = {k: v["status"] for k, v in criteria.items()}
    failed = sorted(k for k, s in statuses.items() if s == FAIL)
    tool = sorted(k for k, s in statuses.items() if s == TOOL_FAILED)
    blocked = sorted(k for k, s in statuses.items() if s == NOT_EVALUATED)
    if failed:
        return 1, "FAILED: " + ", ".join(failed) + (f"; not evaluated: {', '.join(blocked)}" if blocked else "")
    if tool:
        return 2, "CHECK COULD NOT RUN: " + ", ".join(tool)
    if blocked or not statuses:
        return 2, "INCOMPLETE: not evaluated: " + ", ".join(blocked)
    return 0, "PASSED: every implemented criterion (Levels 1 and 2 only)"


# ── running ─────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Recorder:
    """Writes each command's stdout and stderr to their own files, once, with paths redacted."""

    def __init__(self, out: Path, replace: dict):
        self.out, self.replace, self.commands = out, replace, []

    def redact(self, text: str) -> str:
        for real, tag in self.replace.items():
            for form in (real, real.replace("\\", "/"), real.replace("\\", "\\\\")):
                text = re.sub(re.escape(form), lambda m: tag, text, flags=re.I)
        return text

    def save(self, rel: str, obj) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = obj if isinstance(obj, str) else json.dumps(obj, indent=1, sort_keys=True)
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write(self.redact(text))

    def run(self, label: str, cmd: list, env: dict, cwd: Path, timeout: int = 900) -> tuple:
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.perf_counter()
        try:
            p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env=env, cwd=str(cwd), timeout=timeout)
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            code, out, err = -9, "", f"TIMEOUT after {timeout}s"
        except OSError as exc:
            code, out, err = -8, "", f"EXECUTION ERROR: {type(exc).__name__}: {exc}"
        self.save(f"commands/{label}.stdout.txt", out)
        self.save(f"commands/{label}.stderr.txt", err)
        self.commands.append({"label": label, "command": " ".join(str(c) for c in cmd), "exit_code": code,
                              "seconds": round(time.perf_counter() - t0, 2), "started_utc": started,
                              "cwd": str(cwd), "environment_variables": sorted(env),
                              "fie_controls": {k: env[k] for k in FIE_CONTROLS if k in env}})
        return code, out, err


def child_env(home: Path, extra: dict | None = None) -> dict:
    """An explicit environment: nothing inherited that could redirect pip or configure fie."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "tmp").mkdir(exist_ok=True)
    env = {"HOME": str(home), "USERPROFILE": str(home), "TEMP": str(home / "tmp"), "TMP": str(home / "tmp"),
           "PIP_CONFIG_FILE": os.devnull, "PYTHONNOUSERSITE": "1"}
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        base = Path(sys.base_prefix)
        env.update({"SystemRoot": root, "windir": root, "SystemDrive": os.environ.get("SystemDrive", "C:"),
                    "COMSPEC": os.path.join(root, "System32", "cmd.exe"), "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                    "APPDATA": str(home / "AppData" / "Roaming"), "LOCALAPPDATA": str(home / "AppData" / "Local"),
                    "PATH": ";".join([str(base), str(base / "Library" / "bin"), os.path.join(root, "System32"), root])})
    else:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    env.update(extra or {})
    return env


def parse_payload(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith("RELEASE_CHECK_JSON "):
            try:
                return json.loads(line[len("RELEASE_CHECK_JSON "):])
            except ValueError:
                return None
    return None


def check_wheel(wheel: Path, out: Path, wheelhouse: Path | None, workdir: Path | None, keep: bool) -> int:
    if not wheel.is_file() or wheel.suffix != ".whl":
        print(f"release_check: not a wheel file: {wheel}")
        return 2
    if out.exists() and any(out.iterdir()):
        print(f"release_check: the output directory is not empty: {out}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fie-release-check-", dir=str(workdir) if workdir else None)).resolve()
    if REPO in work.parents or work == REPO:
        print("release_check: the work directory must be outside the repository")
        return 2
    rec = Recorder(out, {str(work): "<WORK>", str(Path(sys.base_prefix)): "<BASE_PYTHON>",
                         str(wheel.resolve().parent): "<WHEEL_DIR>",
                         **({str(wheelhouse.resolve()): "<WHEELHOUSE>"} if wheelhouse else {}),
                         str(Path.home()): "<HOME>", os.environ.get("USERNAME") or "\0": "<USER>"})
    env_dir, home, empty = work / "env", work / "home", work / "empty"
    empty.mkdir()
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = env_dir / ("Lib/site-packages" if os.name == "nt" else
                      f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode("utf-8", "replace")
    dist_name = re.search(r"^Name: (.+)$", meta, re.M).group(1).strip()
    dist_version = re.search(r"^Version: (.+)$", meta, re.M).group(1).strip()
    report = {"tool": "scripts/release_check.py", "scope": SCOPE, "not_implemented": NOT_IMPLEMENTED,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "wheel": {"filename": wheel.name, "sha256": sha256(wheel), "size": wheel.stat().st_size,
                        "name": dist_name, "version": dist_version, "files": len(names),
                        "requires_dist_base": sorted(r for r in re.findall(r"^Requires-Dist: (.+)$", meta, re.M)
                                                     if requirement_kind(r) != "extra")},
              "target": "T-base: the wheel with no extra",
              "dependency_source": "local wheelhouse only (--no-index)" if wheelhouse else "pip's configured index",
              "fie_controls": FIE_CONTROLS, "host_python": sys.version.split()[0],
              "working_directory_for_checks": "an empty directory outside the repository",
              "repository_on_path": False}
    criteria: dict = {}
    pip_env = child_env(home / "pip")
    fie_env = child_env(home / "fie", FIE_CONTROLS)

    # ── environment ──────────────────────────────────────────────────────────
    code, _, err = rec.run("00_create_venv", [sys.executable, "-m", "venv", env_dir], pip_env, empty)
    if code != 0 or not py.is_file():
        report["criteria"] = {"environment": {"status": TOOL_FAILED, "detail": "could not create the environment"}}
        return finish(rec, report, work, keep)
    code, pre_out, _ = rec.run("01_preinstall_list", [py, "-I", "-X", "utf8", "-m", "pip", "list", "--format=json",
                                                      "--disable-pip-version-check"], pip_env, empty)
    try:
        preexisting = {p["name"]: p["version"] for p in json.loads(pre_out)} if code == 0 else {}
    except ValueError:
        preexisting = {}
    code, ver_out, _ = rec.run("02_interpreter", [py, "-I", "-c", "import sys;print(sys.version.split()[0]);print(sys.prefix)"],
                               pip_env, empty)
    report["environment"] = {"python": (ver_out.split() or [None])[0], "preinstalled": preexisting,
                             "launcher_sha256": sha256(py)}

    # ── Level 1 ──────────────────────────────────────────────────────────────
    install = [py, "-I", "-m", "pip", "install", "--isolated", "--disable-pip-version-check", "--no-cache-dir"]
    if wheelhouse:
        install += ["--no-index", "--find-links", wheelhouse.resolve()]
    code, _, err = rec.run("10_AC1_install", [*install, wheel.resolve()], pip_env, empty, timeout=1800)
    criteria["AC-1 wheel installs"] = {"status": PASS if code == 0 else FAIL, "exit_code": code,
                                       "detail": "pip install exited 0" if code == 0
                                       else "pip install failed: " + (err.strip().splitlines() or ["no output"])[-1][:300]}
    installed_ok = code == 0
    if installed_ok:
        code, c_out, c_err = rec.run("11_AC2_pip_check", [py, "-I", "-m", "pip", "check", "--disable-pip-version-check"],
                                     pip_env, empty)
        criteria["AC-2 pip check"] = {"status": {0: PASS, 1: FAIL}.get(code, TOOL_FAILED), "exit_code": code,
                                      "detail": (c_out + c_err).strip()[:300]}
        code, i_out, _ = rec.run("12_AC3_pip_inspect", [py, "-I", "-X", "utf8", "-m", "pip", "inspect",
                                                        "--disable-pip-version-check"], pip_env, empty)
        criteria["AC-3 installed set equals declared"] = {"exit_code": code,
                                                          **classify_installed_set(code, i_out, dist_name, preexisting)}
        differing, checked = [], 0
        with zipfile.ZipFile(wheel) as z:
            for name in names:
                if name.endswith("/") or ".dist-info/" in name:
                    continue
                checked += 1
                target = site / name
                if not target.is_file() or hashlib.sha256(z.read(name)).hexdigest() != sha256(target):
                    differing.append(name)
        criteria["AC-4 installed files equal the wheel"] = {
            "status": PASS if checked and not differing else FAIL, "files_checked": checked,
            "files_differing": differing[:20], "detail": f"{checked} files compared, {len(differing)} differ"}
    else:
        for key in ("AC-2 pip check", "AC-3 installed set equals declared", "AC-4 installed files equal the wheel"):
            criteria[key] = {"status": NOT_EVALUATED, "detail": "the wheel did not install"}

    # ── Level 2 ──────────────────────────────────────────────────────────────
    level2 = ("AC-5 import fie", "AC-6 every name in fie.__all__ resolves", "AC-7 command line starts",
              "AC-8 no optional package imported by import fie")
    if not installed_ok:
        for key in level2:
            criteria[key] = {"status": NOT_EVALUATED, "detail": "the wheel did not install"}
    else:
        code, out_text, _ = rec.run("20_AC5_import", [py, "-I", "-c", IMPORT_SNIPPET % {"optional": OPTIONAL_MODULES}],
                                    fie_env, empty)
        payload = parse_payload(out_text)
        criteria["AC-5 import fie"] = {"exit_code": code, **classify_import(code, payload)}
        report["import"] = {k: payload.get(k) for k in ("import_ok", "exception", "missing_module", "message",
                                                        "fie_version", "isolated", "fie_under_prefix",
                                                        "fie_in_site_packages")} if payload else None
        if criteria["AC-5 import fie"]["status"] == PASS:
            bad = payload.get("unresolved_names") or []
            criteria[level2[1]] = {"status": PASS if payload.get("all_names") and not bad else FAIL,
                                   "names": len(payload.get("all_names") or []), "unresolved": bad,
                                   "detail": f"{len(payload.get('all_names') or [])} names, {len(bad)} unresolved"}
            loaded = payload.get("optional_modules_imported") or []
            criteria[level2[3]] = {"status": PASS if not loaded else FAIL, "imported": loaded,
                                   "detail": "none imported" if not loaded else f"imported: {loaded}"}
        else:
            why = "import fie did not complete, so this could not be evaluated"
            criteria[level2[1]] = {"status": NOT_EVALUATED, "detail": why}
            criteria[level2[3]] = {"status": NOT_EVALUATED, "detail": why}
        # The command line is a separate entry point and is run whatever AC-5 said.
        h_code, h_out, h_err = rec.run("21_AC7_cli_help", [py, "-I", "-m", "fie", "--help"], fie_env, empty)
        d_code, d_out, d_err = rec.run("22_AC7_cli_detect", [py, "-I", "-m", "fie", "detect",
                                                             "What is the capital of France?", "--output", "json"],
                                       fie_env, empty)
        try:
            detect_json = isinstance(json.loads(d_out), dict)
        except ValueError:
            detect_json = False
        cli_ok = h_code == 0 and d_code == 0 and detect_json
        last = (h_err if h_code != 0 else d_err).strip().splitlines()
        criteria[level2[2]] = {"status": PASS if cli_ok else FAIL, "help_exit_code": h_code,
                               "detect_exit_code": d_code, "detect_stdout_is_json": detect_json,
                               "detail": "help and detect exited 0 with JSON output" if cli_ok
                               else "command line failed: " + (last[-1][:300] if last else "no error text")}
    report["criteria"] = criteria
    return finish(rec, report, work, keep)


def finish(rec: Recorder, report: dict, work: Path, keep: bool) -> int:
    code, verdict = overall(report.get("criteria", {}))
    removed = False
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
        removed = not work.exists()
    report.update({"commands": rec.commands, "exit_code": code, "verdict": verdict,
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "cleanup": {"temporary_environment_removed": removed, "kept_on_request": keep}})
    rec.save("report.json", report)
    sums = [f"{sha256(p)}  {p.relative_to(rec.out).as_posix()}" for p in sorted(rec.out.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS.txt"]
    (rec.out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(f"release_check: {report.get('wheel', {}).get('filename')}  sha256 {report.get('wheel', {}).get('sha256')}")
    for name, c in report.get("criteria", {}).items():
        print(f"  [{c['status']:<13}] {name}: {c.get('detail', '')}")
    print(f"release_check: {verdict}")
    print(f"release_check: exit {code}; report in {rec.redact(str(rec.out))}")
    return code


def selftest() -> int:
    def insp(packages: dict, requires: dict | None = None) -> str:
        return json.dumps({"installed": [{"metadata": {"name": n, "version": v,
                                                       "requires_dist": (requires or {}).get(n, [])}}
                                         for n, v in packages.items()]})
    req = {"fie-sdk": ["requests>=2.28.0", "langdetect>=1.0.9", "fastapi>=0.100; extra == 'server'"],
           "requests": ["idna<4,>=2.5", "PySocks!=1.5.7; extra == 'socks'", "colorama; platform_system == 'Windows'"]}
    full = {"fie-sdk": "1", "requests": "2", "langdetect": "1", "idna": "3", "pip": "23", "setuptools": "79"}
    pre = {"pip": "23", "setuptools": "79"}
    checks = [
        ("tool failure is not a package finding",
         classify_installed_set(2, "", "fie-sdk", pre)["status"] == TOOL_FAILED
         and classify_installed_set(2, "", "fie-sdk", pre)["missing"] == NOT_EVALUATED),
        ("unusable output is a tool failure", classify_installed_set(0, "{", "fie-sdk", pre)["status"] == TOOL_FAILED),
        ("declared closure passes", classify_installed_set(0, insp(full, req), "fie_sdk", pre)["status"] == PASS),
        ("extras are not required", "fastapi" not in classify_installed_set(0, insp(full, req), "fie-sdk", pre)["missing"]),
        ("a missing dependency fails", classify_installed_set(
            0, insp({k: v for k, v in full.items() if k != "idna"}, req), "fie-sdk", pre)["missing"] == ["idna"]),
        ("an undeclared distribution fails", classify_installed_set(
            0, insp({**full, "starlette": "1"}, req), "fie-sdk", pre)["undeclared"] == ["starlette"]),
        ("a conditional dependency is allowed, not required", classify_installed_set(
            0, insp({**full, "colorama": "1"}, req), "fie-sdk", pre)["status"] == PASS),
        ("a failed import fails AC-5 and names the module", classify_import(1, {
            "import_ok": False, "exception": "ModuleNotFoundError", "missing_module": "starlette",
            "message": "No module named 'starlette'"}) == {
            "status": FAIL, "exception": "ModuleNotFoundError", "missing_module": "starlette",
            "message": "No module named 'starlette'",
            "detail": "import fie raised ModuleNotFoundError: missing module 'starlette'"}),
        ("no result is a tool failure, not a pass", classify_import(1, None)["status"] == TOOL_FAILED),
        ("an import from outside site-packages fails", classify_import(0, {
            "import_ok": True, "fie_under_prefix": False, "fie_in_site_packages": False})["status"] == FAIL),
        ("a completed installed import passes", classify_import(0, {
            "import_ok": True, "fie_under_prefix": True, "fie_in_site_packages": True})["status"] == PASS),
        ("any failure gives exit 1", overall({"a": {"status": PASS}, "b": {"status": FAIL},
                                              "c": {"status": NOT_EVALUATED}})[0] == 1),
        ("not evaluated is never a pass", overall({"a": {"status": PASS}, "b": {"status": NOT_EVALUATED}})[0] == 2),
        ("a tool failure is never a pass", overall({"a": {"status": TOOL_FAILED}})[0] == 2),
        ("nothing checked is never a pass", overall({})[0] == 2),
        ("all passed gives exit 0", overall({"a": {"status": PASS}, "b": {"status": PASS}})[0] == 0),
    ]
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
    failed = [name for name, ok in checks if not ok]
    print(f"selftest: {len(checks) - len(failed)} of {len(checks)} passed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Install a built wheel into a fresh environment and check it.")
    ap.add_argument("--wheel")
    ap.add_argument("--out")
    ap.add_argument("--wheelhouse")
    ap.add_argument("--workdir")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.wheel or not args.out:
        ap.print_usage()
        return 2
    return check_wheel(Path(args.wheel), Path(args.out), Path(args.wheelhouse) if args.wheelhouse else None,
                       Path(args.workdir) if args.workdir else None, args.keep)


if __name__ == "__main__":
    sys.exit(main())
