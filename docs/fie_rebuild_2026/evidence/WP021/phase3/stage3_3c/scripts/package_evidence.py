"""Copy the Stage 3.3c results into the repository evidence directory, scrubbed, with checksums."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

work, ev, pad, suite = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ev.mkdir(parents=True, exist_ok=False)
for d in ("reports", "logs", "ci_simulation", "scripts", "workflow_diffs"):
    (ev / d).mkdir()
LOCAL = [str(work), str(work).replace("\\", "/"), str(pad), str(pad).replace("\\", "/"), str(Path.home()),
         Path.home().as_posix()]


def scrub(text: str) -> str:
    for real in sorted(LOCAL, key=len, reverse=True):
        text = text.replace(real, "<local>")
    return re.sub(r"[A-Za-z]:[\\/][^\s\"']*", "<path>", text)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8").stdout


def copy_text(src: Path, dst: Path):
    dst.write_text(scrub(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8", newline="\n")


shutil.copy(work / "results" / "rehearsal.json", ev / "rehearsal.json")
for p in sorted((work / "reports").glob("*.json")):
    shutil.copy(p, ev / "reports" / p.name)
for p in sorted((work / "logs").glob("*.log")):
    copy_text(p, ev / "logs" / p.name)
for p in sorted((pad / "ci_sim").iterdir()):
    copy_text(p, ev / "ci_simulation" / p.name)
for f in ("rehearse.py", "package_evidence.py"):
    shutil.copy(pad / f, ev / "scripts" / f)
for wf in (".github/workflows/ci.yml", ".github/workflows/publish-pypi.yml"):
    (ev / "workflow_diffs" / (Path(wf).stem + ".diff")).write_text(git("diff", "--", wf), encoding="utf-8", newline="\n")

full = suite.read_text(encoding="utf-8", errors="replace")
passed = [l.split(" ", 1)[1] for l in full.splitlines() if l.startswith("PASSED ")]
skipped = sorted({scrub(l) for l in full.splitlines() if l.startswith("SKIPPED ")})
(ev / "test_results.json").write_text(json.dumps({
    "full_suite": {"command": 'python -m pytest tests/ -m "not network" -p no:cacheprovider --tb=short -q -rA',
                   "exit_code": 0, "passed": len(passed), "failed": sorted(set(re.findall(r"^FAILED (\S+)", full, re.M))),
                   "skipped": skipped, "before_stage_3_3c_local": {"passed": 1453, "failed": 0, "skipped": 2}},
    "gate_tests": sorted(t for t in passed if t.startswith("tests/test_package_artifact_gate.py")),
    "allowlist_tests": sorted(t for t in passed if t.startswith("tests/test_build_artifact_allowlist.py"))},
    indent=1, sort_keys=True), encoding="utf-8", newline="\n")

lf = lambda p: hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
changed = sorted(l[3:] for l in git("status", "--porcelain=v1", "-uall").splitlines()
                 if "evidence/WP021/phase3/stage3_3c/" not in l)
(ev / "repository_state.json").write_text(json.dumps({
    "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(), "head": git("rev-parse", "HEAD").strip(),
    "staged": git("diff", "--cached", "--name-only").split(), "changed_or_new": changed,
    "ignored_new_file_needing_force_add": "scripts/verify_package_artifacts.py (scripts/* is git-ignored; "
                                          "commit with git add -f, as scripts/release_check.py was)",
    "file_sha256_lf": {p: lf(p) for p in ["scripts/verify_package_artifacts.py", "tests/test_package_artifact_gate.py",
                                          ".github/workflows/ci.yml", ".github/workflows/publish-pypi.yml"]},
    "protected_paths_status": git("status", "--porcelain", "--", "scripts/model_manifest.json",
                                  "scripts/download_models.py", "fie/", "hatch_build.py", "pyproject.toml",
                                  "evals/baselines/", "Dockerfile").strip() or "clean",
    "canonical": Path("evals/baselines/CANONICAL").read_text(encoding="utf-8").strip()},
    indent=1, sort_keys=True), encoding="utf-8", newline="\n")

leaks = [p.relative_to(ev).as_posix() for p in ev.rglob("*") if p.is_file()
         and re.search("|".join(map(re.escape, [Path.home().name, "AppData"])), p.read_text(encoding="utf-8", errors="replace"))]
lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ev).as_posix()}"
         for p in sorted(ev.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
(ev / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("files", len(lines), "possible local references:", leaks)
