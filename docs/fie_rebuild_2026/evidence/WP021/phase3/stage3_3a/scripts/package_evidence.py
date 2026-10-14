"""Copy the Stage 3.3a results into the repository evidence directory, deduplicated and scrubbed."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

work, ev, script_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ev.mkdir(parents=True, exist_ok=False)
(ev / "members").mkdir()
(ev / "logs").mkdir()
(ev / "scripts").mkdir()
R = work / "results"


def load(n):
    return json.loads((R / n).read_text(encoding="utf-8"))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8", newline="\n")


builds = load("builds.json")
member_index = {}
summary_runs = []
for run in builds["runs"]:
    if "blocked" in run:
        summary_runs.append(run)
        continue
    s = {k: run[k] for k in ("setup", "scenario", "variant", "repeat", "added_files", "source_tree_unchanged_by_builds",
                             "source_tree_new_files") if k in run}
    s["sdist_unpacked_has_gitignore"] = run.get("sdist_unpacked_has_gitignore")
    s["sdist_unpacked_has_dot_git"] = run.get("sdist_unpacked_has_dot_git")
    for kind in ("sdist", "wheel_direct", "wheel_from_sdist"):
        b = run[kind]
        s[kind] = b
        mem = run.get(f"{kind}_members")
        if mem and b.get("outputs"):
            arch = b["outputs"][0]
            key = arch["sha256"][:16]
            member_index.setdefault(key, {"archive": arch["name"], "archive_sha256": arch["sha256"], "members": mem})
            s[kind + "_member_file"] = f"members/{key}.json"
            s[kind + "_member_count"] = len(mem)
            s[kind + "_fie_models"] = {n: mem[n] for n in sorted(mem) if n.startswith("fie/models/")}
        s[kind + "_classification"] = run.get(f"{kind}_classification")
    summary_runs.append(s)
for key, obj in member_index.items():
    save(ev / "members" / f"{key}.json", obj)
save(ev / "builds.json", {"head": builds["head"], "versions": builds["versions"],
                          "guard_selftest": builds["guard_selftest"], "runs": summary_runs})
for n in ("exports.json", "scenario_inputs.json", "probes.json"):
    shutil.copy(R / n, ev / n)
for log in sorted((work / "logs").glob("*.log")):
    shutil.copy(log, ev / "logs" / log.name)
for f in ("run_experiment.py", "package_evidence.py"):
    shutil.copy(script_dir / f, ev / "scripts" / f)

leaks = []
for p in ev.rglob("*"):
    local = (Path.home().name, "AppData", str(Path.home()), Path.home().as_posix())
    if p.is_file() and re.search("|".join(map(re.escape, local)), p.read_text(encoding="utf-8", errors="replace")):
        leaks.append(p.relative_to(ev).as_posix())
lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ev).as_posix()}"
         for p in sorted(ev.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
(ev / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("files", len(lines), "distinct archives", len(member_index), "leaks", leaks)
