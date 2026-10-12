"""
E51 (WP-020) provisioning checks. PLAN_020 section 5.6 and 6.1.

Run with the repository interpreter, from the repository root:

    python docs/fie_rebuild_2026/experiments/E51/provision.py preflight --out <dir>
    python docs/fie_rebuild_2026/experiments/E51/provision.py index fie-sdk --expect 1.18.0 --out <dir>
    python docs/fie_rebuild_2026/experiments/E51/provision.py resolve --name <set> --python <venv python> --out <dir> -- <requirements...>
    python docs/fie_rebuild_2026/experiments/E51/provision.py download --name <set> --python <venv python> --out <dir> --cache <dir>

Nothing here installs anything. `resolve` is a pip dry run; `download` fetches wheels
into a local cache with hashes required. Every pip call names the one approved index
explicitly and runs with inherited configuration removed.

The only hosts this script may contact are pypi.org and files.pythonhosted.org.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

INDEX_URL = "https://pypi.org/simple"
JSON_API = "https://pypi.org/pypi"
APPROVED_HOSTS = {"pypi.org", "files.pythonhosted.org"}
FILE_HOST = "files.pythonhosted.org"

# Variables that can redirect pip or its TLS. Removed from every pip environment.
STRIP_PREFIXES = ("PIP_",)
STRIP_NAMES = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "REQUESTS_CA_BUNDLE",
               "SSL_CERT_FILE", "CURL_CA_BUNDLE", "PYTHONPATH", "PYTHONHOME"}

PIP_FLAGS = ["--isolated", "--index-url", INDEX_URL, "--no-cache-dir", "--only-binary", ":all:",
             "--disable-pip-version-check", "-vv"]

_HOME = str(Path.home())


def redact(text: str) -> str:
    """Evidence must not carry the user's home path or name."""
    out = text.replace(_HOME, "<HOME>").replace(_HOME.replace("\\", "/"), "<HOME>")
    out = out.replace(_HOME.replace("\\", "\\\\"), "<HOME>")
    return out.replace(os.environ.get("USERNAME", "\0"), "<USER>")


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=1, sort_keys=True)
    path.write_text(redact(text), encoding="utf-8", newline="\n")


def pip_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(STRIP_PREFIXES) and k.upper() not in STRIP_NAMES}
    env["PIP_CONFIG_FILE"] = os.devnull          # no config file can apply
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def fetch_json(url: str) -> dict:
    """PyPI JSON through the standard library, with proxy handling switched off. No pip, no cache."""
    host = re.match(r"https://([^/]+)/", url).group(1)
    if host not in APPROVED_HOSTS:
        raise SystemExit(f"refusing to contact {host}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "fie-e51-provision"})
    with opener.open(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── preflight ────────────────────────────────────────────────────────────────

def cmd_preflight(args) -> int:
    out = Path(args.out)
    problems = []
    cfg = subprocess.run([sys.executable, "-m", "pip", "config", "debug"], capture_output=True, text=True)
    lst = subprocess.run([sys.executable, "-m", "pip", "config", "list"], capture_output=True, text=True)
    neutral = subprocess.run([sys.executable, "-m", "pip", "--isolated", "config", "list"],
                             capture_output=True, text=True, env=pip_env())
    existing = re.findall(r"^\s+(\S.*?), exists: True", cfg.stdout, flags=re.M)
    variables = sorted(k for k in os.environ
                       if k.upper().startswith(STRIP_PREFIXES) or k.upper() in STRIP_NAMES)
    stdlib_proxies = urllib.request.getproxies()
    hosts_file = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "drivers" / "etc" / "hosts"
    hosts_hits = []
    if hosts_file.exists():
        hosts_hits = [ln.strip() for ln in hosts_file.read_text(encoding="utf-8", errors="replace").splitlines()
                      if re.search(r"pypi|pythonhosted", ln, re.I) and not ln.strip().startswith("#")]
    winhttp = subprocess.run(["netsh", "winhttp", "show", "proxy"], capture_output=True, text=True).stdout
    inet = subprocess.run(
        ["reg", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings"],
        capture_output=True, text=True).stdout
    proxy_enable = re.search(r"ProxyEnable\s+REG_DWORD\s+0x(\w+)", inet)
    proxy_server = re.search(r"ProxyServer\s+REG_SZ\s+(\S+)", inet)
    auto_config = re.search(r"AutoConfigURL\s+REG_SZ\s+(\S+)", inet)

    if stdlib_proxies:
        problems.append("the standard library reports a proxy")
    if proxy_enable and int(proxy_enable.group(1), 16) != 0:
        problems.append("a system proxy is enabled in Internet Settings")
    if auto_config:
        problems.append("a proxy auto-configuration URL is set")
    if "Direct access" not in winhttp:
        problems.append("WinHTTP is not set to direct access")
    if hosts_hits:
        problems.append("the hosts file redirects a PyPI host name")
    if neutral.stdout.strip():
        problems.append("pip still reports settings after neutralization")

    report = {
        "pip_version": subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True,
                                      text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "pip_config_debug": cfg.stdout,
        "pip_config_list_inherited": lst.stdout,
        "pip_config_files_existing": existing,
        "pip_config_list_after_neutralization": neutral.stdout,
        "redirecting_variables_set": variables,          # names only
        "stdlib_proxies": stdlib_proxies,
        "winhttp_proxy": winhttp.strip(),
        "internet_settings": {"ProxyEnable": proxy_enable.group(1) if proxy_enable else None,
                              "ProxyServer": "set" if proxy_server else None,
                              "AutoConfigURL": "set" if auto_config else None},
        "hosts_file_pypi_entries": hosts_hits,
        "pip_flags_used_everywhere": PIP_FLAGS,
        "problems": problems,
        "verdict": "PASS" if not problems else "STOP",
    }
    save(out / "provisioning_preflight.json", report)
    print(json.dumps({k: report[k] for k in ("pip_version", "pip_config_files_existing",
                                             "redirecting_variables_set", "stdlib_proxies",
                                             "hosts_file_pypi_entries", "problems", "verdict")}, indent=1))
    return 0 if not problems else 2


# ── index ────────────────────────────────────────────────────────────────────

def cmd_index(args) -> int:
    out = Path(args.out)
    data = fetch_json(f"{JSON_API}/{args.project}/json")
    save(out / f"pypi_{args.project}.json", data)
    latest = data["info"]["version"]
    releases = {v: [{"filename": f["filename"], "size": f["size"], "sha256": f["digests"]["sha256"],
                     "upload_time": f["upload_time_iso_8601"], "yanked": f.get("yanked", False),
                     "host": re.match(r"https://([^/]+)/", f["url"]).group(1)} for f in files]
                for v, files in data["releases"].items()}
    pinned = fetch_json(f"{JSON_API}/{args.project}/{args.expect}/json")
    summary = {
        "project": args.project,
        "latest_version": latest,
        "expected_version": args.expect,
        "latest_is_expected": latest == args.expect,
        "release_count": len(releases),
        "newest_five": sorted(releases, key=lambda v: max((f["upload_time"] for f in releases[v]), default=""))[-5:],
        "expected_files": releases.get(args.expect, []),
        "expected_requires_dist": pinned["info"].get("requires_dist"),
        "expected_requires_python": pinned["info"].get("requires_python"),
        "expected_yanked": pinned["info"].get("yanked"),
    }
    save(out / f"pypi_{args.project}_summary.json", summary)
    print(json.dumps(summary, indent=1))
    return 0 if summary["latest_is_expected"] else 3


# ── resolve (dry run) ────────────────────────────────────────────────────────

def _sizes(name: str, version: str, filename: str) -> dict:
    meta = fetch_json(f"{JSON_API}/{name}/{version}/json")
    for f in meta["urls"]:
        if f["filename"] == filename:
            return {"size": f["size"], "pypi_sha256": f["digests"]["sha256"]}
    return {"size": None, "pypi_sha256": None}


def _local_wheels(args) -> dict:
    """
    Locally built wheels explicitly allowed for this run: {filename: {"path", "sha256", "size"}}.

    The one approved use (Stage B remediation) is the `langdetect` wheel built during provisioning
    from the digest-verified PyPI source archive, because PyPI publishes no Python 3 wheel for it.
    Each must sit alone in its own directory, which is passed to pip as --find-links.
    """
    found = {}
    for raw in getattr(args, "local_wheel", None) or []:
        path = Path(raw).resolve()
        others = [q.name for q in path.parent.iterdir() if q != path]
        if others:
            raise SystemExit(f"the directory of {path.name} must contain only that wheel; found {others}")
        found[path.name] = {"path": path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size": path.stat().st_size}
    return found


def _find_links(local: dict) -> list:
    flags = []
    for info in local.values():
        flags += ["--find-links", str(info["path"].parent)]
    return flags


def cmd_resolve(args) -> int:
    out = Path(args.out)
    local = _local_wheels(args)
    report_path = out / "resolve" / f"{args.name}.report.json"
    log_path = out / "resolve" / f"{args.name}.pip.log"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "--python", args.python, "install", "--dry-run", "--ignore-installed",
           "--report", str(report_path), *PIP_FLAGS, *_find_links(local), *args.requirements]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=pip_env())
    save(log_path, "COMMAND: " + " ".join(cmd) + f"\nEXIT: {proc.returncode}\n\n" + proc.stdout + "\n--- stderr ---\n"
         + proc.stderr)
    log = proc.stdout + proc.stderr
    hosts = sorted(set(re.findall(r"https?://([A-Za-z0-9.\-]+)", log)))
    full_downloads = sorted(set(re.findall(r"Downloading (\S+\.whl)\b(?!\.metadata)", log)))
    full_downloads = [u for u in full_downloads if not u.endswith(".metadata")]
    result = {"name": args.name, "requirements": args.requirements, "pip_exit": proc.returncode,
              "hosts_in_log": hosts, "unapproved_hosts": [h for h in hosts if h not in APPROVED_HOSTS],
              "full_wheel_downloads_during_dry_run": full_downloads}
    if proc.returncode != 0:
        tail = [ln for ln in log.splitlines() if ln.startswith("ERROR")][-6:]
        result["errors"] = tail
        save(out / "resolve" / f"{args.name}.summary.json", result)
        print(json.dumps(result, indent=1))
        return 4

    report = json.loads(report_path.read_text(encoding="utf-8"))
    save(report_path, report)                      # re-save redacted
    items, total, bad_hosts, digest_mismatch, local_used = [], 0, [], [], []
    for it in report["install"]:
        name, version = it["metadata"]["name"], it["metadata"]["version"]
        url = it["download_info"]["url"]
        sha = it["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
        filename = url.rsplit("/", 1)[1].split("#")[0]
        if url.startswith("file:"):
            # Not from PyPI. Accepted only if it is exactly one of the declared local wheels.
            info = local.get(filename)
            if info is None or sha != info["sha256"]:
                bad_hosts.append(f"undeclared local file: {filename}")
                continue
            local_used.append(filename)
            items.append({"name": name, "version": version, "filename": filename, "url": "local:" + filename,
                          "host": "local-build", "sha256": sha, "size": info["size"], "pypi_sha256": None,
                          "origin": "locally built from the digest-verified PyPI source archive"})
            total += info["size"]
            continue
        host = re.match(r"https?://([^/]+)/", url).group(1)
        extra = _sizes(name, version, filename)
        if host != FILE_HOST:
            bad_hosts.append(url)
        if sha is None or sha != extra["pypi_sha256"]:
            digest_mismatch.append(filename)
        total += extra["size"] or 0
        items.append({"name": name, "version": version, "filename": filename, "url": url, "host": host,
                      "sha256": sha, "origin": "pypi", **extra})
    items.sort(key=lambda d: -(d["size"] or 0))
    pins = "".join(f"{d['name']}=={d['version']} --hash=sha256:{d['sha256']}\n"
                   for d in sorted(items, key=lambda d: d["name"].lower()))
    save(out / "pins" / f"{args.name}.txt", pins)
    result.update({"distributions": len(items), "total_download_bytes": total,
                   "total_download_mb": round(total / 1e6, 1), "items": items,
                   "urls_not_on_file_host": bad_hosts, "digest_not_matching_pypi": digest_mismatch,
                   "local_wheels_declared": {k: {"sha256": v["sha256"], "size": v["size"]} for k, v in local.items()},
                   "local_wheels_used": local_used,
                   "from_pypi": sum(1 for d in items if d.get("origin") == "pypi"),
                   "pip_version_used": report.get("pip_version"),
                   "verdict": "PASS" if not (bad_hosts or digest_mismatch or result["unapproved_hosts"]) else "STOP"})
    save(out / "resolve" / f"{args.name}.summary.json", result)
    brief = {k: result[k] for k in ("name", "pip_exit", "distributions", "from_pypi", "local_wheels_used",
                                    "total_download_mb", "hosts_in_log",
                                    "unapproved_hosts", "full_wheel_downloads_during_dry_run",
                                    "urls_not_on_file_host", "digest_not_matching_pypi", "verdict")}
    brief["largest"] = [(d["name"], d["version"], round((d["size"] or 0) / 1e6, 1)) for d in items[:8]]
    print(json.dumps(brief, indent=1))
    return 0 if result["verdict"] == "PASS" else 5


# ── download (hashes required) and verify ────────────────────────────────────

def cmd_download(args) -> int:
    out, cache = Path(args.out), Path(args.cache)
    pins = out / "pins" / f"{args.name}.txt"
    summary = json.loads((out / "resolve" / f"{args.name}.summary.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in cache.iterdir()}
    local = _local_wheels(args)
    cmd = [sys.executable, "-m", "pip", "--python", args.python, "download", "--require-hashes", "--no-deps",
           "-r", str(pins), *PIP_FLAGS, *_find_links(local), "-d", str(cache)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=pip_env())
    log = proc.stdout + proc.stderr
    save(out / "download" / f"{args.name}.pip.log", "COMMAND: " + " ".join(cmd) + f"\nEXIT: {proc.returncode}\n\n" + log)
    hosts = sorted(set(re.findall(r"https?://([A-Za-z0-9.\-]+)", log)))
    expected = {d["filename"]: d for d in summary["items"]}
    checks, problems = [], []
    for name, d in sorted(expected.items()):
        path = cache / name
        if not path.exists():
            problems.append(f"missing from cache: {name}")
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if d.get("origin", "pypi") == "pypi":
            ok = sha == d["sha256"] == d["pypi_sha256"] and path.stat().st_size == d["size"]
        else:   # a declared local wheel: must equal the resolution and the declared file; never claimed as PyPI's
            ok = name in local and sha == d["sha256"] == local[name]["sha256"]
        checks.append({"filename": name, "sha256": sha, "origin": d.get("origin", "pypi"),
                       "matches_resolution_and_pypi": ok})
        if not ok:
            problems.append(f"hash or size mismatch: {name}")
    new_files = {p.name for p in cache.iterdir()} - before
    extra = sorted(new_files - set(expected))
    if extra:
        problems.append(f"files not in the pinned set: {extra}")
    if proc.returncode != 0:
        problems.append(f"pip download exited {proc.returncode}")
    bad_hosts = [h for h in hosts if h not in APPROVED_HOSTS]
    if bad_hosts:
        problems.append(f"unapproved hosts in the log: {bad_hosts}")
    result = {"name": args.name, "pip_exit": proc.returncode, "hosts_in_log": hosts, "files_expected": len(expected),
              "files_verified": sum(c["matches_resolution_and_pypi"] for c in checks), "checks": checks,
              "extra_files": extra, "problems": problems, "verdict": "PASS" if not problems else "STOP"}
    save(out / "download" / f"{args.name}.summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=1))
    return 0 if not problems else 6


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--out", required=True)
    p = sub.add_parser("index"); p.add_argument("project"); p.add_argument("--expect", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("resolve"); p.add_argument("--name", required=True); p.add_argument("--python", required=True)
    p.add_argument("--out", required=True); p.add_argument("--local-wheel", action="append")
    p.add_argument("requirements", nargs="+")
    p = sub.add_parser("download"); p.add_argument("--name", required=True); p.add_argument("--python", required=True)
    p.add_argument("--out", required=True); p.add_argument("--cache", required=True)
    p.add_argument("--local-wheel", action="append")
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "index": cmd_index, "resolve": cmd_resolve, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
