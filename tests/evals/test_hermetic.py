"""
Hermetic guard.

Every test here runs the guard in a CHILD interpreter. The guard is built on a
Python audit hook, which cannot be removed, so it must never be installed in the
pytest process.

All network targets are reserved, never-routable names and addresses (RFC 2606
`.invalid`, RFC 5737 TEST-NET-1), or the product's own endpoints with the guard
armed first — so nothing leaves the machine whether or not the guard works.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evals import hermetic
from _helpers import REPO_ROOT, child_json

PRELUDE = (
    "import json, logging, sys, threading\n"
    "from evals import hermetic\n"
    "hermetic.install()\n"
    "logging.disable(logging.CRITICAL)\n"
)
REPORT = (
    "\nev = hermetic.violations()\n"
    "print(json.dumps({'n': len(ev), 'events': [e['event'] for e in ev], "
    "'targets': [e['target'] for e in ev], 'out': globals().get('out')}))\n"
)


def guarded(body: str, tmp_path: Path, env: dict | None = None) -> dict:
    return child_json(PRELUDE + body + REPORT, tmp_path, env)


def attempt(expr: str) -> str:
    """Child code: evaluate `expr`, recording whether the guard refused it."""
    return (
        "try:\n"
        f"    {expr}\n"
        "    out = 'NOT BLOCKED'\n"
        "except hermetic.GuardViolation as e:\n"
        "    out = 'blocked'\n"
        "except Exception as e:\n"
        "    out = 'wrapped:' + type(e).__name__\n"
    )


# ── the guard proves itself ──────────────────────────────────────────────────

def test_selftest_denies_and_records_every_canary(tmp_path):
    out = child_json(
        "import json\nfrom evals import hermetic\nhermetic.install()\n"
        "st = hermetic.selftest()\ns = hermetic.summary()\n"
        "print(json.dumps({'st': st, 'violations': s['violations'], "
        "'canaries': s['canary_events'], 'audit': s['audit_events_seen']}))\n",
        tmp_path)
    assert out["st"]["passed"] is True
    assert set(out["st"]["checks"]) == {
        "ipv4_socket_audit", "dns_audit", "dns_patched", "connect_patched", "subprocess"}
    assert all(out["st"]["checks"].values())
    assert out["canaries"] == 5
    assert out["violations"] == 0, "canaries must not count as violations"
    assert out["audit"] > 0, "the audit hook never fired"


def test_selftest_requires_an_installed_guard():
    # Safe in-process: this never installs anything.
    assert hermetic.installed() is False, "the guard must never be armed in pytest"
    with pytest.raises(RuntimeError):
        hermetic.selftest()


def test_events_are_written_to_the_log_file(tmp_path):
    log = tmp_path / "guard.jsonl"
    code = (
        "import json, socket\nfrom evals import hermetic\n"
        f"hermetic.install(event_log={str(log)!r})\n"
        "try:\n    socket.create_connection(('192.0.2.1', 9), timeout=1)\n"
        "except hermetic.GuardViolation:\n    pass\n"
        "print(json.dumps({'n': hermetic.violation_count()}))\n"
    )
    assert child_json(code, tmp_path)["n"] == 1
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 1 and lines[0]["canary"] is False
    assert "192.0.2.1" in lines[0]["target"]


# ── generic egress routes ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expr", [
    ("direct socket creation", "import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)"),
    ("raw _socket creation", "import _socket; _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)"),
    ("udp socket creation", "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM)"),
    ("ipv6 socket creation", "import socket; socket.socket(socket.AF_INET6, socket.SOCK_STREAM)"),
    ("create_connection", "import socket; socket.create_connection(('192.0.2.1', 9), timeout=1)"),
    ("loopback connection", "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)"),
    ("dns getaddrinfo", "import socket; socket.getaddrinfo('fie-eval-canary.invalid', 443)"),
    ("dns raw getaddrinfo", "import _socket; _socket.getaddrinfo('fie-eval-canary.invalid', 443)"),
    ("dns gethostbyname", "import socket; socket.gethostbyname('fie-eval-canary.invalid')"),
    ("dns raw gethostbyname", "import _socket; _socket.gethostbyname('fie-eval-canary.invalid')"),
    ("urllib", "import urllib.request; urllib.request.urlopen('http://192.0.2.1/', timeout=1)"),
    ("http.client", "import http.client; http.client.HTTPConnection('192.0.2.1', 80, timeout=1).connect()"),
    ("https client", "import http.client; http.client.HTTPSConnection('192.0.2.1', 443, timeout=1).connect()"),
    ("subprocess.run", "import subprocess; subprocess.run([sys.executable, '-c', 'pass'])"),
    ("subprocess.Popen", "import subprocess; subprocess.Popen(['fie-eval-canary-not-a-program'])"),
    ("os.system", "import os; os.system('echo hi')"),
])
def test_egress_route_is_denied_and_recorded(tmp_path, name, expr):
    out = guarded(attempt(expr), tmp_path)
    assert out["out"] == "blocked", f"{name}: {out}"
    assert out["n"] >= 1, f"{name}: denied but not recorded"


def test_requests_library_is_denied_and_recorded(tmp_path):
    pytest.importorskip("requests")
    out = guarded(attempt("import requests; requests.get('http://192.0.2.1/', timeout=1)"), tmp_path)
    # requests wraps the socket error in its own exception class; what matters
    # is that the attempt was refused and recorded.
    assert out["out"] in ("blocked", "wrapped:ConnectionError", "wrapped:ProxyError",
                          "wrapped:ConnectTimeout"), out
    assert out["n"] >= 1


def test_asyncio_connection_to_an_address_literal_is_denied(tmp_path):
    """On Windows asyncio connects through a native call; the socket is refused first."""
    body = (
        "import asyncio\n"
        "async def go():\n"
        "    await asyncio.open_connection('192.0.2.1', 9)\n"
        "try:\n"
        "    asyncio.run(go())\n"
        "    out = 'NOT BLOCKED'\n"
        "except BaseException as e:\n"
        "    out = type(e).__name__\n"
    )
    out = guarded(body, tmp_path)
    assert out["out"] != "NOT BLOCKED"
    assert out["n"] >= 1


def test_prearm_warmups_are_listed_and_exempt_nothing_afterwards(tmp_path):
    """
    Two local-only probes run before arming (platform's `ver`, urllib3's IPv6
    check). After arming they must not fire again, and creation stays refused.
    """
    body = (
        "import platform, socket\n"
        "system = platform.system(); node = platform.node(); proc = platform.processor()\n"
        "import urllib3\n"
        "before = hermetic.violation_count()\n"
        "try:\n"
        "    socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
        "    made = 'CREATED'\n"
        "except hermetic.GuardViolation:\n"
        "    made = 'denied'\n"
        "out = {'prearm': hermetic.summary()['prearm'], 'system': bool(system),\n"
        "       'before': before, 'made': made}\n"
    )
    out = guarded(body, tmp_path)
    assert out["out"]["prearm"][0] == "platform.uname"
    assert out["out"]["system"] is True
    assert out["out"]["before"] == 0, "platform or urllib3 still triggered the guard"
    assert out["out"]["made"] == "denied"
    assert out["n"] == 1 and out["events"] == ["socket.__new__"], out


def test_local_calls_are_not_false_positives(tmp_path):
    body = (
        "import platform, socket, tempfile, concurrent.futures, hashlib, re, json as _j\n"
        "node = platform.node(); host = socket.gethostname()\n"
        "with concurrent.futures.ThreadPoolExecutor(4) as pool:\n"
        "    vals = list(pool.map(lambda x: x * 2, range(8)))\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    open(d + '/f.txt', 'w').write('x')\n"
        "out = 'ok' if vals[-1] == 14 and host else 'bad'\n"
    )
    out = guarded(body, tmp_path)
    assert out == {"n": 0, "events": [], "targets": [], "out": "ok"}


# ── the product's own egress paths ───────────────────────────────────────────

def test_telemetry_ping_is_denied_and_recorded(tmp_path):
    body = (
        "import fie\n"
        "for t in threading.enumerate():\n"
        "    if t is not threading.current_thread():\n"
        "        t.join(timeout=15)\n"
        "out = 'imported'\n"
    )
    out = guarded(body, tmp_path, {"FIE_NO_TELEMETRY": None})
    assert out["out"] == "imported"
    assert out["n"] >= 1, "the import-time telemetry ping was not intercepted"
    assert any("onrender.com" in t for t in out["targets"]), out["targets"]


def test_translation_call_is_denied_and_recorded(tmp_path):
    pytest.importorskip("deep_translator")
    body = (
        "from fie.multilingual import translate_to_english\n"
        "r = translate_to_english('Подскажите, пожалуйста, как лучше добраться из Москвы "
        "в Санкт-Петербург на поезде?')\n"
        "out = 'none' if r is None else 'TRANSLATED'\n"
    )
    out = guarded(body, tmp_path)
    assert out["out"] == "none", "the translator returned text: something reached the network"
    assert out["n"] >= 1


def test_tiebreaker_call_is_denied_and_recorded(tmp_path):
    pytest.importorskip("requests")
    body = (
        "from fie.llama_guard import query_llama_guard\n"
        "try:\n"
        "    query_llama_guard('an uncertain prompt that would be sent to the tiebreaker')\n"
        "    out = 'RETURNED'\n"
        "except RuntimeError:\n"
        "    out = 'raised'\n"
    )
    out = guarded(body, tmp_path, {"GROQ_API_KEY": "gsk_dummy_value_for_guard_test_only"})
    assert out["out"] == "raised"
    assert out["n"] >= 1


def test_model_auto_download_is_denied_and_recorded(tmp_path):
    pytest.importorskip("numpy")
    target = tmp_path / "no-model-here"
    body = (
        "from pathlib import Path\n"
        "from fie.onnx_encoder import _ensure_model_downloaded\n"
        f"d = Path({str(target)!r})\n"
        "_ensure_model_downloaded(d)\n"
        "out = sorted(p.name for p in d.glob('*')) if d.exists() else []\n"
    )
    out = guarded(body, tmp_path, {"FIE_NO_AUTO_DOWNLOAD": None})
    assert out["out"] == [], f"files appeared in the model directory: {out['out']}"
    assert out["n"] >= 1
    assert any("github.com" in t for t in out["targets"]), out["targets"]


def test_fie_imports_and_warms_up_with_zero_events_when_closed_at_source(tmp_path, need_models):
    body = (
        "import fie.adversarial as adv\n"
        "status = adv.warmup()\n"
        "for t in threading.enumerate():\n"
        "    if t is not threading.current_thread() and not t.name.startswith('fie-layer'):\n"
        "        t.join(timeout=15)\n"
        "out = status.get('pair_classifier')\n"
    )
    out = guarded(body, tmp_path)
    assert out["out"] == "ready"
    assert out["n"] == 0, f"unexpected network attempts: {out['targets']}"


# ── static tripwire for native downloaders ───────────────────────────────────

def test_fie_does_not_use_native_downloaders():
    """`from_pretrained` downloads in native code, which a Python guard cannot see."""
    offenders = []
    for path in sorted((REPO_ROOT / "fie").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in ("from_pretrained(", "hf_hub_download(", "snapshot_download("):
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
    assert not offenders, offenders


# ── orchestrator mode ────────────────────────────────────────────────────────

def test_orchestrator_mode_allows_only_the_listed_executables(tmp_path):
    code = (
        "import json, subprocess, sys\nfrom evals import hermetic\n"
        "hermetic.install(mode='orchestrator', allow_exec=(sys.executable,))\n"
        "ok = subprocess.run([sys.executable, '-c', 'print(40 + 2)'], executable=sys.executable,\n"
        "                    capture_output=True, text=True).stdout.strip()\n"
        "try:\n"
        "    subprocess.run(['fie-eval-canary-not-a-program'])\n"
        "    other = 'NOT BLOCKED'\n"
        "except hermetic.GuardViolation:\n"
        "    other = 'blocked'\n"
        "try:\n"
        "    import socket; socket.create_connection(('192.0.2.1', 9), timeout=1)\n"
        "    net = 'NOT BLOCKED'\n"
        "except hermetic.GuardViolation:\n"
        "    net = 'blocked'\n"
        "print(json.dumps({'ok': ok, 'other': other, 'net': net, 'st': hermetic.selftest()['passed']}))\n"
    )
    out = child_json(code, tmp_path)
    assert out == {"ok": "42", "other": "blocked", "net": "blocked", "st": True}


# ── import blocking ──────────────────────────────────────────────────────────

def test_import_blocker_makes_packages_unimportable(tmp_path):
    code = (
        "import json\nfrom evals import hermetic\n"
        "b = hermetic.block_imports(['colorsys', 'wave'])\n"
        "res = {}\n"
        "for name in ('colorsys', 'wave', 'colorsys.sub', 'textwrap'):\n"
        "    try:\n        __import__(name); res[name] = 'imported'\n"
        "    except ImportError:\n        res[name] = 'blocked'\n"
        "print(json.dumps({'res': res, 'hits': sorted(b.hits)}))\n"
    )
    out = child_json(code, tmp_path)
    assert out["res"] == {"colorsys": "blocked", "wave": "blocked", "colorsys.sub": "blocked",
                          "textwrap": "imported"}
    # A submodule import fails at its blocked parent, so only top-level names are hit.
    assert out["hits"] == ["colorsys", "wave"]


def test_import_blocker_refuses_an_already_loaded_package():
    # No hook involved: block_imports only edits sys.meta_path, and it raises
    # before doing so here.
    with pytest.raises(RuntimeError, match="already-imported"):
        hermetic.block_imports(["json"])


# ── environment allowlist ────────────────────────────────────────────────────

def test_sanitized_env_drops_everything_not_on_the_allowlist(tmp_path):
    parent = dict(os.environ)
    parent.update({
        "GROQ_API_KEY": "gsk_PLANTED_SECRET_1", "MONGODB_URI": "mongodb+srv://PLANTED_SECRET_2",
        "JWT_SECRET_KEY": "PLANTED_SECRET_3", "PYPI_TOKEN": "pypi-PLANTED_SECRET_4",
        "HUGGING_FACE_TOKEN": "hf_PLANTED_SECRET_5", "FIE_API_KEY": "fie-PLANTED_SECRET_6",
        "REDIS_URL": "redis://PLANTED_SECRET_7", "LIBRETRANSLATE_URL": "http://PLANTED_SECRET_8",
        "PYTHONPATH": "/somewhere/else", "FIE_UNCERTAIN_ALLOW": "1",
    })
    env = hermetic.sanitized_env(parent, {"FIE_PAIR_VERSION": "v6_3b", "FIE_NO_TELEMETRY": 1}, tmp_path)
    blob = json.dumps(env)
    assert "PLANTED_SECRET" not in blob
    for name in ("GROQ_API_KEY", "MONGODB_URI", "JWT_SECRET_KEY", "PYTHONPATH", "FIE_UNCERTAIN_ALLOW",
                 "REDIS_URL", "LIBRETRANSLATE_URL"):
        assert name not in env
    assert env["FIE_PAIR_VERSION"] == "v6_3b" and env["FIE_NO_TELEMETRY"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["HTTPS_PROXY"] == hermetic.DEAD_PROXY
    assert Path(env["HOME"]).is_dir() and str(tmp_path) in env["HOME"]
    assert str(tmp_path) in env["TEMP"]
    if "PATH" in os.environ:
        assert env["PATH"] == os.environ["PATH"]
    assert len({k.upper() for k in env}) == len(env) or os.name != "nt", "duplicate names on Windows"
