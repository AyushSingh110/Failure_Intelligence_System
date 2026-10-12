"""
Hermetic execution: proof that an evaluation run opened no network connection.

THE CLAIM, STATED EXACTLY
-------------------------
"Zero outbound connection attempts through the Python runtime, with every known
egress path closed at its source."

It is not a claim of operating-system or native-code isolation. Native code that
calls the OS network API without going through CPython's `socket` module is
invisible to this guard (see evals/README.md, "Limits").

WHY IT IS BUILT THIS WAY
------------------------
The product catches exceptions around its own network calls and carries on, so a
guard that only *raises* proves nothing: the telemetry thread logs "skipped" and
the scan continues. This guard therefore RECORDS every denied event, and the run
fails on the record, not on the exception.

Layers, outermost first:

  1. closed at the source   the profile disables each known egress path
  2. sanitized environment  the worker inherits an allowlist, never a secret
  3. audit hook (primary)   sys.addaudithook: C-level, not removable, sees every
                            socket CPython creates, DNS, and process creation
  4. socket patch           readable errors, and a second net under layer 3
  5. proof                  canaries at start; accounting at the end

The audit hook cannot be uninstalled. Never install it in a process that must
use the network or spawn children afterwards — in particular never in pytest.

PRE-ARM WARM-UPS
----------------
Two well-known, local-only probes would otherwise be denied and counted, although
neither sends anything anywhere. Both are resolved once, immediately BEFORE the
guard is armed, so their results are cached and they never run under the guard:

  * platform.uname()   on Windows the standard library runs `cmd /c ver` to read
                       the OS version; on Linux `uname -p` for the processor.
  * urllib3's import   urllib3 creates one IPv6 socket and binds it to ::1 at
                       import time, to learn whether IPv6 exists. No packet leaves.

Nothing else is exempt. After arming, creating any IPv4/IPv6 socket and starting
any process is denied and counted. The warm-ups performed are listed in every
run's guard summary.

Standard library only.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

try:                                    # the C module behind `socket`
    import _socket
except ImportError:                     # pragma: no cover - CPython always has it
    _socket = None


class GuardViolation(OSError):
    """Raised inside the guarded process when an operation is denied."""


# Audit events that are always denied (subject to the two exceptions handled in
# the hook: local-family sockets, and allow-listed executables in the orchestrator).
_DENY_EVENTS = frozenset({
    "socket.__new__", "socket.connect", "socket.sendto", "socket.sendmsg",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
    "socket.getnameinfo",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
    "poplib.connect", "imaplib.open", "nntplib.connect", "telnetlib.Telnet.open",
    "webbrowser.open",
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.startfile", "os.fork", "os.forkpty",
})
_PROCESS_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.startfile", "os.fork", "os.forkpty",
})

# Reserved, never-routable canary targets (RFC 2606 `.invalid`, RFC 5737 TEST-NET-1).
CANARY_HOST = "fie-eval-canary.invalid"
CANARY_ADDR = ("192.0.2.1", 9)
CANARY_PROGRAM = "fie-eval-canary-not-a-program"


class _State:
    def __init__(self) -> None:
        self.installed = False
        self.mode = ""
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.audit_total = 0
        self.canary_depth = 0
        self.log_fd: int | None = None
        self.allow_exec: frozenset[str] = frozenset()
        self.local_families: frozenset[int] = frozenset()
        self.pickle_classes: set[tuple[str, str]] = set()
        self.selftest: dict | None = None
        self.prearm: list[str] = []


_STATE = _State()


# ── recording ────────────────────────────────────────────────────────────────

def _short(value, limit: int = 200) -> str:
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = "<unprintable>"
    return text[:limit]


def _deny(event: str, target: str) -> None:
    """Record a denied operation, then raise. Recording is what fails the run."""
    st = _STATE
    with st.lock:
        rec = {
            "seq": len(st.events),
            "event": event,
            "target": _short(target),
            "thread": threading.current_thread().name,
            "canary": st.canary_depth > 0,
        }
        st.events.append(rec)
        if st.log_fd is not None:
            try:
                os.write(st.log_fd, (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
            except OSError:
                pass
    raise GuardViolation(f"blocked by the evaluation guard: {event} {_short(target)}")


def _norm_exe(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _describe(event: str, args: tuple) -> str:
    if event in ("socket.connect", "socket.sendto"):
        return _short(args[1] if len(args) > 1 else args)
    if event == "socket.getaddrinfo":
        return _short(args[:2])
    if event == "http.client.connect":
        return _short(args[1:3])
    if event == "urllib.Request":
        return _short(args[0] if args else "")
    return _short(args)


def _audit(event: str, args: tuple) -> None:
    st = _STATE
    st.audit_total += 1
    if event == "pickle.find_class":
        try:
            st.pickle_classes.add((str(args[0]), str(args[1])))
        except Exception:
            pass
        return
    if event not in _DENY_EVENTS:
        return

    if event == "socket.__new__":
        try:
            family = int(args[1])
        except Exception:
            family = -1
        if family in st.local_families:
            return
        _deny(event, f"address family {family}")

    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        try:
            if int(args[0].family) in st.local_families:
                return
        except Exception:
            pass

    if event == "subprocess.Popen" and st.allow_exec:
        exe = args[0] if args else None
        try:
            if exe is not None and _norm_exe(os.fspath(exe)) in st.allow_exec:
                return
        except Exception:
            pass

    _deny(event, _describe(event, args))


# ── installation ─────────────────────────────────────────────────────────────

def _patch_sockets() -> None:
    """Secondary layer: replace the Python-level entry points with deniers."""
    def method_denier(name: str):
        def blocked(self, *args, **kwargs):
            _deny(f"patched:socket.{name}", _short(args[0] if args else ""))
        blocked.__name__ = name
        return blocked

    def function_denier(name: str):
        def blocked(*args, **kwargs):
            _deny(f"patched:{name}", _short(args[:2]))
        blocked.__name__ = name
        return blocked

    for name in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, name, method_denier(name))
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        setattr(socket, name, function_denier(name))


def _prearm(names: tuple[str, ...]) -> list[str]:
    """Run the fixed, local-only warm-ups described in the module docstring."""
    done: list[str] = []
    if "platform" in names:
        import platform
        try:
            uname = platform.uname()
            _ = uname.processor            # lazily computed; may run `uname -p`
            platform.platform()
            done.append("platform.uname")
        except Exception:
            pass
    if "urllib3" in names:
        try:
            import urllib3.util.connection  # noqa: F401  (import-time IPv6 probe)
            done.append("urllib3.ipv6_probe")
        except Exception:
            pass                            # not installed: nothing to warm up
    return done


def install(mode: str = "worker", event_log: str | os.PathLike | None = None,
            allow_exec: tuple[str, ...] = (),
            prearm: tuple[str, ...] | None = None) -> None:
    """
    Arm the guard for the rest of this process. Idempotent.

    mode        "worker" denies all process creation. "orchestrator" permits
                starting the executables in `allow_exec` (the interpreter and git),
                matched by resolved path.
    event_log   file that receives one JSON line per denied event. Opened before
                the hook is armed, so logging itself needs no further file open.
    prearm      which pre-arm warm-ups to run. Default: both for a worker,
                `platform` only for the orchestrator (it never imports urllib3).
    """
    st = _STATE
    if st.installed:
        return
    if event_log is not None:
        Path(event_log).parent.mkdir(parents=True, exist_ok=True)
        st.log_fd = os.open(os.fspath(event_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    st.mode = mode
    st.allow_exec = frozenset(_norm_exe(p) for p in allow_exec) if mode == "orchestrator" else frozenset()
    local = set()
    if hasattr(socket, "AF_UNIX"):
        local.add(int(socket.AF_UNIX))
    st.local_families = frozenset(local)
    if prearm is None:
        prearm = ("platform",) if mode == "orchestrator" else ("platform", "urllib3")
    st.prearm = _prearm(tuple(prearm))
    _patch_sockets()
    sys.addaudithook(_audit)
    st.installed = True


def installed() -> bool:
    return _STATE.installed


@contextmanager
def _canary():
    _STATE.canary_depth += 1
    try:
        yield
    finally:
        _STATE.canary_depth -= 1


def _expect_denied(fn) -> bool:
    """True when `fn` raised GuardViolation and exactly that attempt was recorded."""
    before = len(_STATE.events)
    try:
        result = fn()
    except GuardViolation:
        return len(_STATE.events) > before
    except BaseException:
        return False
    close = getattr(result, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    return False


def selftest() -> dict:
    """
    Prove the guard is live, before anything under test is imported.

    Each canary must be denied AND recorded. Nothing leaves the machine: socket
    creation and name resolution are refused before any packet exists. Canaries
    stop at the first failure so a broken guard cannot reach the next probe.
    """
    if not _STATE.installed:
        raise RuntimeError("guard is not installed")
    import subprocess

    checks: dict[str, bool] = {}
    probes = [
        ("ipv4_socket_audit", lambda: _socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        ("dns_audit", lambda: _socket.getaddrinfo(CANARY_HOST, 443)),
        ("dns_patched", lambda: socket.getaddrinfo(CANARY_HOST, 443)),
        ("connect_patched", lambda: socket.create_connection(CANARY_ADDR, timeout=1)),
        ("subprocess", lambda: subprocess.Popen([CANARY_PROGRAM])),
    ]
    with _canary():
        for name, probe in probes:
            checks[name] = _expect_denied(probe)
            if not checks[name]:
                break
    result = {"passed": len(checks) == len(probes) and all(checks.values()), "checks": checks}
    _STATE.selftest = result
    return result


# ── accounting ───────────────────────────────────────────────────────────────

def events() -> list[dict]:
    with _STATE.lock:
        return list(_STATE.events)


def violations() -> list[dict]:
    """Denied events that were NOT canaries. Any entry here invalidates the run."""
    return [e for e in events() if not e["canary"]]


def violation_count() -> int:
    return len(violations())


def observed_pickle_classes() -> list[list[str]]:
    """(module, name) pairs resolved by any unpickling since the guard was armed."""
    return sorted([m, n] for m, n in _STATE.pickle_classes)


def summary() -> dict:
    ev = events()
    return {
        "installed": _STATE.installed,
        "mode": _STATE.mode,
        "selftest": _STATE.selftest,
        "prearm": list(_STATE.prearm),
        "violations": len([e for e in ev if not e["canary"]]),
        "canary_events": len([e for e in ev if e["canary"]]),
        "audit_events_seen": _STATE.audit_total,
        "violation_events": [e for e in ev if not e["canary"]][:50],
    }


# ── import blocking ──────────────────────────────────────────────────────────

class ImportBlocker(importlib.abc.MetaPathFinder):
    """Makes the named top-level packages unimportable in this process."""

    def __init__(self, names) -> None:
        self.names = frozenset(names)
        self.hits: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.names:
            self.hits.add(fullname)
            raise ImportError(f"'{fullname}' is unimportable in this evaluation profile")
        return None


def block_imports(names) -> ImportBlocker:
    """
    Install an ImportBlocker. Refuses if a named package is already loaded: a
    profile that says a package is absent must never run with it present.
    """
    names = list(names)
    loaded = sorted(n for n in names if n in sys.modules)
    if loaded:
        raise RuntimeError(f"cannot block already-imported packages: {loaded}")
    blocker = ImportBlocker(names)
    sys.meta_path.insert(0, blocker)
    return blocker


# ── environment allowlist ────────────────────────────────────────────────────

# Inherited from the caller. Everything else — including every credential and
# endpoint in the developer's shell — is dropped.
ENV_PASSTHROUGH = (
    "PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS",
    "LD_LIBRARY_PATH", "LANG", "LC_ALL",
)

# An unroutable local address: a native HTTP client that honours proxy variables
# fails closed instead of reaching the network.
DEAD_PROXY = "http://127.0.0.1:9"


def sanitized_env(parent: dict, profile_env: dict, scratch: str | os.PathLike,
                  hash_seed: str = "0") -> dict:
    """
    Build the worker's environment from an allowlist.

    `profile_env` holds the profile's own settings. Temp and home directories are
    redirected into `scratch`, so nothing is read from or written to the user's
    real home (the product otherwise appends to ~/.fie/flagged_events.jsonl).
    """
    scratch = Path(scratch)
    home = scratch / "home"
    tmp = scratch / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Windows environment names are case-insensitive: never emit two spellings
    # of one variable, or the child sees whichever the OS happens to keep.
    windows = os.name == "nt"
    env: dict[str, str] = {}
    seen: set[str] = set()
    for key in ENV_PASSTHROUGH:
        folded = key.upper() if windows else key
        if key in parent and folded not in seen:
            seen.add(folded)
            env[folded if windows else key] = parent[key]
    proxies = {"HTTP_PROXY": DEAD_PROXY, "HTTPS_PROXY": DEAD_PROXY, "ALL_PROXY": DEAD_PROXY}
    if not windows:
        proxies.update({k.lower(): v for k, v in list(proxies.items())})
    env.update(proxies)
    env.update({
        "PYTHONHASHSEED": str(hash_seed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8",
        "TEMP": str(tmp), "TMP": str(tmp), "TMPDIR": str(tmp),
        "HOME": str(home), "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
    })
    for key, value in profile_env.items():
        env[key] = str(value)
    return env
