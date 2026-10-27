"""
Fixtures for the output-reliability suite (Workstream B, D-060).

Every test in this directory runs with outbound networking refused and recorded:
a non-loopback connect or name lookup fails the test at teardown, whatever the
test asserted. The 8B judge, Serper, Wikidata, Wikipedia, the cache and the
shadow fan-out are all replaced at their leaves (harness.py), so a recorded
attempt means a leaf was missed.

The evals/hermetic.py guard is not used here: it installs an audit hook that
cannot be removed from the pytest process. research/reliability/e_r1_part_a/
run_part_a.py runs the same fixtures under that guard in a child interpreter.
"""
from __future__ import annotations

import os
import socket

import pytest

# Set before `fie` is first imported by this suite: no telemetry ping, no model download.
os.environ.setdefault("FIE_NO_TELEMETRY", "1")
os.environ.setdefault("FIE_NO_AUTO_DOWNLOAD", "1")

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
ATTEMPTS: list[str] = []
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo


def _host(address) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


def _connect(self, address):
    if _host(address) in _LOOPBACK:
        return _REAL_CONNECT(self, address)
    ATTEMPTS.append(f"connect {address!r}")
    raise ConnectionRefusedError("reliability suite: outbound connections are blocked")


def _connect_ex(self, address):
    if _host(address) in _LOOPBACK:
        return _REAL_CONNECT_EX(self, address)
    ATTEMPTS.append(f"connect_ex {address!r}")
    return 10061


def _getaddrinfo(host, *args, **kwargs):
    if host is None or str(host) in _LOOPBACK:
        return _REAL_GETADDRINFO(host, *args, **kwargs)
    ATTEMPTS.append(f"getaddrinfo {host!r}")
    raise socket.gaierror("reliability suite: name resolution is blocked")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    ATTEMPTS.clear()
    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    yield
    attempts = list(ATTEMPTS)
    ATTEMPTS.clear()
    assert not attempts, f"outbound network attempted during the test: {attempts}"
