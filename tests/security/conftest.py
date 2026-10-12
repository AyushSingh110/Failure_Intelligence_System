"""
Fixtures for the tenant-isolation suite.

ISOLATION RULES (enforced here, not left to convention)
-------------------------------------------------------
1. The environment is scrubbed at import, before any application module can be
   imported: every key named in the repository's `.env` is blanked, so
   `load_dotenv()` cannot inject a real credential, and the database URI is
   empty. This affects the whole pytest process on purpose.
2. The suite refuses to start if the application settings still point at a
   database or carry a provider key.
3. Every test runs against a fresh in-memory FakeDatabase.
4. Any attempt to open a non-loopback connection fails the test.
5. Every response is recorded. A response that contains another tenant's planted
   marker fails the test at teardown, whatever the test itself asserted.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
from pathlib import Path

import pytest

from .fakes import (
    ALL_CREDENTIALS, ALL_EMAILS, FOREIGN_MARKERS, MARK, PLATFORM_ADMIN, TENANT_A, TENANT_B,
    TEST_JWT_SECRET, FakeDatabase, key_headers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRATCH = Path(tempfile.mkdtemp(prefix="fie-security-suite-"))


def _scrub_environment() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            key = line.split("=", 1)[0].strip()
            if key and not key.startswith("#") and "=" in line:
                os.environ[key] = ""
    # Typed settings cannot be empty strings; list-valued ones must be absent.
    for key in ("OLLAMA_MODELS", "GROQ_MODELS", "GROQ_FAST_MODEL", "CORS_ALLOWED_ORIGINS", "REDIS_URL",
                "SENTRY_DSN", "FIE_AUTO_RECALIBRATE", "FIE_AUTO_RETRAIN", "FIE_FAISS_AUTOGROW",
                "FIE_ALLOW_INSECURE_DEV_SECRET", "FIE_COLLECT_HARD_POSITIVES"):
        os.environ.pop(key, None)
    os.environ.update({
        "MONGODB_URI":           "",
        "MONGODB_DB_NAME":       "fie_test_security",
        "GROQ_API_KEY":          "",
        "GROQ_ENABLED":          "true",
        "OLLAMA_ENABLED":        "false",
        "SERPER_ENABLED":        "false",
        "SERPER_API_KEY":        "",
        "SENDGRID_API_KEY":      "",
        "NOTIFICATION_EMAIL":    "",
        "GOOGLE_CLIENT_ID":      "suite-client-id",
        "GOOGLE_CLIENT_SECRET":  "suite-client-secret",
        "JWT_SECRET_KEY":        TEST_JWT_SECRET,
        "JWT_ALGORITHM":         "HS256",
        "JWT_EXPIRE_HOURS":      "24",
        "ADMIN_EMAIL":           PLATFORM_ADMIN["email"],
        "FIE_API_KEY":           "",
        "FIE_NO_TELEMETRY":      "1",
        "FIE_NO_AUTO_DOWNLOAD":  "1",
        "FIE_FEEDBACK_PATH":     str(_SCRATCH / "flagged_events.jsonl"),
        "FIE_DATA_DIR":          str(_SCRATCH / "data"),
        "DEBUG":                 "false",
    })


_scrub_environment()

NETWORK_ATTEMPTS: list[str] = []
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo


def _host_of(address) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


def _guarded_connect(self, address):
    if _host_of(address) in _LOOPBACK:
        return _REAL_CONNECT(self, address)
    NETWORK_ATTEMPTS.append(f"connect {address!r}")
    raise ConnectionRefusedError("security suite: outbound connections are blocked")


def _guarded_connect_ex(self, address):
    if _host_of(address) in _LOOPBACK:
        return _REAL_CONNECT_EX(self, address)
    NETWORK_ATTEMPTS.append(f"connect_ex {address!r}")
    return 10061


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host is None or str(host) in _LOOPBACK:
        return _REAL_GETADDRINFO(host, *args, **kwargs)
    NETWORK_ATTEMPTS.append(f"getaddrinfo {host!r}")
    raise socket.gaierror("security suite: name resolution is blocked")


@pytest.fixture(scope="session", autouse=True)
def _refuse_real_targets():
    """Abort rather than run against anything real."""
    from config import get_settings

    settings = get_settings()
    problems = []
    if settings.mongodb_uri:
        problems.append("settings.mongodb_uri is set")
    if settings.groq_api_key or settings.serper_api_key or settings.sendgrid_api_key:
        problems.append("a provider key is set")
    import storage.database as database
    if database._client is not None:
        problems.append("a MongoDB client is already connected")
    if problems:
        pytest.exit(
            "tests/security refuses to run: " + "; ".join(problems)
            + ". Run this suite in its own pytest process.", returncode=4,
        )
    yield


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    NETWORK_ATTEMPTS.clear()
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    yield
    attempts = list(NETWORK_ATTEMPTS)
    NETWORK_ATTEMPTS.clear()
    assert not attempts, f"outbound network attempted during the test: {attempts}"


def fake_embedding(text: str) -> list[float]:
    """Deterministic bag-of-words vector: same words in any order → identical vector."""
    import hashlib
    import math
    import re

    vec = [0.0] * 32
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 32] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def fakedb(monkeypatch):
    """Fresh fake database and clean process-global state for every test."""
    import app.auth as auth
    import app.notifications as notifications
    import engine.archetypes.clustering as clustering
    import engine.evolution.tracker as tracker
    import engine.fie_config as fie_config
    import engine.ground_truth_cache as gt_cache
    import engine.groq_service as groq_service
    import engine.model_extraction_tracker as extraction
    import engine.retraining.buffer as buffer
    import engine.session_store as session_store
    import storage.database as database

    db = FakeDatabase()
    for user in (TENANT_A, TENANT_B, PLATFORM_ADMIN):
        db["users"].insert_one(dict(user))
    # The deployed database has a unique index on request_id alone. Tests that
    # model the state after the owner's index migration remove it explicitly.
    db["inferences"].create_index("request_id", unique=True)

    monkeypatch.setattr(database, "_db", db)
    monkeypatch.setattr(database, "_collection", db["inferences"])
    monkeypatch.setattr(database, "_client", None)
    monkeypatch.setattr(database, "_fallback_mode", False)
    database._fallback_records.clear()
    monkeypatch.setattr(auth, "_get_users_collection", lambda: db["users"])
    monkeypatch.setattr(session_store, "_get_collection", lambda: db["session_context"])
    monkeypatch.setattr(buffer, "_get_buffer_collection", lambda: db["retraining_buffer"])
    monkeypatch.setattr(gt_cache, "_embed_question", fake_embedding)

    # Process-global state, reset in place so already-imported references see it.
    groq_service._response_cache.clear()
    clustering.archetype_registry._clusters.clear()
    tracker.evolution_tracker.__init__()
    session_store._fallback.clear()
    session_store._fallback_summaries.clear()
    extraction._memory_store.clear()
    notifications._spike_last_sent.clear()
    monkeypatch.setattr(fie_config, "_thresholds", dict(fie_config._DEFAULTS))
    monkeypatch.setattr(fie_config, "_attack_thresholds", {})
    monkeypatch.setattr(fie_config, "_config_version", "default")
    monkeypatch.setattr(fie_config, "_feedback_count_at_last_calib", 0)
    monkeypatch.setattr(fie_config, "_preflight_block_enabled", True)
    try:
        from app.tenancy import tenant_analytics
        tenant_analytics.clear()
    except ImportError:
        pass
    try:
        from app.limiter import limiter
        if limiter is not None:
            limiter.reset()
    except Exception:
        pass
    return db


@pytest.fixture(autouse=True)
def _quiet_pipeline(monkeypatch):
    """Keep `/monitor` local and fast: no model load, no translation call, no encoder."""
    import fie.preflight as preflight

    monkeypatch.setattr(
        preflight, "preflight_check",
        lambda prompt, session_id=None, domain=None: preflight.GuardResult(
            blocked=False, attack_type="", confidence=0.0, layers_fired=[], refusal_message="",
        ),
    )
    distance = lambda primary, secondary: {"embedding_distance": 0.0}  # noqa: E731
    monkeypatch.setattr("engine.detector.embedding.compute_embedding_distance", distance)
    monkeypatch.setattr("app.routes.inference.compute_embedding_distance", distance, raising=False)
    monkeypatch.setattr("engine.agents.failure_agent.compute_embedding_distance", distance, raising=False)


RESPONSE_LOG: list[tuple[str, str, str, int, str]] = []


def squash(text: str) -> str:
    """Lower-case and drop punctuation, so a normalised or re-cased copy of a marker still matches."""
    import re

    return re.sub(r"[^a-z0-9]", "", text.lower())


class Actor:
    """A caller with a fixed credential. Every response it receives is recorded."""

    def __init__(self, client, name: str, headers: dict):
        self._client = client
        self.name = name
        self.headers = dict(headers)

    def request(self, method: str, url: str, **kwargs):
        headers = {**self.headers, **kwargs.pop("headers", {})}
        response = self._client.request(method, url, headers=headers, **kwargs)
        RESPONSE_LOG.append((self.name, method, url, response.status_code, response.text))
        return response

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def with_headers(self, headers: dict) -> "Actor":
        return Actor(self._client, self.name, headers)


@pytest.fixture
def client(fakedb):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_foreign_markers():
    """Hard failure if any tenant or anonymous caller was sent another tenant's marker."""
    RESPONSE_LOG.clear()
    yield
    leaks = []
    for name, method, url, status, text in RESPONSE_LOG:
        flat = squash(text)
        for marker in FOREIGN_MARKERS.get(name, []):
            if squash(marker) in flat:
                leaks.append(f"{name} {method} {url} [{status}] received {marker!r}")
    RESPONSE_LOG.clear()
    assert not leaks, "cross-tenant data in a response:\n  " + "\n  ".join(leaks)


@pytest.fixture
def a(client):
    return Actor(client, "a", key_headers(TENANT_A))


@pytest.fixture
def b(client):
    return Actor(client, "b", key_headers(TENANT_B))


@pytest.fixture
def admin(client):
    return Actor(client, "admin", key_headers(PLATFORM_ADMIN))


@pytest.fixture
def anon(client):
    return Actor(client, "anon", {})


class _EventCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.raw: list[str] = []

    def emit(self, record):
        self.raw.append(record.getMessage())

    @property
    def events(self) -> list[dict]:
        out = []
        for line in self.raw:
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"event": "<unparseable>", "raw": line})
        return out

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


@pytest.fixture(autouse=True)
def events():
    """Security events emitted during the test. None may contain a secret or user text."""
    collector = _EventCollector()
    logger = logging.getLogger("fie.security")
    previous = logger.level
    logger.addHandler(collector)
    logger.setLevel(logging.DEBUG)
    yield collector
    logger.removeHandler(collector)
    logger.setLevel(previous)
    forbidden = (ALL_CREDENTIALS + ALL_EMAILS + list(MARK["a"].values()) + list(MARK["b"].values())
                 + [TENANT_A["tenant_id"], TENANT_B["tenant_id"], PLATFORM_ADMIN["tenant_id"]])
    bad = [f"{needle!r} in {line}" for line in collector.raw for needle in forbidden if needle in line]
    assert not bad, "a security event carries a secret, an identity or user text:\n  " + "\n  ".join(bad)


def inference_ids(actor) -> list[str]:
    response = actor.get("/api/v1/inferences")
    assert response.status_code == 200, response.text
    return [row["request_id"] for row in response.json()]


def scope_for(user: dict):
    """A TenantScope for `user`, built the only way the application allows."""
    from app.auth_guard import Principal
    from app.tenancy import TenantScope

    return TenantScope(Principal(
        tenant_id=user["tenant_id"], subject=user["email"],
        role="platform_admin" if user.get("is_admin") else "tenant",
        credential_kind="api_key", admin_verified=bool(user.get("is_admin")),
    ))
