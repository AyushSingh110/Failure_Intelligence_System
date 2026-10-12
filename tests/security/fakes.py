"""
In-repository stand-ins used by the tenant-isolation suite.

FakeCollection implements exactly the MongoDB operations the server calls, and
nothing else. It exists so the suite never needs a database, a new dependency,
or the network. If the server starts using an operation that is missing here,
the test fails with AttributeError, which is the intended signal to extend it.
"""
from __future__ import annotations

import copy
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

# ── Planted identities ────────────────────────────────────────────────────────
# Every value is unique so that a leak across tenants is detectable by a plain
# substring search over response bodies and security events.

TENANT_A = {
    "email":       "alpha.owner@tenant-a.test",
    "name":        "Alpha Owner",
    "picture":     "",
    "api_key":     "fie-aaaa1111aaaa1111",
    "tenant_id":   "alphaowner-a1a1a1",
    "is_admin":    False,
    "plan":        "free",
    "calls_used":  0,
    "calls_limit": 100000,
}
TENANT_B = {
    "email":       "bravo.owner@tenant-b.test",
    "name":        "Bravo Owner",
    "picture":     "",
    "api_key":     "fie-bbbb2222bbbb2222",
    "tenant_id":   "bravoowner-b2b2b2",
    "is_admin":    False,
    "plan":        "free",
    "calls_used":  0,
    "calls_limit": 100000,
}
PLATFORM_ADMIN = {
    "email":       "platform-admin@example.test",
    "name":        "Platform Admin",
    "picture":     "",
    "api_key":     "fie-cccc3333cccc3333",
    "tenant_id":   "platformad-c3c3c3",
    "is_admin":    True,
    "plan":        "admin",
    "calls_used":  0,
    "calls_limit": 999999,
}

TEST_JWT_SECRET = "security-suite-signing-secret-0123456789abcdef"
DEV_CONSTANT_SECRET = "fie-insecure-dev-secret-DO-NOT-USE-IN-PROD"

# Text markers planted in prompts and answers.
MARK = {
    "a": {"prompt": "ZQ-ALPHA-PROMPT-7f3a", "answer": "ZQ-ALPHA-ANSWER-91c2", "fix": "ZQ-ALPHA-CORRECTION-55d0"},
    "b": {"prompt": "ZQ-BRAVO-PROMPT-2e8b", "answer": "ZQ-BRAVO-ANSWER-64f7", "fix": "ZQ-BRAVO-CORRECTION-0a3e"},
}


def private_markers(user: dict, key: str) -> list[str]:
    """Every string that must never reach a different tenant."""
    return [user["email"], user["api_key"], user["tenant_id"], *MARK[key].values()]


FOREIGN_MARKERS = {
    "a":    private_markers(TENANT_B, "b") + [PLATFORM_ADMIN["email"], PLATFORM_ADMIN["api_key"]],
    "b":    private_markers(TENANT_A, "a") + [PLATFORM_ADMIN["email"], PLATFORM_ADMIN["api_key"]],
    "anon": private_markers(TENANT_A, "a") + private_markers(TENANT_B, "b")
            + [PLATFORM_ADMIN["email"], PLATFORM_ADMIN["api_key"]],
}
ALL_CREDENTIALS = [TENANT_A["api_key"], TENANT_B["api_key"], PLATFORM_ADMIN["api_key"], TEST_JWT_SECRET]
ALL_EMAILS = [TENANT_A["email"], TENANT_B["email"], PLATFORM_ADMIN["email"]]


def make_token(user: dict, *, secret: str = TEST_JWT_SECRET, expires_in_s: int = 3600, **overrides) -> str:
    """
    A session token in the format issued before WP-002 (it still carries the
    `api_key` field), so the suite also proves that already-issued tokens keep
    verifying after the change. `overrides` forges individual claims.
    """
    import jwt

    payload = {
        "email":     user["email"],
        "name":      user["name"],
        "picture":   user.get("picture", ""),
        "tenant_id": user["tenant_id"],
        "api_key":   user["api_key"],
        "is_admin":  user.get("is_admin", False),
        "plan":      user.get("plan", "free"),
        "exp":       datetime.now(timezone.utc) + timedelta(seconds=expires_in_s),
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, secret, algorithm="HS256")


def key_headers(user: dict) -> dict:
    return {"X-API-Key": user["api_key"]}


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Fake MongoDB ──────────────────────────────────────────────────────────────

_MISSING = object()


def _get_path(doc: dict, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _set_path(doc: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = doc
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _matches(doc: dict, flt: dict | None) -> bool:
    for key, cond in (flt or {}).items():
        value = _get_path(doc, key)
        present = value is not _MISSING
        actual = value if present else None
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, operand in cond.items():
                if op == "$ne":
                    if actual == operand:
                        return False
                elif op == "$in":
                    if actual not in operand:
                        return False
                elif op == "$exists":
                    if bool(operand) != present:
                        return False
                elif op in ("$gte", "$gt", "$lte", "$lt"):
                    if actual is None:
                        return False
                    try:
                        ok = {"$gte": actual >= operand, "$gt": actual > operand,
                              "$lte": actual <= operand, "$lt": actual < operand}[op]
                    except TypeError:
                        return False
                    if not ok:
                        return False
                else:
                    raise NotImplementedError(f"FakeCollection: operator {op}")
        elif actual != cond:
            return False
    return True


def _project(doc: dict, projection: dict | None) -> dict:
    out = copy.deepcopy(doc)
    if projection and projection.get("_id", 1) == 0:
        out.pop("_id", None)
    return out


class _Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeCursor:
    def __init__(self, docs: list[dict]):
        self._docs = docs

    def sort(self, key, direction=None):
        spec = key if isinstance(key, list) else [(key, direction if direction is not None else 1)]
        for field, order in reversed(spec):
            self._docs.sort(
                key=lambda d: (_get_path(d, field) is _MISSING or _get_path(d, field) is None,
                               "" if _get_path(d, field) in (_MISSING, None) else _get_path(d, field)),
                reverse=order < 0,
            )
        return self

    def skip(self, n: int):
        if n < 0:
            raise ValueError("skip must be >= 0")
        self._docs = self._docs[n:]
        return self

    def limit(self, n: int):
        if n:
            self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self.docs: list[dict] = []
        self.indexes: list[tuple[tuple, dict]] = []
        self.calls: Counter = Counter()
        self.fail_with: Exception | None = None
        self._lock = threading.RLock()

    def __bool__(self):
        # pymongo raises here, and code that writes `if collection:` only fails
        # against a real database. The fake must fail the same way.
        raise NotImplementedError(
            "Collection objects do not implement truth value testing or bool(). "
            "Please compare with None instead: collection is not None"
        )

    # -- helpers -------------------------------------------------------------
    def _enter(self, op: str) -> None:
        self.calls[op] += 1
        if self.fail_with is not None:
            raise self.fail_with

    def _check_unique(self, candidate: dict, ignore: dict | None = None) -> None:
        for other in self.docs:
            if other is ignore:
                continue
            if "_id" in candidate and other.get("_id") == candidate["_id"]:
                raise DuplicateKeyError(f"E11000 duplicate key error collection: {self.name} index: _id_")
        for keys, options in self.indexes:
            if not options.get("unique"):
                continue
            wanted = tuple(candidate.get(k) for k in keys)
            for other in self.docs:
                if other is ignore:
                    continue
                if tuple(other.get(k) for k in keys) == wanted:
                    raise DuplicateKeyError(
                        f"E11000 duplicate key error collection: {self.name} index: {'_'.join(keys)}"
                    )

    # -- index ---------------------------------------------------------------
    def create_index(self, keys, **options):
        with self._lock:
            self._enter("create_index")
            names = tuple(k if isinstance(k, str) else k[0] for k in (keys if isinstance(keys, list) else [keys]))
            for existing, existing_options in self.indexes:
                if existing == names and bool(existing_options.get("unique")) != bool(options.get("unique")):
                    from pymongo.errors import OperationFailure
                    raise OperationFailure("IndexOptionsConflict", code=85)
            if (names, options) not in self.indexes:
                self.indexes.append((names, options))
            return "_".join(names)

    def index_keys(self) -> list[tuple]:
        return [keys for keys, _ in self.indexes]

    # -- reads ---------------------------------------------------------------
    def find_one(self, flt=None, projection=None, sort=None):
        with self._lock:
            self._enter("find_one")
            docs = [d for d in self.docs if _matches(d, flt)]
            if sort:
                docs = list(FakeCursor(docs).sort(sort))
            return _project(docs[0], projection) if docs else None

    def find(self, flt=None, projection=None, sort=None, limit=0, skip=0):
        with self._lock:
            self._enter("find")
            cursor = FakeCursor([_project(d, projection) for d in self.docs if _matches(d, flt)])
            if sort:
                cursor.sort(sort)
            if skip:
                cursor.skip(skip)
            if limit:
                cursor.limit(limit)
            return cursor

    def count_documents(self, flt=None):
        with self._lock:
            self._enter("count_documents")
            return sum(1 for d in self.docs if _matches(d, flt))

    # -- writes --------------------------------------------------------------
    def insert_one(self, doc):
        with self._lock:
            self._enter("insert_one")
            new = copy.deepcopy(doc)
            new.setdefault("_id", uuid.uuid4().hex)
            self._check_unique(new)
            self.docs.append(new)
            doc.setdefault("_id", new["_id"])
            return _Result(inserted_id=new["_id"])

    def update_one(self, flt, update, upsert=False):
        with self._lock:
            self._enter("update_one")
            unknown = set(update) - {"$set", "$inc", "$setOnInsert"}
            if unknown:
                raise NotImplementedError(f"FakeCollection: update operators {unknown}")
            target = next((d for d in self.docs if _matches(d, flt)), None)
            if target is not None:
                changed = copy.deepcopy(target)
                for path, value in update.get("$set", {}).items():
                    _set_path(changed, path, copy.deepcopy(value))
                for path, value in update.get("$inc", {}).items():
                    current = _get_path(changed, path)
                    _set_path(changed, path, (0 if current is _MISSING else current) + value)
                if changed.get("_id") != target.get("_id"):
                    raise RuntimeError("FakeCollection: _id is immutable")
                self._check_unique(changed, ignore=target)
                modified = changed != target
                target.clear()
                target.update(changed)
                return _Result(matched_count=1, modified_count=int(modified), upserted_id=None)
            if not upsert:
                return _Result(matched_count=0, modified_count=0, upserted_id=None)
            new: dict = {}
            for key, cond in (flt or {}).items():
                if not (isinstance(cond, dict) and any(k.startswith("$") for k in cond)):
                    _set_path(new, key, copy.deepcopy(cond))
            for path, value in update.get("$setOnInsert", {}).items():
                _set_path(new, path, copy.deepcopy(value))
            for path, value in update.get("$set", {}).items():
                _set_path(new, path, copy.deepcopy(value))
            for path, value in update.get("$inc", {}).items():
                _set_path(new, path, value)
            new.setdefault("_id", uuid.uuid4().hex)
            self._check_unique(new)
            self.docs.append(new)
            return _Result(matched_count=0, modified_count=0, upserted_id=new["_id"])

    def replace_one(self, flt, doc, upsert=False):
        with self._lock:
            self._enter("replace_one")
            target = next((d for d in self.docs if _matches(d, flt)), None)
            new = copy.deepcopy(doc)
            if target is not None:
                new.setdefault("_id", target.get("_id"))
                self._check_unique(new, ignore=target)
                target.clear()
                target.update(new)
                return _Result(matched_count=1, modified_count=1, upserted_id=None)
            if not upsert:
                return _Result(matched_count=0, modified_count=0, upserted_id=None)
            new.setdefault("_id", uuid.uuid4().hex)
            self._check_unique(new)
            self.docs.append(new)
            return _Result(matched_count=0, modified_count=0, upserted_id=new["_id"])

    def delete_one(self, flt):
        with self._lock:
            self._enter("delete_one")
            for i, d in enumerate(self.docs):
                if _matches(d, flt):
                    del self.docs[i]
                    return _Result(deleted_count=1)
            return _Result(deleted_count=0)

    def delete_many(self, flt):
        with self._lock:
            self._enter("delete_many")
            keep = [d for d in self.docs if not _matches(d, flt)]
            removed = len(self.docs) - len(keep)
            self.docs[:] = keep
            return _Result(deleted_count=removed)


class FakeDatabase:
    """`db["name"]` creates collections on first use, as MongoDB does."""

    def __init__(self):
        self._collections: dict[str, FakeCollection] = {}
        self._lock = threading.Lock()
        self.ping_error: Exception | None = None

    def __bool__(self):
        raise NotImplementedError(
            "Database objects do not implement truth value testing or bool(). "
            "Please compare with None instead: database is not None"
        )

    def __getitem__(self, name: str) -> FakeCollection:
        with self._lock:
            if name not in self._collections:
                self._collections[name] = FakeCollection(name)
            return self._collections[name]

    def command(self, name, *args, **kwargs):
        if self.ping_error is not None:
            raise self.ping_error
        return {"ok": 1}

    def names(self) -> list[str]:
        return sorted(self._collections)

    def all_docs(self) -> list[tuple[str, dict]]:
        return [(n, d) for n, c in self._collections.items() for d in c.docs]


# ── Request helpers shared by the test modules ────────────────────────────────

def monitor_body(prompt: str, answer: str, **extra) -> dict:
    body = {
        "prompt":             prompt,
        "primary_output":     answer,
        "primary_model_name": "suite-model",
        "run_full_jury":      False,
        "latency_ms":         10.0,
    }
    body.update(extra)
    return body


def track_body(request_id: str, prompt: str, answer: str, **extra) -> dict:
    body = {
        "request_id":    request_id,
        "timestamp":     "2026-01-01T00:00:00",
        "model_name":    "suite-model",
        "model_version": "v1",
        "temperature":   0.0,
        "latency_ms":    1.0,
        "input_text":    prompt,
        "output_text":   answer,
    }
    body.update(extra)
    return body
