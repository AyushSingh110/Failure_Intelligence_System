"""The fingerprint: each field moves exactly the key it should, and secrets never appear."""
from __future__ import annotations

import copy
import json

import pytest

from evals import canonical, fingerprint
from _helpers import REPO_ROOT


def _identity() -> dict:
    return {
        "subject": {"fie_tree_sha256": "a" * 64, "fie_py_files": 35, "fie_version": "1.18.0"},
        "git": {"commit": "c" * 40, "dirty_subject": False, "dirty_other": True},
        "models": {"p": {"pair_classifier": {"file": "m.pkl", "sha256": "b" * 64, "size": 1, "in_manifest": True}}},
        "datasets": {"d": {"content_sha256": "d" * 64, "rows": 10, "unique": 10, "label": "attack"}},
        "suites": {"std.d": "e" * 64},
        "fixtures": {"padding_filler": "f" * 64},
        "harness": {"version": "1.0.0", "schema_version": 1, "tree_sha256": "9" * 64},
    }


def _configuration() -> dict:
    return {"p": {"profile": {"id": "p"}, "langdetect_seed": 0, "translation": "unavailable",
                  "env": {"PYTHONHASHSEED": "0"}, "subject": {"pair": {"threshold": 0.5}}}}


def _environment() -> dict:
    return {"python": {"version": "3.10.19", "implementation": "CPython"},
            "platform": {"system": "Windows", "release": "10", "machine": "AMD64"},
            "cpu": {"model": "x", "count": 8},
            "packages": {"numpy": "2.2.6", "scikit-learn": "1.7.2", "onnxruntime": "1.23.2",
                         "tokenizers": "0.22.2", "xgboost": "3.2.0", "requests": "2.32.5"},
            "threads": {}, "machine_tag": "abcd1234"}


def _keys(identity=None, configuration=None, environment=None) -> dict:
    return fingerprint.build(identity or _identity(), configuration or _configuration(),
                             environment or _environment())["keys"]


def _changed(base: dict, other: dict) -> set:
    return {k for k in base if base[k] != other[k]}


def test_fingerprint_has_three_blocks_and_four_keys():
    fp = fingerprint.build(_identity(), _configuration(), _environment())
    assert set(fp) == {"schema_version", "keys", "identity", "configuration", "environment"}
    assert set(fp["keys"]) == {"dataset_key", "config_key", "subject_key", "env_key"}
    assert all(len(v) == 64 for v in fp["keys"].values())
    assert fingerprint.build(_identity(), _configuration(), _environment()) == fp


@pytest.mark.parametrize("mutate,expected", [
    (lambda i, c, e: i["datasets"]["d"].update(content_sha256="0" * 64), {"dataset_key"}),
    (lambda i, c, e: i["datasets"]["d"].update(rows=11), {"dataset_key"}),
    (lambda i, c, e: i["suites"].update({"std.d": "1" * 64}), {"dataset_key"}),
    (lambda i, c, e: i["fixtures"].update(padding_filler="2" * 64), {"dataset_key"}),
    (lambda i, c, e: i["subject"].update(fie_tree_sha256="3" * 64), {"subject_key"}),
    (lambda i, c, e: i["models"]["p"]["pair_classifier"].update(sha256="4" * 64), {"subject_key"}),
    (lambda i, c, e: c["p"].update(langdetect_seed=None), {"config_key"}),
    (lambda i, c, e: c["p"]["env"].update(PYTHONHASHSEED="random"), {"config_key"}),
    (lambda i, c, e: c["p"]["subject"]["pair"].update(threshold=0.45), {"config_key"}),
    (lambda i, c, e: c["p"].update(translation="fixed benign sentence"), {"config_key"}),
    (lambda i, c, e: e["packages"].update(numpy="2.1.3"), {"env_key"}),
    (lambda i, c, e: e["packages"].update(onnxruntime="1.24.0"), {"env_key"}),
    (lambda i, c, e: e["python"].update(version="3.11.9"), {"env_key"}),
    (lambda i, c, e: e["platform"].update(system="Linux"), {"env_key"}),
])
def test_each_field_changes_exactly_one_key(mutate, expected):
    i, c, e = _identity(), _configuration(), _environment()
    mutate(i, c, e)
    assert _changed(_keys(), _keys(i, c, e)) == expected


@pytest.mark.parametrize("mutate", [
    lambda i, c, e: e["python"].update(version="3.10.4"),          # patch release
    lambda i, c, e: e["packages"].update(requests="2.31.0"),       # not a numeric package
    lambda i, c, e: e.update(machine_tag="ffffffff"),              # another machine, same class
    lambda i, c, e: e["cpu"].update(count=4),
    lambda i, c, e: i["git"].update(commit="d" * 40),              # the same tree at another commit
    lambda i, c, e: i["harness"].update(tree_sha256="8" * 64),
])
def test_fields_that_do_not_affect_comparability_leave_every_key_alone(mutate):
    i, c, e = _identity(), _configuration(), _environment()
    mutate(i, c, e)
    assert _changed(_keys(), _keys(i, c, e)) == set()


def test_comparability_verdicts():
    base = _keys()
    def other(**changed):
        k = dict(base); k.update(changed); return k
    assert fingerprint.comparability(base, base)["status"] == "identical inputs"
    assert fingerprint.comparability(base, other(subject_key="x"))["status"] == "comparable"
    assert fingerprint.comparability(base, other(env_key="x"))["status"] == "comparable with a warning"
    assert "configuration change" in fingerprint.comparability(base, other(config_key="x"))["status"]
    assert fingerprint.comparability(base, other(dataset_key="x"))["status"] == "incomparable"
    assert fingerprint.comparability(base, other(dataset_key="x", subject_key="y"))["status"] == "incomparable"


# ── source-tree hash ─────────────────────────────────────────────────────────

def _tree(tmp_path, files: dict):
    for rel, data in files.items():
        path = tmp_path / "pkg" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return fingerprint.tree_sha256(tmp_path, "pkg")


def test_tree_hash_ignores_line_endings_and_bytecode_but_not_content(tmp_path):
    base, count = _tree(tmp_path / "a", {"a.py": b"x = 1\ny = 2\n", "sub/b.py": b"z = 3\n"})
    assert count == 2
    crlf, _ = _tree(tmp_path / "b", {"a.py": b"x = 1\r\ny = 2\r\n", "sub/b.py": b"z = 3\r\n"})
    assert crlf == base
    noise, n2 = _tree(tmp_path / "c", {"a.py": b"x = 1\ny = 2\n", "sub/b.py": b"z = 3\n",
                                       "__pycache__/a.cpython-310.pyc": b"\x00\x01", "notes.txt": b"hi"})
    assert noise == base and n2 == 2
    edited, _ = _tree(tmp_path / "d", {"a.py": b"x = 1\ny = 3\n", "sub/b.py": b"z = 3\n"})
    assert edited != base
    renamed, _ = _tree(tmp_path / "e", {"a.py": b"x = 1\ny = 2\n", "sub/c.py": b"z = 3\n"})
    assert renamed != base
    added, _ = _tree(tmp_path / "f", {"a.py": b"x = 1\ny = 2\n", "sub/b.py": b"z = 3\n", "new.py": b""})
    assert added != base


def test_real_fie_tree_hash_is_stable_and_covers_every_module():
    first = fingerprint.subject_identity(REPO_ROOT)
    assert first == fingerprint.subject_identity(REPO_ROOT)
    assert first["fie_py_files"] == len([p for p in (REPO_ROOT / "fie").rglob("*.py")
                                         if "__pycache__" not in p.parts])
    assert first["fie_version"]


# ── secrets ──────────────────────────────────────────────────────────────────

def test_secret_variables_are_recorded_by_presence_only(monkeypatch):
    planted = {"GROQ_API_KEY": "gsk_PLANTED_ONE", "MONGODB_URI": "mongodb+srv://PLANTED_TWO",
               "JWT_SECRET_KEY": "PLANTED_THREE", "FIE_API_KEY": "fie-PLANTED_FOUR",
               "HUGGING_FACE_TOKEN": "hf_PLANTED_FIVE", "REDIS_URL": "redis://PLANTED_SIX"}
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FIE_PAIR_VERSION", "v6_3b")
    profile = {"title": "t", "pair_version": "v6_3b", "blocked_imports": ["engine"]}
    block = fingerprint.profile_configuration("p", profile, {"x": 1}, "unavailable", 0)
    text = json.dumps(block)
    assert "PLANTED" not in text
    assert block["env_present"]["GROQ_API_KEY"] is True and block["env_present"]["MONGODB_URI"] is True
    assert block["env"]["FIE_PAIR_VERSION"] == "v6_3b"
    assert not set(fingerprint.ENV_VALUE_NAMES) & set(fingerprint.ENV_PRESENCE_NAMES)
    for name in fingerprint.ENV_VALUE_NAMES:
        assert not any(word in name for word in ("KEY", "TOKEN", "SECRET", "URI", "URL", "PASSWORD")), name


def test_environment_block_stores_no_host_name_or_path():
    import platform
    block = fingerprint.environment_block()
    text = json.dumps(block)
    node = platform.node()
    assert not node or node not in text
    assert len(block["machine_tag"]) == 8
    assert str(REPO_ROOT) not in text
    assert block["packages"]["numpy"] is None or isinstance(block["packages"]["numpy"], str)
    assert copy.deepcopy(block) == fingerprint.environment_block()
    canonical.to_bytes(block)                      # serializable
