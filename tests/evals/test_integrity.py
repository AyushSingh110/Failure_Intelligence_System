"""Model integrity: mismatch, missing file and missing role all abort. Never a warning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals import integrity
from evals.integrity import ModelIntegrityError
from _helpers import REPO_ROOT, models_present

ROLES = {
    "pair_classifier": "fie/models/pair.pkl",
    "pair_meta": "fie/models/pair_meta.json",
    "encoder": "fie/models/enc/model.onnx",
}
REQUIRED = ["pair_classifier", "pair_meta", "encoder"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A temporary repository with three model files and a matching manifest."""
    contents = {
        "fie/models/pair.pkl": b"\x80\x04 pretend pickle bytes",
        "fie/models/pair_meta.json": json.dumps(
            {"version": "v9.9", "threshold": 0.5, "embed_model": "x/y"}).encode(),
        "fie/models/enc/model.onnx": b"onnx-bytes" * 300_000,       # ~3 MB: spans several chunks
    }
    artifacts = []
    for rel, data in contents.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        artifacts.append({"path": rel, "sha256": _sha(data), "size": len(data)})
    manifest = tmp_path / "scripts" / "model_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"release_tag": "models-test", "artifacts": artifacts}),
                        encoding="utf-8")
    return tmp_path


def test_streaming_hash_equals_one_shot_hash(repo):
    path = repo / "fie/models/enc/model.onnx"
    assert integrity.sha256_file(path) == _sha(path.read_bytes())
    assert integrity.sha256_file(path, chunk=7) == _sha(path.read_bytes())


def test_matching_files_verify(repo):
    table = integrity.verify_roles(ROLES, repo, required=REQUIRED)
    assert set(table) == set(ROLES)
    assert table["pair_classifier"]["file"] == "pair.pkl"
    assert table["pair_classifier"]["expected_from"] == "manifest"
    assert all(len(v["sha256"]) == 64 for v in table.values())


def test_hash_mismatch_aborts_and_names_the_file(repo):
    (repo / "fie/models/pair.pkl").write_bytes(b"tampered")
    with pytest.raises(ModelIntegrityError) as exc:
        integrity.verify_roles(ROLES, repo, required=REQUIRED)
    msg = str(exc.value)
    assert "fie/models/pair.pkl" in msg and "expected (manifest)" in msg and "actual" in msg
    assert "download_models.py --strict" in msg
    assert "aborted before any model was loaded" in msg


def test_missing_file_aborts(repo):
    (repo / "fie/models/pair.pkl").unlink()
    with pytest.raises(ModelIntegrityError, match="file is missing: fie/models/pair.pkl"):
        integrity.verify_roles(ROLES, repo, required=REQUIRED)


def test_missing_role_aborts(repo):
    roles = {k: v for k, v in ROLES.items() if k != "encoder"}
    with pytest.raises(ModelIntegrityError, match="role 'encoder' is required"):
        integrity.verify_roles(roles, repo, required=REQUIRED)


def test_role_without_manifest_entry_aborts(repo):
    extra = repo / "fie/models/unpublished.pkl"
    extra.write_bytes(b"not in the manifest")
    roles = dict(ROLES, pair_classifier="fie/models/unpublished.pkl")
    with pytest.raises(ModelIntegrityError, match="has no entry in scripts/model_manifest.json"):
        integrity.verify_roles(roles, repo, required=REQUIRED)


def test_all_problems_are_reported_together(repo):
    (repo / "fie/models/pair.pkl").write_bytes(b"tampered")
    (repo / "fie/models/pair_meta.json").unlink()
    with pytest.raises(ModelIntegrityError) as exc:
        integrity.verify_roles(ROLES, repo, required=REQUIRED)
    assert "pair.pkl" in str(exc.value) and "pair_meta.json" in str(exc.value)


def test_explicit_hash_override_is_marked_as_not_from_the_manifest(repo):
    extra = repo / "fie/models/unpublished.pkl"
    extra.write_bytes(b"research model")
    roles = dict(ROLES, pair_classifier="fie/models/unpublished.pkl")
    table = integrity.verify_roles(roles, repo, required=REQUIRED,
                                   overrides={"pair_classifier": _sha(b"research model")})
    assert table["pair_classifier"]["expected_from"] == "command line"
    assert table["pair_classifier"]["in_manifest"] is False
    with pytest.raises(ModelIntegrityError):
        integrity.verify_roles(roles, repo, required=REQUIRED,
                               overrides={"pair_classifier": "0" * 64})


def test_file_changed_during_the_run_is_caught(repo):
    table = integrity.verify_roles(ROLES, repo, required=REQUIRED)
    integrity.reverify(table, repo)                          # unchanged: passes
    (repo / "fie/models/pair.pkl").write_bytes(b"swapped mid-run")
    with pytest.raises(ModelIntegrityError, match="changed during the run"):
        integrity.reverify(table, repo)
    (repo / "fie/models/pair.pkl").unlink()
    with pytest.raises(ModelIntegrityError, match="disappeared during the run"):
        integrity.reverify(table, repo)


@pytest.mark.parametrize("path", ["../outside.pkl", "fie/../../outside.pkl"])
def test_paths_cannot_escape_the_repository(repo, path):
    with pytest.raises(ModelIntegrityError, match="escapes the repository"):
        integrity.verify_roles({"pair_classifier": path}, repo,
                               overrides={"pair_classifier": "0" * 64})


def test_unreadable_manifest_aborts(tmp_path):
    with pytest.raises(ModelIntegrityError, match="cannot read model manifest"):
        integrity.load_manifest(tmp_path)
    bad = tmp_path / "scripts" / "model_manifest.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="cannot read model manifest"):
        integrity.load_manifest(tmp_path)


def test_declared_metadata_is_read_from_verified_files(repo):
    integrity.verify_roles(ROLES, repo, required=REQUIRED)
    meta = integrity.declared_metadata(ROLES, repo)
    assert meta["pair"] == {"declared_version": "v9.9", "threshold": 0.5,
                            "embed_model": "x/y", "model_type": None}


def test_unmanifested_model_files_are_listed(repo):
    (repo / "fie/models/experiment.pkl").write_bytes(b"x")
    assert integrity.unmanifested_model_files(repo) == ["fie/models/experiment.pkl"]


# ── the real repository ──────────────────────────────────────────────────────

def test_real_manifest_parses_and_names_the_shipped_model():
    manifest = integrity.load_manifest(REPO_ROOT)
    assert manifest["release_tag"]
    assert "fie/models/pair_intent_classifier_v6_3b.pkl" in manifest["artifacts"]
    assert "fie/models/minilm-onnx/model.onnx" in manifest["artifacts"]


def test_real_shipped_models_match_the_manifest(need_models):
    roles = {
        "pair_classifier": "fie/models/pair_intent_classifier_v6_3b.pkl",
        "pair_meta": "fie/models/pair_intent_meta_v6_3b.json",
        "meta_classifier": "fie/models/meta_clf.pkl",
        "meta_classifier_meta": "fie/models/meta_clf.json",
        "encoder": "fie/models/minilm-onnx/model.onnx",
        "tokenizer": "fie/models/minilm-onnx/tokenizer.json",
    }
    table = integrity.verify_roles(roles, REPO_ROOT, required=list(roles))
    assert table["pair_classifier"]["sha256"].startswith("9c682b28")
    meta = integrity.declared_metadata(roles, REPO_ROOT)
    assert meta["pair"]["declared_version"] == "v6.3b" and meta["pair"]["threshold"] == 0.5
