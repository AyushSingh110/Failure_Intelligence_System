"""
Publish the backend to the Hugging Face Space.

Runs both locally and from CI. Uploads only what the running Space needs —
the SDK, the server, and the model artifacts — not notebooks, evaluation data,
papers, tests or the frontend, which together are ~100 MB the Space never reads.

Uses huggingface_hub.upload_folder rather than a raw git push because it
diffs against the remote and skips unchanged files. That matters here: the
ONNX encoder alone is 90 MB, and a force-push would re-upload it on every
commit. With upload_folder, a typical backend change transfers a few KB.

Usage:
    export HF_TOKEN=hf_...            # or HUGGING_FACE_TOKEN in .env locally
    python deploy/huggingface/push_space.py
    python deploy/huggingface/push_space.py --repo Ayush-Singh9791/fie --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Directories copied wholesale into the Space.
RUNTIME_DIRS = ["fie", "app", "engine", "storage", "scripts", "models"]

# Individual files the server imports or reads at runtime.
#   config.py       -> settings, and _resolve_version() reads pyproject.toml
#   pyproject.toml  -> version source of truth; without it the API reports the
#                      stale literal fallback (this actually shipped once)
RUNTIME_FILES = ["config.py", "requirements.txt", "pyproject.toml", "LICENSE"]

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".git", "node_modules",
    ".pytest_cache", ".ruff_cache", "*.log",
)


def build_tree(stage: Path) -> None:
    """Assemble exactly what the Space should contain."""
    for d in RUNTIME_DIRS:
        src = ROOT / d
        if not src.exists():
            print(f"  ! missing directory, skipping: {d}")
            continue
        shutil.copytree(src, stage / d, ignore=IGNORE)

    for f in RUNTIME_FILES:
        src = ROOT / f
        if src.exists():
            shutil.copy(src, stage / f)
        else:
            print(f"  ! missing file, skipping: {f}")

    # Space entrypoint and metadata must sit at the repo root.
    shutil.copy(ROOT / "deploy/huggingface/space_app.py", stage / "space_app.py")

    # Dockerfile: rewrite the two paths that assume the repo layout.
    df = (ROOT / "deploy/huggingface/Dockerfile").read_text(encoding="utf-8")
    df = df.replace(
        'CMD ["python", "deploy/huggingface/space_app.py"]',
        'CMD ["python", "space_app.py"]',
    )
    # Models are uploaded with the tree, so no build-time fetch is needed — and
    # skipping it removes a whole class of build flake (a GitHub release being
    # briefly unreachable would otherwise fail the Space build).
    df = df.replace(
        "RUN python scripts/download_models.py --strict && mkdir -p storage",
        "RUN mkdir -p storage",
    )
    # gradio is a Space-only dependency: the library and the API server do not
    # need it, so it is deliberately absent from requirements.txt.
    df = df.replace(
        "RUN pip install --user --no-warn-script-location -r requirements.txt",
        'RUN pip install --user --no-warn-script-location -r requirements.txt '
        '"gradio>=4.44,<6" "faiss-cpu>=1.8.0"',
    )
    (stage / "Dockerfile").write_text(df, encoding="utf-8")

    # Space README carries the YAML frontmatter that configures the Space.
    shutil.copy(ROOT / "deploy/huggingface/SPACE_README.md", stage / "README.md")

    # Binary artifacts must go through git-lfs on the Hub.
    (stage / ".gitattributes").write_text(
        "*.onnx filter=lfs diff=lfs merge=lfs -text\n"
        "*.pkl filter=lfs diff=lfs merge=lfs -text\n"
        "*.index filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getenv("HF_SPACE_REPO", "Ayush-Singh9791/fie"))
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true", help="build the tree, upload nothing")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        # Local convenience: fall back to .env, which CI does not have.
        try:
            from dotenv import dotenv_values
            token = (dotenv_values(ROOT / ".env").get("HUGGING_FACE_TOKEN") or "").strip()
        except Exception:
            token = ""
    if not token and not args.dry_run:
        print("ERROR: no HF token. Set HF_TOKEN (CI) or HUGGING_FACE_TOKEN in .env (local).")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        stage.mkdir(parents=True)
        print(f"building tree for {args.repo} ...")
        build_tree(stage)

        files = [f for f in stage.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        print(f"  {len(files)} files, {size / 1e6:.1f} MB")

        # Fail loudly rather than shipping a Space that boots without its
        # detector. A missing model does not crash the server — it degrades to
        # reduced recall while still returning confident verdicts.
        required = [
            "fie/models/minilm-onnx/model.onnx",
            "fie/models/pair_intent_classifier_v6_3b.pkl",
            "app/main.py",
            "space_app.py",
            "Dockerfile",
            "README.md",
        ]
        missing = [r for r in required if not (stage / r).exists()]
        if missing:
            print("ERROR: refusing to deploy, missing required files:")
            for m in missing:
                print("   -", m)
            return 1
        print("  required artifacts present")

        if args.dry_run:
            print("dry run — nothing uploaded")
            return 0

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        commit = args.message or os.getenv("GITHUB_SHA", "")[:8] or "manual deploy"
        print(f"uploading to https://huggingface.co/spaces/{args.repo} ...")
        api.upload_folder(
            folder_path=str(stage),
            repo_id=args.repo,
            repo_type="space",
            commit_message=f"deploy: {commit}",
            # Remove files that no longer exist locally, so a deleted module
            # cannot linger and shadow an import in the running Space.
            delete_patterns=["*"],
        )

    print("\ndeployed")
    print(f"  Space: https://huggingface.co/spaces/{args.repo}")
    owner, name = args.repo.split("/", 1)
    print(f"  App:   https://{owner.lower()}-{name}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
