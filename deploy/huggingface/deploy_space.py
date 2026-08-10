"""
Create and configure the Hugging Face Space, end to end.

Does everything the manual walkthrough does, but without the chance of pasting a
secret into the wrong box:

  1. validates the HF token actually has WRITE scope (a read token fails late
     and confusingly, after the repo appears to exist)
  2. creates the Space (docker SDK) if it does not exist
  3. pushes Space-specific Dockerfile + README to the Space's `main`
  4. uploads the backend secrets from .env as Space SECRETS, never as public
     Variables — the distinction matters: Variables are world-readable on the
     Space page.

Usage:
    python deploy/huggingface/deploy_space.py --name fie
    python deploy/huggingface/deploy_space.py --name fie --secrets-only

The token is read from HUGGING_FACE_TOKEN in .env and never printed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

API = "https://huggingface.co/api"

# Backend configuration copied from .env into the Space.
# Deliberately explicit — never ship the whole .env, which also holds the PyPI
# token, the HF token itself and local-only paths that would leak or confuse.
SECRET_KEYS = [
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "JWT_SECRET_KEY",
    "JWT_ALGORITHM",
    "JWT_EXPIRE_HOURS",
    "ADMIN_EMAIL",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_REDIRECT_URI",
    "CORS_ALLOWED_ORIGINS",
    "GROQ_API_KEY",
    "GROQ_ENABLED",
    "GROQ_MODELS",
    "GROQ_FAST_MODEL",
    "SERPER_API_KEY",
    "SERPER_ENABLED",
    "SENDGRID_API_KEY",
    "NOTIFICATION_EMAIL",
    "FIE_FROM_EMAIL",
]

# Never copy these into the Space, even if present in .env.
NEVER_COPY = {"HUGGING_FACE_TOKEN", "PYPI_TOKEN", "pypi_username", "FIE_API_KEY", "FIE_API_URL"}


def _req(method: str, path: str, token: str, payload: dict | None = None) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fie", help="Space name (default: fie)")
    ap.add_argument("--secrets-only", action="store_true",
                    help="Skip create/push; only sync secrets")
    ap.add_argument("--private", action="store_true", help="Create the Space private")
    args = ap.parse_args()

    from dotenv import dotenv_values
    cfg = dotenv_values(ROOT / ".env")
    token = (cfg.get("HUGGING_FACE_TOKEN") or "").strip()
    if not token:
        print("ERROR: HUGGING_FACE_TOKEN not set in .env")
        return 1

    # ── 1. Token scope ────────────────────────────────────────────────────────
    status, me = _req("GET", "/whoami-v2", token)
    if status != 200 or not isinstance(me, dict):
        print(f"ERROR: token rejected by Hugging Face (HTTP {status})")
        return 1
    user = me.get("name")
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"authenticated as: {user}   token role: {role}")

    if role != "write":
        print(
            "\nERROR: this token is READ-only. Creating a Space needs WRITE.\n"
            "  1. https://huggingface.co/settings/tokens\n"
            "  2. New token -> type 'Write'\n"
            "  3. Replace HUGGING_FACE_TOKEN in .env\n"
        )
        return 1

    repo_id = f"{user}/{args.name}"
    space_url = f"https://huggingface.co/spaces/{repo_id}"

    # ── 2. Create the Space ───────────────────────────────────────────────────
    if not args.secrets_only:
        status, resp = _req("POST", "/repos/create", token, {
            "type": "space",
            "name": args.name,
            "private": bool(args.private),
            "sdk": "docker",
        })
        if status in (200, 201):
            print(f"created Space: {space_url}")
        elif status == 409:
            print(f"Space already exists: {space_url}")
        else:
            print(f"ERROR creating Space (HTTP {status}): {resp}")
            return 1

        # ── 3. Push code ──────────────────────────────────────────────────────
        # Space needs Dockerfile + README.md at the repo ROOT, so build a
        # dedicated branch rather than polluting main.
        print("\npushing code to the Space ...")
        remote = f"https://{user}:{token}@huggingface.co/spaces/{repo_id}"

        def git(*a, check=True):
            r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
            if check and r.returncode != 0:
                raise RuntimeError(f"git {' '.join(a)} failed:\n{r.stderr[:400]}")
            return r

        original = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        git("branch", "-D", "hf-space", check=False)
        git("checkout", "-q", "-b", "hf-space")
        try:
            (ROOT / "Dockerfile").write_bytes(
                (ROOT / "deploy/huggingface/Dockerfile").read_bytes())
            (ROOT / "README.md").write_bytes(
                (ROOT / "deploy/huggingface/SPACE_README.md").read_bytes())
            git("add", "-A")
            git("commit", "-q", "-m", "Hugging Face Space build", check=False)
            r = subprocess.run(
                ["git", "push", "--force", remote, "hf-space:main"],
                cwd=ROOT, capture_output=True, text=True,
            )
            if r.returncode != 0:
                # Scrub the token out of any error output before showing it.
                print("ERROR pushing:", r.stderr.replace(token, "***")[:500])
                return 1
            print("pushed. build starts automatically.")
        finally:
            git("checkout", "-q", "--force", original)
            git("branch", "-D", "hf-space", check=False)

    # ── 4. Secrets ────────────────────────────────────────────────────────────
    print("\nsyncing secrets ...")
    sent, skipped = [], []
    for key in SECRET_KEYS:
        if key in NEVER_COPY:
            continue
        value = (cfg.get(key) or "").strip()
        if not value:
            skipped.append(key)
            continue
        status, resp = _req("POST", f"/spaces/{repo_id}/secrets", token,
                            {"key": key, "value": value})
        if status in (200, 201, 204):
            sent.append(key)
        else:
            print(f"  ! {key}: HTTP {status} {str(resp)[:100]}")

    # Space-specific runtime settings (not from .env).
    for key, value in [
        ("FIE_SCAN_FAILURE_MODE", "closed"),
        ("FIE_LAYER_POOL_SIZE", "4"),
        ("FIE_NO_AUTO_DOWNLOAD", "1"),
        ("DEBUG", "false"),
    ]:
        status, _ = _req("POST", f"/spaces/{repo_id}/secrets", token,
                         {"key": key, "value": value})
        if status in (200, 201, 204):
            sent.append(key)

    print(f"  set {len(sent)} secrets: {', '.join(sorted(sent))}")
    if skipped:
        print(f"  skipped (empty in .env): {', '.join(skipped)}")

    print(f"""
──────────────────────────────────────────────────────────
  Space:  {space_url}
  API:    https://{user.lower()}-{args.name}.hf.space

  Build takes ~4 min. Watch it under the Logs tab.

  Then verify:
    curl https://{user.lower()}-{args.name}.hf.space/health
    curl https://{user.lower()}-{args.name}.hf.space/health/deep

  And point the dashboard at it (Cloudflare Pages -> env vars):
    VITE_API_URL = https://{user.lower()}-{args.name}.hf.space/api/v1
──────────────────────────────────────────────────────────
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
