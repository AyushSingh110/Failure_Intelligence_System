r"""
Print your own API key, so you can reach the dashboard without Google sign-in.

WHY THIS EXISTS
---------------
API keys are minted once, on a user's FIRST successful Google login, and stored
in MongoDB (app/auth.py::_generate_api_key). If Google sign-in later breaks —
a consent-screen policy change, a publishing-status change, an account that is
no longer an approved tester — the key is still there and still valid, but
there is no way to see it, because the only screen that shows it is behind the
login that is broken.

That is a circular lockout: you cannot reach the operator console to diagnose
the outage because the outage is in the login. This script breaks the circle by
reading the record directly.

The backend has always accepted `X-API-Key` as an alternative to the Google
bearer token (app/auth_guard.py::resolve_user), and the login page has an
"API key" option, so this is not a bypass of authentication — it is the same
credential check with a different credential.

SECURITY
--------
Prints a live secret to your terminal. Run it locally, not on a shared screen,
and do not paste the output anywhere. Anyone holding the key can act as you.
If it leaks, sign in and use "Regenerate API key" (POST /auth/regenerate-key),
which invalidates the old one immediately.

Usage (Windows CMD):

    conda activate failure-engine
    python scripts\show_api_key.py
    python scripts\show_api_key.py --email you@example.com
    python scripts\show_api_key.py --list
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _mask(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}  (length {len(key)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", help="which user to show (default: the admin)")
    ap.add_argument("--list", action="store_true",
                    help="list accounts and whether they hold a key, without "
                         "revealing any key")
    ap.add_argument("--reveal", action="store_true",
                    help="print the key in full (default masks it)")
    args = ap.parse_args()

    from dotenv import dotenv_values
    cfg = {**dotenv_values(ROOT / ".env"), **os.environ}
    uri = (cfg.get("MONGODB_URI") or "").strip()
    if not uri:
        print("ERROR: MONGODB_URI is not set in .env")
        return 1

    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pymongo is not installed in this environment.")
        print("       conda activate failure-engine && pip install pymongo")
        return 1

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=15000)
        db = client.get_default_database()
        names = db.list_collection_names()
    except Exception as exc:
        print(f"ERROR: cannot reach MongoDB ({type(exc).__name__}): {str(exc)[:200]}")
        return 1

    # The collection name has changed across versions; find it rather than
    # hardcoding a guess that silently returns "no users".
    col_name = next((n for n in names if "user" in n.lower()), None)
    if not col_name:
        print(f"ERROR: no user collection found. Collections: {', '.join(names) or '(none)'}")
        return 1
    col = db[col_name]

    total = col.count_documents({})
    print(f"database: {db.name}   collection: {col_name}   users: {total}")
    if total == 0:
        print("\nNo user records exist yet. A record is only created on a "
              "SUCCESSFUL Google login,\nso if Google sign-in has never worked, "
              "there is no key to recover — fix the consent\nscreen first.")
        return 1

    if args.list:
        print()
        for u in col.find({}, {"email": 1, "is_admin": 1, "api_key": 1, "_id": 0}):
            has = "yes" if (u.get("api_key") or "") else "NO KEY"
            print(f"  {u.get('email','?'):<40} admin={bool(u.get('is_admin'))}  key={has}")
        return 0

    if args.email:
        user = col.find_one({"email": args.email})
        if not user:
            print(f"\nNo user with email {args.email!r}. Run --list to see accounts.")
            return 1
    else:
        user = col.find_one({"is_admin": True}) or col.find_one({})
        print(f"(no --email given; showing {user.get('email','?')})")

    key = (user.get("api_key") or "").strip()
    if not key:
        print(f"\n{user.get('email')} has no api_key stored.")
        print("Sign in once by any means, then use Regenerate API key.")
        return 1

    print(f"\nemail : {user.get('email')}")
    print(f"admin : {bool(user.get('is_admin'))}")
    print(f"plan  : {user.get('plan', '?')}")
    if args.reveal:
        print(f"\nAPI KEY (secret - do not share or paste publicly):\n\n    {key}\n")
    else:
        print(f"\nAPI KEY: {_mask(key)}")
        print("\nRe-run with --reveal to print it in full.")
    print("Paste it into the login page: \"Sign in with an API key instead\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
