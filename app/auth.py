from __future__ import annotations
import hmac
import logging
import os
import secrets
import string
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

#Config
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_H  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
ADMIN_EMAIL   = os.getenv("ADMIN_EMAIL", "")

# ── Signing secret ────────────────────────────────────────────────────────────
# A session token is only as trustworthy as the secret that signs it. Without a
# strong JWT_SECRET_KEY no token is issued and none is accepted, and the server
# refuses to start (app.main.enforce_startup_security). The old behaviour — sign
# with a constant that sits in the public source — let anyone forge an admin
# token on a deployment that forgot the variable.
#
# FIE_ALLOW_INSECURE_DEV_SECRET=1 restores that constant for LOCAL DEVELOPMENT
# ONLY. It is never an acceptable production setting and is logged as an error
# at every start.
MIN_SECRET_LENGTH    = 32
_DEV_CONSTANT_SECRET = "fie-insecure-dev-secret-DO-NOT-USE-IN-PROD"


class UserStoreUnavailable(Exception):
    """The users collection could not be read. Callers must fail closed."""


def dev_secret_allowed() -> bool:
    return os.getenv("FIE_ALLOW_INSECURE_DEV_SECRET", "").strip().lower() in ("1", "true", "yes")


def has_strong_secret() -> bool:
    return len(os.getenv("JWT_SECRET_KEY", "")) >= MIN_SECRET_LENGTH


def signing_secret() -> Optional[str]:
    """The secret to sign and verify with, or None when tokens must not be used."""
    raw = os.getenv("JWT_SECRET_KEY", "")
    if len(raw) >= MIN_SECRET_LENGTH:
        return raw
    if dev_secret_allowed():
        return _DEV_CONSTANT_SECRET
    return None


# Helpers
def _generate_api_key() -> str:
    chars = string.ascii_lowercase + string.digits
    rand  = ''.join(secrets.choice(chars) for _ in range(16))
    return f"fie-{rand}"


def _generate_tenant_id(email: str) -> str:
    email_part = email.split("@")[0].lower()
    email_part = ''.join(c for c in email_part if c.isalnum())[:10]
    rand = secrets.token_hex(3)
    return f"{email_part}-{rand}"


# MongoDB Operations
#
# A single module-level MongoClient is reused across all auth lookups.
# MongoClient maintains its own connection pool — creating a new client per
# request bypasses the pool, leaks connections, and exhausts the Atlas
# free-tier connection cap (500) under load.
_mongo_client = None
_mongo_lock   = threading.Lock()
_api_key_index_requested = False


def _get_users_collection():
    """Returns the MongoDB users collection, reusing one pooled client."""
    global _mongo_client, _api_key_index_requested
    try:
        from config import get_settings
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        settings = get_settings()
        if _mongo_client is None:
            with _mongo_lock:
                if _mongo_client is None:
                    _mongo_client = MongoClient(
                        settings.mongodb_uri, server_api=ServerApi('1')
                    )
        db = _mongo_client[settings.mongodb_db_name]
        users = db["users"]
        if not _api_key_index_requested:
            # Every API-key request looks a user up by this field.
            _api_key_index_requested = True
            try:
                users.create_index("api_key", background=True)
            except Exception as exc:
                logger.warning("users.api_key index request failed: %s", type(exc).__name__)
        return users
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        return None


def get_or_create_user(
    email:   str,
    name:    str,
    picture: str = "",
) -> dict:
    # An account without an e-mail has no owner. Before this check, every login
    # that returned no e-mail shared one account, and that account became a
    # platform admin whenever ADMIN_EMAIL was unset ("" == "").
    if not isinstance(email, str) or not email.strip():
        raise ValueError("a verified e-mail address is required to create an account")

    collection = _get_users_collection()
    if collection is None:
        raise Exception("Database unavailable — check MONGODB_URI in .env")

    # Check if user already exists
    existing = collection.find_one({"email": email})
    if existing:
        collection.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        logger.info("Existing user logged in")
        return existing

    # New user — build full profile. An empty ADMIN_EMAIL grants admin to nobody.
    is_admin  = bool(ADMIN_EMAIL) and (email.lower() == ADMIN_EMAIL.lower())
    api_key   = _generate_api_key()
    tenant_id = _generate_tenant_id(email)

    new_user = {
        "email":       email,
        "name":        name,
        "picture":     picture,
        "api_key":     api_key,
        "tenant_id":   tenant_id,
        "is_admin":    is_admin,
        "plan":        "admin" if is_admin else "free",
        "calls_used":  0,
        "calls_limit": 999999 if is_admin else 1000,
        "created_at":  datetime.now(timezone.utc),
        "last_login":  datetime.now(timezone.utc),
    }

    collection.insert_one(new_user)
    # Never log the API key itself — keys in log sinks outlive rotation.
    logger.info("New user created | admin=%s", is_admin)
    return new_user


def env_key_user(api_key: str) -> Optional[dict]:
    """
    The operator's database-independent credential: the server's own FIE_API_KEY.
    Compared in constant time. Returns a synthetic platform-admin user, or None.
    """
    env_key = os.getenv("FIE_API_KEY", "")
    if not env_key or not isinstance(api_key, str):
        return None
    if not hmac.compare_digest(api_key.encode("utf-8"), env_key.encode("utf-8")):
        return None
    admin = os.getenv("ADMIN_EMAIL", "") or "local@fie.dev"
    return {
        "tenant_id":   admin,
        "email":       admin,
        "name":        "operator",
        "api_key":     api_key,
        "is_admin":    True,
        "plan":        "admin",
        "calls_limit": 100_000,
        "calls_used":  0,
    }


def find_user_by_api_key(api_key: str) -> Optional[dict]:
    """Database lookup by API key. Raises UserStoreUnavailable when the store cannot be read."""
    if not api_key or not isinstance(api_key, str):
        return None
    collection = _get_users_collection()
    if collection is None:
        raise UserStoreUnavailable("users collection unavailable")
    try:
        user = collection.find_one({"api_key": api_key})
    except Exception as exc:
        raise UserStoreUnavailable(type(exc).__name__) from exc
    return user if isinstance(user, dict) else None


def find_user_by_email(email: str) -> Optional[dict]:
    """Database lookup by e-mail. Raises UserStoreUnavailable when the store cannot be read."""
    if not email or not isinstance(email, str):
        return None
    collection = _get_users_collection()
    if collection is None:
        raise UserStoreUnavailable("users collection unavailable")
    try:
        user = collection.find_one({"email": email})
    except Exception as exc:
        raise UserStoreUnavailable(type(exc).__name__) from exc
    return user if isinstance(user, dict) else None


def get_user_by_api_key(api_key: str) -> Optional[dict]:
    """
    Finds user by API key: the environment key first, then the database.
    Returns None when the key is unknown or the store is unavailable.
    """
    if not api_key:
        return None
    user = env_key_user(api_key)
    if user is not None:
        return user
    try:
        return find_user_by_api_key(api_key)
    except UserStoreUnavailable:
        return None


def get_user_by_email(email: str) -> Optional[dict]:
    """Finds user by email address. None when unknown or the store is unavailable."""
    try:
        return find_user_by_email(email)
    except UserStoreUnavailable:
        return None


def get_all_users() -> list[dict]:
    """Admin only — returns all registered users, without their API keys."""
    collection = _get_users_collection()
    if collection is None:
        return []
    users = list(collection.find({}, {"_id": 0, "api_key": 0}))
    for u in users:
        u.pop("api_key", None)
        for k in ["created_at", "last_login"]:
            if k in u and hasattr(u[k], "isoformat"):
                u[k] = u[k].isoformat()
    return users


def increment_usage(tenant_id: str) -> bool:
    """
    Increments call counter for a user.
    Returns True if within limit, False if exceeded.
    Admin users have no limit (calls_limit=999999).
    """
    collection = _get_users_collection()
    if collection is None:
        return True  # fail open — never block on DB error

    user = collection.find_one({"tenant_id": tenant_id})
    if not user:
        return True

    if not user.get("is_admin"):
        if user.get("calls_used", 0) >= user.get("calls_limit", 1000):
            logger.warning("Usage limit exceeded for a tenant")
            return False

    collection.update_one(
        {"tenant_id": tenant_id},
        {"$inc": {"calls_used": 1}}
    )
    return True


def regenerate_api_key(email: str) -> Optional[str]:
    """
    Replace the user's API key. Returns the new key only when it was stored;
    None when the store is unavailable or the user does not exist. The old key
    stops working as soon as this returns.
    """
    collection = _get_users_collection()
    if collection is None:
        return None
    new_key = _generate_api_key()
    try:
        result = collection.update_one(
            {"email": email},
            {"$set": {"api_key": new_key}}
        )
    except Exception as exc:
        logger.error("API key rotation failed: %s", type(exc).__name__)
        return None
    if getattr(result, "matched_count", 0) != 1:
        return None
    logger.info("API key regenerated")
    return new_key


# JWT Session Tokens
def create_session_token(user: dict) -> str:
    """
    Sign a 24-hour session token. The payload carries identity only — never the
    API key: a token is readable by anyone who holds it.
    """
    secret = signing_secret()
    if secret is None:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set or is shorter than 32 characters; "
            "session tokens cannot be issued."
        )
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_H)
    payload = {
        "email":     user["email"],
        "name":      user["name"],
        "picture":   user.get("picture", ""),
        "tenant_id": user["tenant_id"],
        "is_admin":  user.get("is_admin", False),   # a hint for the dashboard; never used for authorization
        "plan":      user.get("plan", "free"),
        "exp":       expire,
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_session_token(token: str) -> Optional[dict]:
    secret = signing_secret()
    if secret is None:
        logger.error("Session token refused: no strong JWT_SECRET_KEY is configured")
        return None
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.info("Session token expired — user must re-login")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning(
            "degraded capability=verify_session_token impact='this optional step was skipped' "
            "reason=%s: %s", type(exc).__name__, exc,
        )
        return None
