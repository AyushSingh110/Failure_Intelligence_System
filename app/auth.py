"""
app/auth.py
Complete authentication system
"""

from __future__ import annotations

import logging
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
JWT_SECRET    = os.getenv("JWT_SECRET_KEY", "fie-default-secret-change-this-now-32chars")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_H  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
ADMIN_EMAIL   = os.getenv("ADMIN_EMAIL", "ayushsingh355vns@gmail.com")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _generate_api_key() -> str:
    """
    Generates unique API key for each user.
    Format: fie-xxxxxxxxxxxxxxxx
    Example: fie-k8m2x9p1q4r7t3n6
    Cryptographically secure — uses secrets module.
    """
    chars = string.ascii_lowercase + string.digits
    rand  = ''.join(secrets.choice(chars) for _ in range(16))
    return f"fie-{rand}"


def _generate_tenant_id(email: str) -> str:
    """
    Generates unique tenant ID from email.
    Used to isolate each user's data in MongoDB.
    Format: emailpart-random6chars
    Example: ayush-k8m2x9
    """
    email_part = email.split("@")[0].lower()
    email_part = ''.join(c for c in email_part if c.isalnum())[:10]
    rand = secrets.token_hex(3)
    return f"{email_part}-{rand}"


# ── MongoDB Operations ──────────────────────────────────────────────────────

def _get_users_collection():
    """Returns MongoDB users collection."""
    try:
        from config import get_settings
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        settings = get_settings()
        client   = MongoClient(settings.mongodb_uri, server_api=ServerApi('1'))
        db       = client[settings.mongodb_db_name]
        return db["users"]
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        return None


def get_or_create_user(
    email:   str,
    name:    str,
    picture: str = "",
) -> dict:
    """
    Gets existing user OR creates new one on first Google login.

    Called every time user logs in with Google:
      First time  → creates account + generates API key
      Every time  → updates last_login timestamp
      Returns     → full user dict from MongoDB
    """
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
        logger.info("Existing user logged in: %s", email)
        return existing

    # New user — build full profile
    is_admin  = (email.lower() == ADMIN_EMAIL.lower())
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
    logger.info(
        "New user created | email=%s | tenant_id=%s | api_key=%s | admin=%s",
        email, tenant_id, api_key, is_admin,
    )
    return new_user


def get_user_by_api_key(api_key: str) -> Optional[dict]:
    """
    Finds user by API key.
    Called on every /monitor request to identify tenant.
    Returns None if key is invalid.
    """
    if not api_key:
        return None
    collection = _get_users_collection()
    if collection is None:
        return None
    return collection.find_one({"api_key": api_key})


def get_user_by_email(email: str) -> Optional[dict]:
    """Finds user by email address."""
    collection = _get_users_collection()
    if collection is None:
        return None
    return collection.find_one({"email": email})


def get_all_users() -> list[dict]:
    """Admin only — returns all registered users."""
    collection = _get_users_collection()
    if collection is None:
        return []
    users = list(collection.find({}, {"_id": 0}))
    for u in users:
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
            logger.warning("Usage limit exceeded for %s", tenant_id)
            return False

    collection.update_one(
        {"tenant_id": tenant_id},
        {"$inc": {"calls_used": 1}}
    )
    return True


def regenerate_api_key(email: str) -> str:
    """
    Generates new API key for user.
    Old key immediately becomes invalid.
    Returns new API key.
    """
    new_key    = _generate_api_key()
    collection = _get_users_collection()
    if collection:
        collection.update_one(
            {"email": email},
            {"$set": {"api_key": new_key}}
        )
    logger.info("API key regenerated for %s", email)
    return new_key


# ── JWT Session Tokens ──────────────────────────────────────────────────────

def create_session_token(user: dict) -> str:
    """
    Creates JWT token for logged-in user.
    Stored in Streamlit session_state.
    Expires after JWT_EXPIRE_HOURS (default 24h).

    Token payload contains everything needed
    to identify the user without hitting MongoDB:
      email, name, tenant_id, api_key, is_admin, plan
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_H)
    payload = {
        "email":     user["email"],
        "name":      user["name"],
        "picture":   user.get("picture", ""),
        "tenant_id": user["tenant_id"],
        "api_key":   user["api_key"],
        "is_admin":  user.get("is_admin", False),
        "plan":      user.get("plan", "free"),
        "exp":       expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_session_token(token: str) -> Optional[dict]:
    """
    Verifies JWT token — returns payload or None.
    None means: invalid token or expired → show login page.
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.info("Session token expired — user must re-login")
        return None
    except jwt.InvalidTokenError:
        return None