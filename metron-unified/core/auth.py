"""
Simple file-based auth. Users are defined in users.json at the project root.
Passwords are hashed in memory on startup — never stored as hashes on disk.
JWT stored in an httpOnly cookie for persistent login.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt as _bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Request

SECRET_KEY = os.environ.get("SESSION_SECRET", "metron-change-this-secret-before-production")
ALGORITHM  = "HS256"
COOKIE_NAME = "metron_session"
_EXPIRE_DAYS = 7
_SECURE_COOKIE = os.environ.get("METRON_SECURE_COOKIE", "false").lower() == "true"

# email (lowercase) → {name, password_hash}
_USERS: dict[str, dict] = {}

def _load_users() -> None:
    path = Path(__file__).parent.parent / "users.json"
    if not path.exists():
        print(f"[Auth] users.json not found at {path}. No users loaded.")
        return
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        for u in entries:
            email = u.get("email", "").lower().strip()
            password = u.get("password", "")
            if email and password:
                _USERS[email] = {
                    "password_hash": _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()),
                }
        print(f"[Auth] Loaded {len(_USERS)} user(s) from users.json")
    except Exception as exc:
        print(f"[Auth] Failed to load users.json: {exc}")

_load_users()


def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = _USERS.get(email.lower().strip())
    if not user:
        return None
    if not _bcrypt.checkpw(password.encode(), user["password_hash"]):
        return None
    return {"email": email.lower().strip()}


def create_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    return jwt.encode({"sub": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"email": payload["sub"]}
    except JWTError:
        return None


def get_cookie_params() -> dict:
    return {
        "key":      COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure":   _SECURE_COOKIE,
        "max_age":  _EXPIRE_DAYS * 24 * 3600,
        "path":     "/",
    }


def get_current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user
