"""
Cognito JWT auth. Verifies RS256 Bearer tokens issued by AWS Cognito User Pools.
JWKS keys are cached for 1 hour to avoid repeated network fetches.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from typing import Optional

from jose import jwt, jwk
from fastapi import HTTPException, Request

_REGION    = lambda: os.environ.get("COGNITO_REGION", "us-east-1")
_POOL_ID   = lambda: os.environ.get("COGNITO_USER_POOL_ID", "")
_CLIENT_ID = lambda: os.environ.get("COGNITO_CLIENT_ID", "")

_jwks_cache: dict = {"keys": {}, "expires": 0.0}


def _fetch_jwks() -> dict:
    now = time.monotonic()
    if _jwks_cache["expires"] > now and _jwks_cache["keys"]:
        return _jwks_cache["keys"]
    pool_id = _POOL_ID()
    if not pool_id:
        raise RuntimeError("COGNITO_USER_POOL_ID env var not set")
    url = (
        f"https://cognito-idp.{_REGION()}.amazonaws.com"
        f"/{pool_id}/.well-known/jwks.json"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    keys = {k["kid"]: k for k in data["keys"]}
    _jwks_cache.update({"keys": keys, "expires": now + 3600})
    return keys


def verify_cognito_token(token: str) -> Optional[dict]:
    try:
        headers = jwt.get_unverified_headers(token)
        kid = headers.get("kid")
        keys = _fetch_jwks()
        if kid not in keys:
            _jwks_cache["expires"] = 0.0  # force refresh once
            keys = _fetch_jwks()
        if kid not in keys:
            return None
        public_key = jwk.construct(keys[kid])
        issuer = (
            f"https://cognito-idp.{_REGION()}.amazonaws.com/{_POOL_ID()}"
        )
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=_CLIENT_ID(),
            issuer=issuer,
        )
        email = claims.get("email") or claims.get("cognito:username", "")
        return {"email": email}
    except Exception as exc:
        print(f"[Auth] Token verification failed: {exc}")
        return None


def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header[7:]
    user = verify_cognito_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user
