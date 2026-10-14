"""Signed access tokens for PetMind users (HMAC-SHA256, no external dependencies)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

DEFAULT_TTL_SECONDS = 60 * 60 * 24
_DEV_SECRET = "petmind-local-development-secret"


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class AccessTokenCodec:
    """Issues and verifies `<payload>.<signature>` tokens carrying the user id and expiry."""

    def __init__(self, secret: str | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._secret = secret
        self.ttl_seconds = ttl_seconds

    @property
    def secret(self) -> bytes:
        # Read lazily so tests and scripts may set PETMIND_AUTH_SECRET after import.
        return (self._secret or os.getenv("PETMIND_AUTH_SECRET") or _DEV_SECRET).encode()

    def issue(self, user_id: str) -> str:
        payload = json.dumps({"sub": str(user_id), "exp": int(time.time()) + self.ttl_seconds},
                             separators=(",", ":")).encode()
        encoded = _b64encode(payload)
        signature = hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest()
        return f"{encoded}.{_b64encode(signature)}"

    def verify(self, token: str) -> str:
        """Return the user id or raise ValueError for a malformed, forged or expired token."""
        try:
            encoded, signature = token.split(".", 1)
            expected = hmac.new(self.secret, encoded.encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(expected, _b64decode(signature)):
                raise ValueError("Invalid access token")
            payload = json.loads(_b64decode(encoded))
            if int(payload["exp"]) < int(time.time()):
                raise ValueError("Access token expired")
            return str(payload["sub"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Потрібно увійти в обліковий запис.") from exc


default_codec = AccessTokenCodec()


class TokenAuthMixin:
    """Gives a service the token methods the API layer relies on."""

    token_codec: AccessTokenCodec = default_codec

    def issue_access_token(self, user_id: str) -> str:
        return self.token_codec.issue(user_id)

    def _verify_access_token(self, token: str) -> str:
        return self.token_codec.verify(token)
