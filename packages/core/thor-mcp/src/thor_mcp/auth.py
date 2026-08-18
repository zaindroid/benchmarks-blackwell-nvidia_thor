"""Authentication for ThorMCP.

Uses HMAC-signed tokens (stdlib only) so the server works out of the
box on Python 3.10+ without pulling in JWT/bcrypt stacks.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional


class ThorAuth:
    """Issues and verifies HMAC-signed bearer tokens."""

    def __init__(self, secret_key: str, token_ttl_s: int = 3600):
        self._secret = secret_key.encode("utf-8")
        self._ttl = token_ttl_s

    def issue_token(self, subject: str = "thor-client",
                    scopes: Optional[List[str]] = None) -> str:
        payload = {
            "sub": subject,
            "scopes": scopes or ["benchmark", "models", "experiments"],
            "exp": int(time.time()) + self._ttl,
        }
        return self._sign(payload)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Return the payload for a valid token, or None."""
        try:
            body, signature = token.rsplit(".", 1)
        except ValueError:
            return None
        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = json.loads(self._decode(body))
        except Exception:
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload

    def require_token(self, authorization: Optional[str]) -> Dict[str, Any]:
        """Validate an ``Authorization: Bearer <token>`` header value."""
        if not authorization:
            raise PermissionError("missing Authorization header")
        token = authorization.removeprefix("Bearer ").strip()
        payload = self.verify_token(token)
        if payload is None:
            raise PermissionError("invalid or expired token")
        return payload

    def _sign(self, payload: Dict[str, Any]) -> str:
        body = self._encode(json.dumps(payload, separators=(",", ":")))
        signature = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        return f"{body}.{signature}"

    @staticmethod
    def _encode(raw: str) -> str:
        return base64.urlsafe_b64encode(raw.encode()).rstrip(b"=").decode()

    @staticmethod
    def _decode(body: str) -> str:
        padding = "=" * (-len(body) % 4)
        return base64.urlsafe_b64decode(body + padding).decode()
