"""Tests for thor_mcp.auth."""

import time

from thor_mcp.auth import ThorAuth


def test_token_roundtrip():
    auth = ThorAuth("test-secret")
    token = auth.issue_token(subject="bench-bot")
    payload = auth.verify_token(token)
    assert payload is not None
    assert payload["sub"] == "bench-bot"
    assert "benchmark" in payload["scopes"]


def test_tampered_token_rejected():
    auth = ThorAuth("test-secret")
    token = auth.issue_token()
    tampered = token[:-4] + ("aaaa" if token[-4:] != "aaaa" else "bbbb")
    assert auth.verify_token(tampered) is None


def test_garbage_rejected():
    assert ThorAuth("secret").verify_token("not-a-token") is None
    assert ThorAuth("secret").verify_token("") is None


def test_expired_token_rejected():
    auth = ThorAuth("test-secret", token_ttl_s=-10)
    token = auth.issue_token()
    assert auth.verify_token(token) is None


def test_require_token():
    auth = ThorAuth("test-secret")
    token = auth.issue_token()
    assert auth.require_token(f"Bearer {token}") is not None
    try:
        auth.require_token(None)
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass
    try:
        auth.require_token("Bearer bogus")
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass
