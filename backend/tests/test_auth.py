"""Auth flow tests — exercise the real JWT path (no dependency override)."""
from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.auth.security import hash_password, verify_password
from app.config import get_settings
from app.main import app

from fastapi.testclient import TestClient

pytestmark = pytest.mark.realauth

client = TestClient(app)


def _register(email, password="password123"):
    return client.post("/auth/register", json={"email": email, "password": password})


def _make_token(subject, *, secret=None, minutes=60):
    settings = get_settings()
    payload = {"sub": subject, "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes)}
    return jwt.encode(payload, secret or settings.jwt_secret, algorithm=settings.jwt_algorithm)


def test_register_login_and_protected_access():
    r = _register("alice@example.com")
    assert r.status_code == 201, r.text
    assert r.json()["email"] == "alice@example.com"
    assert "hashed_password" not in r.json()  # never leak the hash

    # duplicate registration is rejected
    assert _register("alice@example.com").status_code == 409

    r = client.post("/auth/login", json={"email": "alice@example.com", "password": "password123"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/documents", headers=headers).status_code == 200
    assert client.post("/auth/logout", headers=headers).status_code == 200


def test_register_rejects_short_password():
    assert _register("shorty@example.com", password="123").status_code == 422


def test_login_wrong_password_is_401():
    _register("bob@example.com")
    r = client.post("/auth/login", json={"email": "bob@example.com", "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_user_is_401():
    r = client.post("/auth/login", json={"email": "ghost@example.com", "password": "password123"})
    assert r.status_code == 401


def test_protected_routes_reject_missing_token():
    assert client.get("/documents").status_code == 401
    assert client.post("/ask", json={"question": "hi"}).status_code == 401


def test_protected_routes_reject_invalid_token():
    bad = {"Authorization": "Bearer not-a-real-token"}
    assert client.get("/documents", headers=bad).status_code == 401
    assert client.post("/ask", json={"question": "hi"}, headers=bad).status_code == 401


def test_health_is_public():
    assert client.get("/health").status_code == 200


# ---- password hashing -----------------------------------------------------

def test_password_hashing_salts_and_verifies():
    h1 = hash_password("password123")
    h2 = hash_password("password123")
    assert h1 != h2  # distinct salts
    assert verify_password("password123", h1)
    assert not verify_password("password123!", h1)


# ---- email handling -------------------------------------------------------

def test_email_is_case_insensitive():
    assert _register("Mixed@Example.com").status_code == 201
    r = client.post("/auth/login", json={"email": "mixed@example.com", "password": "password123"})
    assert r.status_code == 200


def test_invalid_email_rejected():
    assert _register("not-an-email").status_code == 422


# ---- token verification edge cases ----------------------------------------

def test_expired_token_rejected():
    headers = {"Authorization": f"Bearer {_make_token('1', minutes=-5)}"}
    assert client.get("/documents", headers=headers).status_code == 401


def test_token_with_wrong_secret_rejected():
    headers = {"Authorization": f"Bearer {_make_token('1', secret='a-different-secret')}"}
    assert client.get("/documents", headers=headers).status_code == 401


def test_token_for_unknown_user_rejected():
    headers = {"Authorization": f"Bearer {_make_token('999999')}"}
    assert client.get("/documents", headers=headers).status_code == 401


def test_token_with_non_int_subject_rejected():
    headers = {"Authorization": f"Bearer {_make_token('not-an-int')}"}
    assert client.get("/documents", headers=headers).status_code == 401
