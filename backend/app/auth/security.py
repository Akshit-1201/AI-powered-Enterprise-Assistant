"""Password hashing (bcrypt) and JWT issue/verify.

Uses the ``bcrypt`` library directly rather than passlib: passlib 1.7.4 is unmaintained
and crashes against bcrypt 5.x (it feeds a >72-byte probe to bcrypt during backend init).
bcrypt-direct is the robust, current choice.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

from app.config import get_settings


def hash_password(password: str) -> str:
    # bcrypt only considers the first 72 bytes; truncate explicitly before hashing.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
