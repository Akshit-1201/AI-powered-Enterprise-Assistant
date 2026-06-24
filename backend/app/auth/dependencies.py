"""Auth dependency: resolve the current user from a Bearer JWT.

This is the Phase 3 end of the user-scoping seam: endpoints depend on get_current_user
and derive ``user_id = str(current_user.id)``, which then scopes memory + RAG.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.auth.security import decode_token
from app.db.database import SessionLocal
from app.db.models import User

# auto_error=False so a missing header yields our 401 (not HTTPBearer's default 403).
_bearer = HTTPBearer(auto_error=False)

_credentials_exc = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    if creds is None or not creds.credentials:
        raise _credentials_exc
    try:
        payload = decode_token(creds.credentials)
        subject = payload.get("sub")
        if subject is None:
            raise _credentials_exc
        user_id = int(subject)
    except (JWTError, ValueError, TypeError):
        raise _credentials_exc

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
    finally:
        db.close()
    if user is None:
        raise _credentials_exc
    return user
