"""Request-scoped identity.

This is the single seam for user scoping (plan D8). In Phase 1 it holds a placeholder
id; in Phase 3 the auth layer sets the authenticated user's id here. Tools and graph
nodes read the current user via ``get_current_user_id`` rather than threading it through
every signature.
"""
from contextvars import ContextVar

from app.config import get_settings

_current_user_id: ContextVar[str] = ContextVar(
    "current_user_id", default=get_settings().placeholder_user_id
)


def set_current_user_id(user_id: str) -> None:
    _current_user_id.set(user_id)


def get_current_user_id() -> str:
    return _current_user_id.get()
