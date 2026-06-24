"""Request-scoped identity.

This is the single seam for user scoping (plan D8): the auth layer sets the authenticated
user's id here, and tools and graph nodes read it via ``get_current_user_id`` rather than
threading it through every signature.

There is deliberately NO default. Reading the id before ``set_current_user_id`` ran means
a tool/node is executing outside a request context — a bug we want to surface LOUDLY
(RuntimeError) rather than silently attributing another tenant's data to a phantom user.
Multi-tenancy must never degrade quietly (P0.4).
"""
from contextvars import ContextVar

_current_user_id: ContextVar[str] = ContextVar("current_user_id")


def set_current_user_id(user_id: str) -> None:
    _current_user_id.set(user_id)


def get_current_user_id() -> str:
    try:
        return _current_user_id.get()
    except LookupError as exc:
        raise RuntimeError(
            "Current user id is not set in this request context. set_current_user_id() must "
            "run before any tool or graph node reads it (the multi-tenancy seam)."
        ) from exc
