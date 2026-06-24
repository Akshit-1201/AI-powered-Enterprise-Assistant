"""Test setup: point the app at a throwaway SQLite db before any app import.

Setting DB_PATH here (before app.config is imported) means the cached Settings and the
SQLAlchemy engine bind to a temp file, so tests never touch the real app.db.
"""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="eai-test-")
os.environ["DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["CHROMA_PATH"] = os.path.join(_tmp, "chroma")
# Force "no key" in tests so nothing ever hits real OpenAI (the guardrail adds an LLM
# call); tests stub get_chat_llm / embeddings where they need model behavior.
os.environ["OPENAI_API_KEY"] = ""

import pytest


@pytest.fixture
def default_user():
    """A persisted user used to satisfy auth in tests that aren't auth-focused."""
    from app.db.database import SessionLocal, init_db
    from app.db.models import User

    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email="tester@example.com").first()
        if user is None:
            user = User(email="tester@example.com", hashed_password="x")
            db.add(user)
            db.commit()
            db.refresh(user)
        db.expunge(user)  # detach so .id stays usable after the session closes
        return user
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _auth_override(request, default_user):
    """Override get_current_user so endpoint tests don't each need a real token.

    Tests marked ``@pytest.mark.realauth`` exercise the real JWT flow instead (the
    override is not installed for them).
    """
    if "realauth" in request.keywords:
        yield
        return
    from app.auth.dependencies import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: default_user
    yield
    app.dependency_overrides.pop(get_current_user, None)
