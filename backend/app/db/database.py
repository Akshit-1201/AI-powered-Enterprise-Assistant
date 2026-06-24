"""SQLAlchemy engine, session factory, and Base for the relational store."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

_settings = get_settings()

# check_same_thread=False: FastAPI may touch the session from worker threads.
engine = create_engine(
    f"sqlite:///{_settings.db_path}",
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record):
    # WAL improves concurrent read/write; busy_timeout avoids spurious "database is
    # locked" errors under light concurrency.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Create tables. Import models so they register on Base before create_all."""
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
