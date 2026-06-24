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


_OLD_CONV_META = "_conversation_meta_old"


def _rename_stale_conversation_meta() -> None:
    """If an existing DB has the pre-`title` / pre-composite-PK `conversation_meta`, rename
    it aside so create_all rebuilds the current schema (we copy the rows back afterward).

    This is a tiny hand-rolled migration in lieu of Alembic — `conversation_meta` is only a
    session index, and the actual chat history lives in the checkpointer tables, so this is
    safe and non-destructive. Idempotent: a fresh or already-current DB is left alone."""
    with engine.begin() as conn:
        info = conn.exec_driver_sql("PRAGMA table_info(conversation_meta)").fetchall()
        columns = [row[1] for row in info]
        if info and "title" not in columns:  # stale schema present
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {_OLD_CONV_META}")
            conn.exec_driver_sql(f"ALTER TABLE conversation_meta RENAME TO {_OLD_CONV_META}")


def _restore_conversation_meta_rows() -> None:
    """Copy the renamed old session index into the rebuilt table, then drop the old one."""
    with engine.begin() as conn:
        exists = conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_OLD_CONV_META,),
        ).fetchall()
        if not exists:
            return
        conn.exec_driver_sql(
            "INSERT OR IGNORE INTO conversation_meta "
            "(user_id, session_id, title, created_at, last_active) "
            f"SELECT user_id, session_id, NULL, created_at, last_active FROM {_OLD_CONV_META}"
        )
        conn.exec_driver_sql(f"DROP TABLE {_OLD_CONV_META}")


def init_db() -> None:
    """Create tables. Import models so they register on Base before create_all."""
    from app.db import models  # noqa: F401

    _rename_stale_conversation_meta()
    Base.metadata.create_all(bind=engine)
    _restore_conversation_meta_rows()
