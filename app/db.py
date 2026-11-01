"""SQLAlchemy engine and session factory with the SQLite pragmas required by the SPEC."""

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

SQLITE_BACKEND_NAME = "sqlite"
SQLITE_IN_MEMORY_DATABASE = ":memory:"
SQLITE_CONNECT_EVENT = "connect"
PRAGMA_FOREIGN_KEYS_ON = "PRAGMA foreign_keys=ON"
PRAGMA_SECURE_DELETE_ON = "PRAGMA secure_delete=ON"
SQLITE_PRAGMAS_ON_CONNECT = (PRAGMA_FOREIGN_KEYS_ON, PRAGMA_SECURE_DELETE_ON)
SQLITE_CONNECT_ARGUMENTS = {"check_same_thread": False}


def apply_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
    """Set the per-connection pragmas on a freshly opened SQLite connection.

    Args:
        dbapi_connection: The raw DB-API connection that was just opened.
        connection_record: SQLAlchemy pool bookkeeping object (unused).
    """
    cursor = dbapi_connection.cursor()
    try:
        for pragma_statement in SQLITE_PRAGMAS_ON_CONNECT:
            cursor.execute(pragma_statement)
    finally:
        cursor.close()


def ensure_database_directory(database_url: str) -> None:
    """Create the parent directory of a file-based SQLite database.

    SQLite creates the database file but never its directory.

    Args:
        database_url: SQLAlchemy URL of the database.
    """
    url = make_url(database_url)
    if url.get_backend_name() != SQLITE_BACKEND_NAME:
        return
    if not url.database or url.database == SQLITE_IN_MEMORY_DATABASE:
        return
    Path(url.database).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str) -> Engine:
    """Build an engine whose SQLite connections all carry the required pragmas.

    Args:
        database_url: SQLAlchemy URL of the database.

    Returns:
        A new engine. The caller owns it and must call `dispose()` when done.
    """
    ensure_database_directory(database_url)
    is_sqlite = make_url(database_url).get_backend_name() == SQLITE_BACKEND_NAME
    connect_arguments = SQLITE_CONNECT_ARGUMENTS if is_sqlite else {}
    engine = create_engine(database_url, connect_args=connect_arguments)
    if is_sqlite:
        event.listen(engine, SQLITE_CONNECT_EVENT, apply_sqlite_pragmas)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to the given engine.

    Args:
        engine: Engine created by `create_db_engine`.
    """
    return sessionmaker(bind=engine)
