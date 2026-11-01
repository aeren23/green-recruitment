"""SQLite pragmas must be active on every connection."""

from pathlib import Path

from sqlalchemy import Connection, Engine, text

from app.config import Settings

ENABLED = 1


def read_pragma(connection: Connection, pragma_name: str) -> int:
    """Return the integer value of a SQLite pragma on the given connection."""
    return connection.execute(text(f"PRAGMA {pragma_name}")).scalar_one()


def test_pragmas_are_on_for_first_and_second_connection(engine: Engine) -> None:
    """Both pragmas are 1 on two distinct connections of the same engine."""
    with engine.connect() as first_connection, engine.connect() as second_connection:
        assert first_connection.connection.dbapi_connection is not (
            second_connection.connection.dbapi_connection
        )
        for connection in (first_connection, second_connection):
            assert read_pragma(connection, "foreign_keys") == ENABLED
            assert read_pragma(connection, "secure_delete") == ENABLED


def test_database_directory_is_created(
    engine: Engine, test_settings: Settings
) -> None:
    """The missing parent directory of the database file is created."""
    database_file = Path(test_settings.database_url.removeprefix("sqlite:///"))
    assert database_file.parent.is_dir()
