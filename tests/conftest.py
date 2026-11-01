"""Shared fixtures. Tests never read the real `.env`."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from app.config import Settings
from app.db import create_db_engine

DUMMY_HASH_SECRET = "test-hash-secret"
DUMMY_EMPLOYER_TOKEN = "test-employer-token"
TEST_DATABASE_DIRECTORY = "data"
TEST_DATABASE_FILE = "test.db"


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Explicit settings with a temp-path SQLite database and dummy secrets."""
    database_path = tmp_path / TEST_DATABASE_DIRECTORY / TEST_DATABASE_FILE
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{database_path.as_posix()}",
        hash_secret=DUMMY_HASH_SECRET,
        employer_token=DUMMY_EMPLOYER_TOKEN,
    )


@pytest.fixture
def engine(test_settings: Settings) -> Iterator[Engine]:
    """Engine for the test database, disposed so Windows can delete the file."""
    test_engine = create_db_engine(test_settings.database_url)
    yield test_engine
    test_engine.dispose()
