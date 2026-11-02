"""Shared fixtures. Tests never read the real `.env` and never load the real model."""

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import create_db_engine, create_demo_schema, create_session_factory
from app.scoring.embed import PASSAGE_PREFIX, QUERY_PREFIX, FloatMatrix

DUMMY_HASH_SECRET = "test-hash-secret"
DUMMY_EMPLOYER_TOKEN = "test-employer-token"
TEST_DATABASE_DIRECTORY = "data"
TEST_DATABASE_FILE = "test.db"
FAKE_EMBEDDING_DIMENSION = 8
FAKE_MODEL_VERSION = "fake-embedder-v1"
SEED_BYTES = 8


class FakeEmbedder:
    """Deterministic `EmbeddingProvider` that needs no GPU and no model.

    A text always maps to the same unit vector; queries and passages of the same
    text map to different vectors, so a missing prefix would be noticed. Every
    call is recorded, which lets tests prove that nothing was recomputed.

    Args:
        model_version: Reported model version (part of every `text_hash`).
        dimension: Length of the produced vectors.
    """

    def __init__(
        self,
        model_version: str = FAKE_MODEL_VERSION,
        dimension: int = FAKE_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_version = model_version
        self.dimension = dimension
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    @property
    def passage_texts_embedded(self) -> int:
        """Total number of passages embedded over all calls."""
        return sum(len(call) for call in self.passage_calls)

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Record the call and return one deterministic vector per text."""
        self.query_calls.append(list(texts))
        return self._vectors(QUERY_PREFIX, texts)

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Record the call and return one deterministic vector per text."""
        self.passage_calls.append(list(texts))
        return self._vectors(PASSAGE_PREFIX, texts)

    def _vectors(self, prefix: str, texts: Sequence[str]) -> FloatMatrix:
        """Stack the unit vector of every prefixed text into a float32 matrix."""
        rows = [self._vector(prefix + text) for text in texts]
        if not rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack(rows)

    def _vector(self, prefixed_text: str) -> FloatMatrix:
        """A unit vector seeded by the model version and the text."""
        digest = hashlib.sha256(
            (self.model_version + prefixed_text).encode("utf-8")
        ).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:SEED_BYTES], "big"))
        vector = generator.standard_normal(self.dimension)
        return (vector / np.linalg.norm(vector)).astype(np.float32)


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


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A fresh deterministic embedder with empty call records."""
    return FakeEmbedder()


@pytest.fixture
def demo_session(engine: Engine) -> Iterator[Session]:
    """A session on the test database with the demo schema created."""
    create_demo_schema(engine)
    with create_session_factory(engine)() as test_session:
        yield test_session
