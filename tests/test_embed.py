"""The embedding service: prefixes, model loading, blobs, hashes and section reuse.

No test loads the real model: `sentence_transformers` is replaced by a stub
module, and section reuse runs on the deterministic `FakeEmbedder`.
"""

import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.green.counters import read_counters
from app.models import CandidateProfile, Section, SectionType
from app.scoring.embed import (
    EMBED_BATCH_SIZE,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    EmbedderNotLoadedError,
    EmptySectionTextError,
    MalformedEmbeddingBlobError,
    SentenceTransformerEmbedder,
    UnexpectedEmbeddingShapeError,
    blob_to_embedding,
    compute_text_hash,
    create_embedder,
    embedding_to_blob,
    ensure_section_embeddings,
)
from conftest import FakeEmbedder

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "stub/e5-model"
DEVICE = "cpu"
STUB_DIMENSION = 4
NOW = datetime(2026, 1, 1, tzinfo=UTC)
TEXT_ALPHA = "Built REST APIs in Python."
TEXT_BETA = "Maintained PostgreSQL databases."
TEXT_GAMMA = "Deployed services with Docker."
TEXT_DELTA = "Wrote unit tests for billing."
SUBPROCESS_TIMEOUT_SECONDS = 120
HEAVY_MODULES = ("sentence_transformers", "torch")


class StubSentenceTransformer:
    """Stands in for `sentence_transformers.SentenceTransformer`."""

    instances: list["StubSentenceTransformer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        StubSentenceTransformer.instances.append(self)

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Record the call; return float64 so the embedder's cast is visible."""
        self.encode_calls.append((list(sentences), kwargs))
        return np.ones((len(sentences), STUB_DIMENSION), dtype=np.float64)

    def get_sentence_embedding_dimension(self) -> int:
        """Embedding length of the stub."""
        return STUB_DIMENSION


@pytest.fixture
def stub_library(monkeypatch: pytest.MonkeyPatch) -> type[StubSentenceTransformer]:
    """Replace the `sentence_transformers` module with the stub for one test."""
    StubSentenceTransformer.instances = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return StubSentenceTransformer


@pytest.fixture
def loaded_embedder(
    stub_library: type[StubSentenceTransformer],
) -> SentenceTransformerEmbedder:
    """An embedder whose stubbed model is loaded."""
    embedder = SentenceTransformerEmbedder(MODEL_NAME, DEVICE)
    embedder.load()
    return embedder


# --- SentenceTransformerEmbedder -------------------------------------------------


def test_embedding_before_load_raises_a_typed_error() -> None:
    """Both embed methods refuse to run before `load()`."""
    embedder = SentenceTransformerEmbedder(MODEL_NAME, DEVICE)
    with pytest.raises(EmbedderNotLoadedError, match="load"):
        embedder.embed_queries(["anything"])
    with pytest.raises(EmbedderNotLoadedError, match="load"):
        embedder.embed_passages(["anything"])


def test_model_version_is_known_before_load() -> None:
    """Hashes need the version at any time, so it does not depend on `load()`."""
    assert SentenceTransformerEmbedder(MODEL_NAME, DEVICE).model_version == MODEL_NAME


def test_load_builds_the_model_with_name_and_device_only(
    stub_library: type[StubSentenceTransformer],
) -> None:
    """The model gets the configured name and device, and no remote code flag."""
    SentenceTransformerEmbedder(MODEL_NAME, DEVICE).load()
    (instance,) = stub_library.instances
    assert instance.init_args == (MODEL_NAME,)
    assert instance.init_kwargs == {"device": DEVICE}


def test_second_load_does_not_rebuild_the_model(
    stub_library: type[StubSentenceTransformer],
) -> None:
    """`load()` is idempotent."""
    embedder = SentenceTransformerEmbedder(MODEL_NAME, DEVICE)
    embedder.load()
    embedder.load()
    assert len(stub_library.instances) == 1


def test_queries_get_the_query_prefix(
    loaded_embedder: SentenceTransformerEmbedder,
    stub_library: type[StubSentenceTransformer],
) -> None:
    """The prefix is added inside the embedder."""
    loaded_embedder.embed_queries(["rest apis", "sql"])
    sentences, _ = stub_library.instances[0].encode_calls[0]
    assert sentences == [f"{QUERY_PREFIX}rest apis", f"{QUERY_PREFIX}sql"]
    assert QUERY_PREFIX == "query: "


def test_passages_get_the_passage_prefix(
    loaded_embedder: SentenceTransformerEmbedder,
    stub_library: type[StubSentenceTransformer],
) -> None:
    """The prefix is added inside the embedder."""
    loaded_embedder.embed_passages(["built apis"])
    sentences, _ = stub_library.instances[0].encode_calls[0]
    assert sentences == [f"{PASSAGE_PREFIX}built apis"]
    assert PASSAGE_PREFIX == "passage: "


@pytest.mark.parametrize("method_name", ["embed_queries", "embed_passages"])
def test_encode_options(
    method_name: str,
    loaded_embedder: SentenceTransformerEmbedder,
    stub_library: type[StubSentenceTransformer],
) -> None:
    """Vectors are normalized, batched with the named size and shown no progress bar."""
    getattr(loaded_embedder, method_name)(["text"])
    _, options = stub_library.instances[0].encode_calls[0]
    assert options["normalize_embeddings"] is True
    assert options["batch_size"] == EMBED_BATCH_SIZE
    assert options["show_progress_bar"] is False


@pytest.mark.parametrize("method_name", ["embed_queries", "embed_passages"])
def test_result_is_a_two_dimensional_float32_matrix(
    method_name: str, loaded_embedder: SentenceTransformerEmbedder
) -> None:
    """Whatever the library returns, callers get float32 with one row per text."""
    matrix = getattr(loaded_embedder, method_name)(["one", "two", "three"])
    assert matrix.dtype == np.float32
    assert matrix.shape == (3, STUB_DIMENSION)


def test_empty_input_gives_an_empty_matrix_without_calling_the_model(
    loaded_embedder: SentenceTransformerEmbedder,
    stub_library: type[StubSentenceTransformer],
) -> None:
    """No texts, no model call, but the matrix keeps the embedding width."""
    matrix = loaded_embedder.embed_passages([])
    assert matrix.shape == (0, STUB_DIMENSION)
    assert matrix.dtype == np.float32
    assert stub_library.instances[0].encode_calls == []


def test_create_embedder_uses_the_settings(test_settings: Settings) -> None:
    """The factory reads `EMBED_MODEL` and `EMBED_DEVICE`."""
    embedder = create_embedder(test_settings)
    assert embedder.model_version == test_settings.embed_model


def test_importing_the_application_does_not_import_the_model_library() -> None:
    """A fresh interpreter that imports the app and builds an embedder stays light."""
    script = (
        "import sys\n"
        "from app.main import create_app\n"
        "from app.config import Settings\n"
        "from app.scoring.embed import create_embedder\n"
        "settings = Settings(_env_file=None, hash_secret='x', employer_token='y')\n"
        "create_app(settings)\n"
        "create_embedder(settings)\n"
        f"loaded = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_test_session_never_imported_the_model_library() -> None:
    """The stub fixture is not active here, so the real library must be absent."""
    assert "sentence_transformers" not in sys.modules


# --- blobs and hashes ------------------------------------------------------------


def test_blob_round_trip_is_exact_float32() -> None:
    """Bytes in, the same float32 values out, and the result is writable."""
    vector = np.array([0.25, -1.5, 3.0e-3, 7.0], dtype=np.float32)
    blob = embedding_to_blob(vector)
    restored = blob_to_embedding(blob)
    assert len(blob) == vector.size * 4
    assert restored.dtype == np.float32
    assert np.array_equal(restored, vector)
    restored[0] = 1.0


def test_blob_conversion_casts_float64_to_float32() -> None:
    """Float64 input is stored as float32."""
    blob = embedding_to_blob(np.array([0.5, 0.25], dtype=np.float64))
    assert len(blob) == 2 * 4


def test_embedding_to_blob_rejects_a_matrix() -> None:
    """Only one vector at a time is serialized."""
    with pytest.raises(ValueError, match="1-D"):
        embedding_to_blob(np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize("blob", [b"", b"\x00", b"\x00" * 7, b"\x00" * 13])
def test_malformed_blob_length_is_rejected(blob: bytes) -> None:
    """Empty blobs and lengths that are not a multiple of four are errors."""
    with pytest.raises(MalformedEmbeddingBlobError):
        blob_to_embedding(blob)


def test_text_hash_differs_per_model_version() -> None:
    """The same text under another model gets another hash."""
    assert compute_text_hash("model-a", TEXT_ALPHA) != compute_text_hash(
        "model-b", TEXT_ALPHA
    )


def test_text_hash_is_stable_across_case_and_whitespace() -> None:
    """Normalization makes case, spacing and punctuation irrelevant."""
    reference = compute_text_hash("model-a", "Built REST APIs in Python.")
    assert compute_text_hash("model-a", "  built   rest\napis IN python!! ") == reference


def test_text_hash_is_a_sha256_hex_digest() -> None:
    """64 lowercase hex characters."""
    digest = compute_text_hash("model-a", TEXT_ALPHA)
    assert len(digest) == 64
    assert digest == digest.lower()


def test_text_hash_rejects_text_that_normalizes_to_nothing() -> None:
    """Punctuation-only text has nothing to hash."""
    with pytest.raises(EmptySectionTextError):
        compute_text_hash("model-a", " ?! ... ")


# --- section reuse ---------------------------------------------------------------


def add_profile(session: Session, raw_hash: str, texts: list[str]) -> list[Section]:
    """Persist a profile with experience sections for the given texts."""
    profile = CandidateProfile(
        raw_hash=raw_hash,
        profile_json={},
        years_experience=None,
        languages=[],
        location=None,
        parser_version="1",
        created_at=NOW,
        confirmed_at=None,
        last_used_at=NOW,
    )
    profile.sections = [
        Section(type=SectionType.EXPERIENCE, ordinal=ordinal, text=text)
        for ordinal, text in enumerate(texts)
    ]
    session.add(profile)
    session.flush()
    return profile.sections


def expected_blob(text: str, model_version: str | None = None) -> bytes:
    """The blob the fake embedder produces for a passage."""
    embedder = FakeEmbedder() if model_version is None else FakeEmbedder(model_version)
    return embedding_to_blob(embedder.embed_passages([text])[0])


def test_first_confirmation_computes_every_section(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Fresh sections get hash, vector and model version; the counter shows them."""
    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    report = ensure_section_embeddings(demo_session, sections, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed) == (0, 2)
    for section in sections:
        assert section.text_hash == compute_text_hash(
            fake_embedder.model_version, section.text
        )
        assert section.embedding == expected_blob(section.text)
        assert section.model_version == fake_embedder.model_version
    snapshot = read_counters(demo_session)
    assert (snapshot.embed_cache_hits, snapshot.embeddings_computed) == (0, 2)


def test_second_confirmation_with_same_text_computes_nothing(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """T-06 acceptance: counter and call count prove no embedding was computed."""
    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    ensure_section_embeddings(demo_session, sections, fake_embedder)
    row_ids = [section.id for section in sections]
    embedded_before = fake_embedder.passage_texts_embedded
    report = ensure_section_embeddings(demo_session, sections, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed) == (2, 0)
    assert fake_embedder.passage_texts_embedded == embedded_before == 2
    assert len(fake_embedder.passage_calls) == 1
    assert [section.id for section in sections] == row_ids
    snapshot = read_counters(demo_session)
    assert (snapshot.embed_cache_hits, snapshot.embeddings_computed) == (2, 2)


def test_reuse_across_two_profiles_is_a_cache_hit_with_identical_bytes(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Another candidate with the same text reuses the stored vector."""
    first = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    ensure_section_embeddings(demo_session, first, fake_embedder)
    demo_session.commit()
    second = add_profile(demo_session, "b" * 8, [TEXT_ALPHA, TEXT_BETA])
    report = ensure_section_embeddings(demo_session, second, fake_embedder)
    demo_session.commit()
    demo_session.expire_all()
    assert (report.cache_hits, report.embeddings_computed) == (2, 0)
    assert fake_embedder.passage_texts_embedded == 2
    stored = demo_session.scalars(select(Section).order_by(Section.id)).all()
    assert [section.embedding for section in stored[:2]] == [
        section.embedding for section in stored[2:]
    ]
    assert read_counters(demo_session).embed_cache_hits == 2


def test_duplicate_texts_in_one_batch_are_embedded_once(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Two sections with equal normalized text share one computed vector."""
    sections = add_profile(
        demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA, TEXT_ALPHA.upper()]
    )
    report = ensure_section_embeddings(demo_session, sections, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed, report.deduplicated) == (
        0,
        2,
        1,
    )
    assert fake_embedder.passage_calls == [[TEXT_ALPHA, TEXT_BETA]]
    assert sections[0].embedding == sections[2].embedding
    assert read_counters(demo_session).embeddings_computed == 2


def test_mixed_cached_and_new_sections_keep_their_own_vectors_in_order(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Interleaved hits and misses are each assigned to the right section."""
    first = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    ensure_section_embeddings(demo_session, first, fake_embedder)
    texts = [TEXT_GAMMA, TEXT_ALPHA, TEXT_DELTA, TEXT_BETA]
    second = add_profile(demo_session, "b" * 8, texts)
    report = ensure_section_embeddings(demo_session, second, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed) == (2, 2)
    assert fake_embedder.passage_calls[-1] == [TEXT_GAMMA, TEXT_DELTA]
    assert [section.text for section in second] == texts
    assert [section.embedding for section in second] == [
        expected_blob(text) for text in texts
    ]


def test_a_changed_section_is_recomputed_and_the_rest_keep_their_rows(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Only the edited section gets a new vector."""
    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    ensure_section_embeddings(demo_session, sections, fake_embedder)
    unchanged, edited = sections
    old_hash = edited.text_hash
    edited.text = TEXT_GAMMA
    report = ensure_section_embeddings(demo_session, sections, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed) == (1, 1)
    assert fake_embedder.passage_calls[-1] == [TEXT_GAMMA]
    assert edited.text_hash != old_hash
    assert edited.embedding == expected_blob(TEXT_GAMMA)
    assert unchanged.embedding == expected_blob(TEXT_ALPHA)


def test_a_model_version_change_forces_recomputation(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """A new version never reuses vectors of the old one, even for equal text."""
    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    ensure_section_embeddings(demo_session, sections, fake_embedder)
    old_hashes = [section.text_hash for section in sections]
    successor = FakeEmbedder(model_version="fake-embedder-v2")
    report = ensure_section_embeddings(demo_session, sections, successor)
    assert (report.cache_hits, report.embeddings_computed) == (0, 2)
    assert [section.text_hash for section in sections] != old_hashes
    assert all(section.model_version == "fake-embedder-v2" for section in sections)
    assert sections[0].embedding == expected_blob(TEXT_ALPHA, "fake-embedder-v2")


def test_a_hash_without_a_vector_is_not_a_cache_source(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Only sections that hold an embedding can feed the global lookup."""
    hash_only = add_profile(demo_session, "a" * 8, [TEXT_ALPHA])
    hash_only[0].text_hash = compute_text_hash(fake_embedder.model_version, TEXT_ALPHA)
    fresh = add_profile(demo_session, "b" * 8, [TEXT_ALPHA])
    report = ensure_section_embeddings(demo_session, fresh, fake_embedder)
    assert (report.cache_hits, report.embeddings_computed) == (0, 1)


def test_empty_section_fails_fast_before_anything_changes(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """One empty section stops the whole call; no text appears in the message."""
    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, " ?! ... "])
    with pytest.raises(EmptySectionTextError) as error_info:
        ensure_section_embeddings(demo_session, sections, fake_embedder)
    assert TEXT_ALPHA not in str(error_info.value)
    assert "?!" not in str(error_info.value)
    assert fake_embedder.passage_calls == []
    assert all(section.embedding is None for section in sections)
    assert read_counters(demo_session).embeddings_computed == 0


def test_wrong_row_count_from_the_embedder_is_rejected(
    demo_session: Session,
) -> None:
    """A broken provider cannot silently misalign vectors and sections."""

    class ShortEmbedder(FakeEmbedder):
        def embed_passages(self, texts: Any) -> Any:
            return super().embed_passages(texts)[:-1]

    sections = add_profile(demo_session, "a" * 8, [TEXT_ALPHA, TEXT_BETA])
    with pytest.raises(UnexpectedEmbeddingShapeError):
        ensure_section_embeddings(demo_session, sections, ShortEmbedder())
    assert all(section.embedding is None for section in sections)


def test_no_sections_is_a_no_op(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """An empty list embeds nothing and counts nothing."""
    report = ensure_section_embeddings(demo_session, [], fake_embedder)
    assert (report.cache_hits, report.embeddings_computed, report.deduplicated) == (
        0,
        0,
        0,
    )
    assert fake_embedder.passage_calls == []
