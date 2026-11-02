"""Seeding of the three postings of SPEC 7.2."""

from datetime import UTC, datetime

import pytest
from conftest import FakeEmbedder
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import JobPosting, JobStatus, Requirement
from app.scoring.embed import UnexpectedEmbeddingShapeError, blob_to_embedding
from app.seed import (
    ACCOUNTANT_TITLE,
    BACKEND_DEVELOPER_TITLE,
    BENCHMARK_FILTERS,
    HR_SPECIALIST_TITLE,
    SEED_POSTINGS,
    PostingFilters,
    SeedOptions,
    seed_postings,
)

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
REQUIREMENTS_PER_POSTING = 4
EXPECTED_WEIGHTS = [3, 2, 2, 1]
EXPECTED_MIN_YEARS = {BACKEND_DEVELOPER_TITLE: 2, ACCOUNTANT_TITLE: 3, HR_SPECIALIST_TITLE: 2}
EXPECTED_CATEGORIES = {
    BACKEND_DEVELOPER_TITLE: ["INFORMATION-TECHNOLOGY"],
    ACCOUNTANT_TITLE: ["ACCOUNTANT", "FINANCE"],
    HR_SPECIALIST_TITLE: ["HR"],
}
DEMO_FILTERS = PostingFilters(
    required_languages=("english",), location="denizli", remote_ok=False
)


def count_rows(session: Session, model: type) -> int:
    """Number of rows of a model."""
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_three_postings_with_weighted_requirements(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Titles, categories, minimum years and requirement weights follow SPEC 7.2."""
    postings = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))

    assert [posting.title for posting in postings] == [seed.title for seed in SEED_POSTINGS]
    for posting in postings:
        assert posting.status is JobStatus.OPEN
        assert posting.description
        assert posting.min_years == EXPECTED_MIN_YEARS[posting.title]
        assert posting.relevant_categories == EXPECTED_CATEGORIES[posting.title]
        assert [item.weight for item in posting.requirements] == EXPECTED_WEIGHTS
        assert len(posting.requirements) == REQUIREMENTS_PER_POSTING
        assert posting.created_at == NOW


def test_requirements_are_embedded_in_one_query_batch(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """One `embed_queries` call, vectors and model version stored."""
    postings = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))
    assert len(fake_embedder.query_calls) == 1
    assert len(fake_embedder.query_calls[0]) == len(SEED_POSTINGS) * REQUIREMENTS_PER_POSTING
    first = postings[0].requirements[0]
    stored = blob_to_embedding(first.embedding)
    expected = fake_embedder.embed_queries([first.text])[0]
    assert stored.tolist() == pytest.approx(expected.tolist())
    assert first.model_version == fake_embedder.model_version


def test_benchmark_mode_switches_language_and_location_filters_off(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """The default values are the benchmark ones."""
    postings = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))
    assert BENCHMARK_FILTERS == PostingFilters()
    for posting in postings:
        assert posting.required_languages == []
        assert posting.location is None
        assert posting.remote_ok is True


def test_second_call_creates_nothing_and_embeds_nothing(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Idempotent: same postings, no new rows, no new embedding call."""
    first = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))
    second = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))
    assert [posting.id for posting in second] == [posting.id for posting in first]
    assert count_rows(demo_session, JobPosting) == len(SEED_POSTINGS)
    assert count_rows(demo_session, Requirement) == len(SEED_POSTINGS) * REQUIREMENTS_PER_POSTING
    assert len(fake_embedder.query_calls) == 1


def test_only_missing_postings_are_created_and_embedded(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """With one posting already stored, the batch holds the other two only."""
    last_id = seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))[-1].id
    demo_session.expunge_all()
    demo_session.execute(Requirement.__table__.delete().where(Requirement.job_id == last_id))
    demo_session.execute(JobPosting.__table__.delete().where(JobPosting.id == last_id))
    fake_embedder.query_calls.clear()

    seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))

    assert len(fake_embedder.query_calls) == 1
    assert len(fake_embedder.query_calls[0]) == REQUIREMENTS_PER_POSTING
    assert count_rows(demo_session, JobPosting) == len(SEED_POSTINGS)


def test_demo_filters_can_be_supplied_per_title(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """The T-12 extension point: filter values keyed by posting title."""
    options = SeedOptions(now=NOW, filters_by_title={BACKEND_DEVELOPER_TITLE: DEMO_FILTERS})
    postings = {item.title: item for item in seed_postings(demo_session, fake_embedder, options)}
    backend = postings[BACKEND_DEVELOPER_TITLE]
    assert backend.required_languages == ["english"]
    assert backend.location == "denizli"
    assert backend.remote_ok is False
    assert postings[ACCOUNTANT_TITLE].location is None


def test_wrong_number_of_vectors_fails_before_anything_is_stored(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """A misbehaving embedder is caught, not silently misaligned."""
    fake_embedder.embed_queries = lambda texts: fake_embedder._vectors("q", texts[:-1])  # type: ignore[method-assign]
    with pytest.raises(UnexpectedEmbeddingShapeError):
        seed_postings(demo_session, fake_embedder, SeedOptions(now=NOW))
    assert count_rows(demo_session, JobPosting) == 0
