"""The posting service: one embedding batch, validation, and filter normalization."""

from datetime import UTC, date, datetime

import pytest
from conftest import FakeEmbedder
from pipeline_support import ControlledEmbedder, RecordingTracker, build_cv_pdf
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import JobPosting, JobStatus, Section
from app.pipeline import (
    ConfirmInput,
    ConfirmOutcomeKind,
    PipelineContext,
    ProfileEdits,
    SectionEdit,
    UploadInput,
    confirm_draft,
    submit_upload,
)
from app.postings import (
    InvalidPostingError,
    PostingInput,
    RequirementInput,
    create_posting,
    create_postings,
)
from app.privacy.reasons import StoredReasonCode
from app.scoring.embed import UnexpectedEmbeddingShapeError, blob_to_embedding

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
REFERENCE_DATE = date(2026, 1, 1)
TWO_REQUIREMENTS = (RequirementInput("Python programming", 3), RequirementInput("SQL", 1))


def make_input(**changes: object) -> PostingInput:
    """A valid posting input with fields overridden by `changes`."""
    fields: dict[str, object] = {
        "title": "Backend Developer",
        "description": "Build services.",
        "requirements": TWO_REQUIREMENTS,
        "created_at": NOW,
    }
    fields.update(changes)
    return PostingInput(**fields)  # type: ignore[arg-type]


def test_requirements_are_embedded_in_one_batch_with_model_version(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """One `embed_queries` call holds every requirement text; vectors and version are stored."""
    posting = create_posting(demo_session, fake_embedder, make_input())

    assert fake_embedder.query_calls == [["Python programming", "SQL"]]
    assert fake_embedder.passage_calls == []
    first = posting.requirements[0]
    expected = fake_embedder.embed_queries(["Python programming"])[0]
    assert blob_to_embedding(first.embedding).tolist() == pytest.approx(expected.tolist())
    assert [item.model_version for item in posting.requirements] == [
        fake_embedder.model_version
    ] * 2
    assert [item.weight for item in posting.requirements] == [3, 1]
    assert posting.status is JobStatus.OPEN
    assert posting.created_at == NOW


def test_several_postings_share_a_single_embedding_call(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """The seed relies on this: all postings of one call, one batch."""
    inputs = [make_input(title="A"), make_input(title="B")]
    postings = create_postings(demo_session, fake_embedder, inputs)

    assert len(postings) == 2
    assert len(fake_embedder.query_calls) == 1
    assert len(fake_embedder.query_calls[0]) == 4


def test_no_postings_means_no_embedding_call(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Nothing to create, nothing to embed."""
    assert create_postings(demo_session, fake_embedder, []) == []
    assert fake_embedder.query_calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "   "},
        {"requirements": ()},
        {"requirements": (RequirementInput("   ", 2),)},
        {"requirements": (RequirementInput("SQL", 0),)},
        {"requirements": (RequirementInput("SQL", 4),)},
    ],
)
def test_invalid_postings_fail_before_embedding_or_storing(
    demo_session: Session, fake_embedder: FakeEmbedder, changes: dict[str, object]
) -> None:
    """Fail fast: no embedding call and no row for a posting that breaks an invariant."""
    with pytest.raises(InvalidPostingError):
        create_posting(demo_session, fake_embedder, make_input(**changes))
    assert fake_embedder.query_calls == []
    assert demo_session.scalars(select(JobPosting)).all() == []


@pytest.mark.parametrize("weight", [1, 3])
def test_weight_bounds_are_accepted(
    demo_session: Session, fake_embedder: FakeEmbedder, weight: int
) -> None:
    """Weights 1 and 3 are the inclusive bounds."""
    requirements = (RequirementInput("SQL", weight),)
    posting = create_posting(demo_session, fake_embedder, make_input(requirements=requirements))
    assert posting.requirements[0].weight == weight


def test_wrong_vector_count_stores_nothing(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """A misbehaving embedder is caught before anything is added."""
    fake_embedder.embed_queries = lambda texts: fake_embedder._vectors("q", texts[:-1])  # type: ignore[method-assign]
    with pytest.raises(UnexpectedEmbeddingShapeError):
        create_posting(demo_session, fake_embedder, make_input())
    assert demo_session.scalars(select(JobPosting)).all() == []


def test_filter_values_are_normalized_like_candidate_values(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """Folded location, English language names, blank location becomes `None`."""
    posting = create_posting(
        demo_session,
        fake_embedder,
        make_input(required_languages=["İngilizce", " ALMANCA ", "english"], location=" DENİZLİ "),
    )
    assert posting.location == "denizli"
    assert posting.required_languages == ["english", "german"]
    blank = create_posting(demo_session, fake_embedder, make_input(location="  "))
    assert blank.location is None


@pytest.fixture
def pipeline_context(demo_session: Session, test_settings: Settings) -> PipelineContext:
    """A pipeline context with the marker-driven embedder (no model)."""
    return PipelineContext(
        session=demo_session,
        settings=test_settings,
        embedder=ControlledEmbedder(),
        tracker=RecordingTracker(),
        now=NOW,
        reference_date=REFERENCE_DATE,
    )


def apply_with_values(
    context: PipelineContext, job_id: int, location: str, languages: list[str]
):
    """Upload a CV, then confirm it with the candidate's location and languages."""
    outcome = submit_upload(context, UploadInput(job_id, build_cv_pdf("normalization")))
    profile_id = context.session.scalars(select(Section.profile_id)).first()
    sections = context.session.scalars(
        select(Section).where(Section.profile_id == profile_id).order_by(Section.ordinal)
    ).all()
    edits = ProfileEdits(
        sections=[SectionEdit(section.type, section.text) for section in sections],
        years_experience=8.0,
        languages=languages,
        location=location,
    )
    return confirm_draft(context, ConfirmInput(outcome.draft_token, edits))


@pytest.mark.parametrize(
    ("posting_location", "posting_languages", "candidate_location", "candidate_languages"),
    [
        ("Denizli", ["english"], "DENİZLİ", ["İngilizce"]),
        ("DENİZLİ", ["İngilizce"], "Denizli", ["english"]),
        ("IZMIR", ["Turkish"], "İzmir", ["Türkçe"]),
    ],
)
def test_posting_matches_a_candidate_normalized_by_the_pipeline(
    pipeline_context: PipelineContext,
    posting_location: str,
    posting_languages: list[str],
    candidate_location: str,
    candidate_languages: list[str],
) -> None:
    """The exact-match SQL filters pass: both sides were folded the same way."""
    posting = create_posting(
        pipeline_context.session,
        pipeline_context.embedder,
        make_input(
            location=posting_location,
            required_languages=posting_languages,
            remote_ok=False,
        ),
    )
    result = apply_with_values(
        pipeline_context, posting.id, candidate_location, candidate_languages
    )
    assert result.kind is ConfirmOutcomeKind.RANKED


@pytest.mark.parametrize(
    ("posting_location", "posting_languages", "reason"),
    [
        ("İzmir", ["english"], StoredReasonCode.LOCATION_MISMATCH),
        ("Denizli", ["almanca"], StoredReasonCode.LANG_MISSING),
    ],
)
def test_genuine_mismatches_still_eliminate(
    pipeline_context: PipelineContext,
    posting_location: str,
    posting_languages: list[str],
    reason: StoredReasonCode,
) -> None:
    """Normalization does not make the filters pass everything."""
    posting = create_posting(
        pipeline_context.session,
        pipeline_context.embedder,
        make_input(
            location=posting_location, required_languages=posting_languages, remote_ok=False
        ),
    )
    result = apply_with_values(pipeline_context, posting.id, "DENİZLİ", ["İngilizce"])
    assert result.kind is ConfirmOutcomeKind.ELIMINATED
    assert result.reason_code is reason
