"""Schema, constraint, cascade and round-trip behaviour of `app/models.py`.

Every test runs on a temp-path SQLite engine built by `create_db_engine`, so the
real pragmas (`foreign_keys=ON`) apply.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.db import (
    create_benchmark_schema,
    create_db_engine,
    create_demo_schema,
    create_session_factory,
)
from app.models import (
    BENCHMARK_TABLE_NAMES,
    DEMO_TABLE_NAMES,
    GREEN_COUNTER_ROW_ID,
    REQUIREMENT_WEIGHT_MAX,
    REQUIREMENT_WEIGHT_MIN,
    Application,
    ApplicationStatus,
    BenchmarkBase,
    BenchmarkRun,
    CandidateContact,
    CandidateProfile,
    DecisionRecord,
    DemoBase,
    GreenCounter,
    JobPosting,
    JobStatus,
    PipelineName,
    Requirement,
    Section,
    SectionType,
    StageMeasurement,
)

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(days=30)
EMBEDDING_DIMENSION = 384
MODEL_VERSION = "intfloat/multilingual-e5-small"
FIRST_HASH = "hash-first"
SECOND_HASH = "hash-second"
FIRST_TOKEN = "token-first"
SECOND_TOKEN = "token-second"
INVALID_VALUE = "bogus"
BENCHMARK_DATABASE_FILE = "bench.db"


@pytest.fixture
def demo_engine(engine: Engine) -> Engine:
    """The conftest test engine with the demo schema created."""
    create_demo_schema(engine)
    return engine


@pytest.fixture
def session(demo_engine: Engine) -> Iterator[Session]:
    """A session on the demo database, closed after the test."""
    with create_session_factory(demo_engine)() as test_session:
        yield test_session


def make_job(title: str = "Backend Developer") -> JobPosting:
    """A valid open job posting."""
    return JobPosting(
        title=title,
        description="Build services.",
        status=JobStatus.OPEN,
        created_at=NOW,
        required_languages=[],
        remote_ok=True,
        relevant_categories=["INFORMATION-TECHNOLOGY"],
    )


def make_profile(raw_hash: str = FIRST_HASH) -> CandidateProfile:
    """A valid, unconfirmed profile."""
    return CandidateProfile(
        raw_hash=raw_hash,
        profile_json={"sections": []},
        languages=["english"],
        parser_version="1",
        created_at=NOW,
        last_used_at=NOW,
    )


def make_section(ordinal: int = 0, section_type: str = SectionType.SKILLS) -> Section:
    """A valid section without embedding (as in a draft)."""
    return Section(type=section_type, ordinal=ordinal, text="python sql")


def make_application(
    job: JobPosting,
    profile: CandidateProfile,
    raw_hash: str = FIRST_HASH,
    draft_token: str = FIRST_TOKEN,
) -> Application:
    """A valid draft application with no score yet."""
    return Application(
        job_id=job.id,
        profile_id=profile.id,
        raw_hash=raw_hash,
        status=ApplicationStatus.DRAFT,
        draft_token=draft_token,
        created_at=NOW,
    )


def make_decision(
    job_id: int, application_token: str | None = None, raw_hash: str = FIRST_HASH
) -> DecisionRecord:
    """A valid decision record."""
    return DecisionRecord(
        job_id=job_id,
        raw_hash=raw_hash,
        reason_code="MIN_YEARS",
        decided_at=NOW,
        expires_at=LATER,
        application_token=application_token,
    )


def persisted_job_and_profile(session: Session) -> tuple[JobPosting, CandidateProfile]:
    """Commit one job and one profile and return them with ids assigned."""
    job = make_job()
    profile = make_profile()
    session.add_all([job, profile])
    session.commit()
    return job, profile


def count_rows(session: Session, table_name: str) -> int:
    """Number of rows in a table, read with raw SQL."""
    return session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def test_demo_and_benchmark_schemas_create_disjoint_table_sets(
    engine: Engine, tmp_path: Path
) -> None:
    """Each schema creates exactly its own tables, in separate databases."""
    benchmark_engine = create_db_engine(
        f"sqlite:///{(tmp_path / BENCHMARK_DATABASE_FILE).as_posix()}"
    )
    try:
        create_demo_schema(engine)
        create_benchmark_schema(benchmark_engine)
        assert set(inspect(engine).get_table_names()) == DEMO_TABLE_NAMES
        assert set(inspect(benchmark_engine).get_table_names()) == BENCHMARK_TABLE_NAMES
    finally:
        benchmark_engine.dispose()
    assert DEMO_TABLE_NAMES.isdisjoint(BENCHMARK_TABLE_NAMES)
    assert set(DemoBase.metadata.tables) == DEMO_TABLE_NAMES
    assert set(BenchmarkBase.metadata.tables) == BENCHMARK_TABLE_NAMES


def test_schema_creation_is_idempotent(demo_engine: Engine) -> None:
    """Creating the schema twice (an app restart) does not fail."""
    create_demo_schema(demo_engine)
    assert set(inspect(demo_engine).get_table_names()) == DEMO_TABLE_NAMES


def test_duplicate_job_and_raw_hash_is_rejected(session: Session) -> None:
    """The same raw_hash cannot be applied twice to one job."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_application(job, profile))
    session.commit()
    session.add(make_application(job, profile, draft_token=SECOND_TOKEN))
    with pytest.raises(IntegrityError):
        session.commit()


def test_same_raw_hash_on_different_job_is_allowed(session: Session) -> None:
    """The same CV may apply to a different job."""
    job, profile = persisted_job_and_profile(session)
    other_job = make_job(title="Accountant")
    session.add(other_job)
    session.commit()
    session.add(make_application(job, profile))
    session.add(make_application(other_job, profile, draft_token=SECOND_TOKEN))
    session.commit()
    assert count_rows(session, Application.__tablename__) == 2


def test_draft_token_must_be_unique(session: Session) -> None:
    """Two applications cannot share a draft_token."""
    job, profile = persisted_job_and_profile(session)
    other_job = make_job(title="Accountant")
    session.add(other_job)
    session.commit()
    session.add(make_application(job, profile))
    session.add(make_application(other_job, profile))
    with pytest.raises(IntegrityError):
        session.commit()


def test_profile_raw_hash_must_be_unique(session: Session) -> None:
    """Two profiles cannot share a raw_hash."""
    session.add_all([make_profile(), make_profile()])
    with pytest.raises(IntegrityError):
        session.commit()


def test_contact_profile_id_must_be_unique(session: Session) -> None:
    """A profile has at most one contact row."""
    profile = make_profile()
    session.add(profile)
    session.commit()
    session.add_all(
        [
            CandidateContact(profile_id=profile.id, name="A"),
            CandidateContact(profile_id=profile.id, name="B"),
        ]
    )
    with pytest.raises(IntegrityError):
        session.commit()


def make_profile_with_children() -> CandidateProfile:
    """A profile with two sections and a contact row."""
    profile = make_profile()
    profile.sections = [make_section(0), make_section(1, SectionType.EXPERIENCE)]
    profile.contact = CandidateContact(name="Ada", email="ada@example.com", phone="1")
    return profile


def test_orm_profile_delete_cascades_to_sections_and_contact(
    session: Session,
) -> None:
    """Deleting a profile through the ORM removes its sections and contact."""
    profile = make_profile_with_children()
    session.add(profile)
    session.commit()
    session.delete(profile)
    session.commit()
    assert count_rows(session, CandidateProfile.__tablename__) == 0
    assert count_rows(session, Section.__tablename__) == 0
    assert count_rows(session, CandidateContact.__tablename__) == 0


def test_raw_sql_profile_delete_cascades_in_the_database(session: Session) -> None:
    """A raw DELETE cascades too, proving ondelete=CASCADE and foreign_keys=ON."""
    profile = make_profile_with_children()
    session.add(profile)
    session.commit()
    session.execute(
        text(f"DELETE FROM {CandidateProfile.__tablename__} WHERE id = :id"),
        {"id": profile.id},
    )
    session.commit()
    assert count_rows(session, Section.__tablename__) == 0
    assert count_rows(session, CandidateContact.__tablename__) == 0


def test_profile_with_application_cannot_be_deleted_through_orm(
    session: Session,
) -> None:
    """The application-to-profile link has no cascade, so the delete fails."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_application(job, profile))
    session.commit()
    session.delete(profile)
    with pytest.raises(IntegrityError):
        session.commit()


def test_profile_with_application_cannot_be_deleted_with_raw_sql(
    session: Session,
) -> None:
    """The database itself refuses to orphan an application."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_application(job, profile))
    session.commit()
    with pytest.raises(IntegrityError):
        session.execute(
            text(f"DELETE FROM {CandidateProfile.__tablename__} WHERE id = :id"),
            {"id": profile.id},
        )


def test_invalid_job_status_is_rejected(session: Session) -> None:
    """The job status CHECK rejects unknown values."""
    job = make_job()
    job.status = INVALID_VALUE  # type: ignore[assignment]
    session.add(job)
    with pytest.raises(IntegrityError):
        session.commit()


def test_invalid_application_status_is_rejected(session: Session) -> None:
    """The application status CHECK rejects unknown values, such as the removed `eliminated`."""
    job, profile = persisted_job_and_profile(session)
    application = make_application(job, profile)
    application.status = "eliminated"  # type: ignore[assignment]
    session.add(application)
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize("invalid_type", [INVALID_VALUE, "languages"])
def test_invalid_section_type_is_rejected(
    session: Session, invalid_type: str
) -> None:
    """Section types are limited to the six embedded ones; `languages` is not one."""
    profile = make_profile()
    profile.sections = [make_section(section_type=invalid_type)]
    session.add(profile)
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    "invalid_weight", [REQUIREMENT_WEIGHT_MIN - 1, REQUIREMENT_WEIGHT_MAX + 1]
)
def test_requirement_weight_outside_bounds_is_rejected(
    session: Session, invalid_weight: int
) -> None:
    """Weights 0 and 4 violate the 1 to 3 CHECK."""
    job = make_job()
    job.requirements = [
        Requirement(
            text="SQL",
            weight=invalid_weight,
            embedding=b"x",
            model_version=MODEL_VERSION,
        )
    ]
    session.add(job)
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    "valid_weight", [REQUIREMENT_WEIGHT_MIN, REQUIREMENT_WEIGHT_MAX]
)
def test_requirement_weight_at_bounds_is_accepted(
    session: Session, valid_weight: int
) -> None:
    """Weights 1 and 3 are valid."""
    job = make_job()
    job.requirements = [
        Requirement(
            text="SQL", weight=valid_weight, embedding=b"x", model_version=MODEL_VERSION
        )
    ]
    session.add(job)
    session.commit()
    assert count_rows(session, Requirement.__tablename__) == 1


def test_green_counter_defaults_to_zero_and_allows_one_row(session: Session) -> None:
    """The single counter row starts at zero in every counter column."""
    session.add(GreenCounter())
    session.commit()
    counter = session.get_one(GreenCounter, GREEN_COUNTER_ROW_ID)
    assert counter.llm_calls == 0
    assert counter.records_deleted == 0


def test_second_green_counter_row_is_rejected(session: Session) -> None:
    """A row with any id other than the fixed one violates the CHECK."""
    session.add(GreenCounter())
    session.commit()
    session.add(GreenCounter(id=GREEN_COUNTER_ROW_ID + 1))
    with pytest.raises(IntegrityError):
        session.commit()


def test_duplicate_green_counter_id_is_rejected(session: Session) -> None:
    """Re-inserting the fixed id violates the primary key."""
    session.add(GreenCounter())
    session.commit()
    session.add(GreenCounter())
    with pytest.raises(IntegrityError):
        session.commit()


def test_application_nullable_fields_can_be_null(session: Session) -> None:
    """A draft application has no score, matrix, summary or decision time."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_application(job, profile))
    session.commit()
    row = session.execute(
        text(
            "SELECT score IS NULL, matrix_json IS NULL, summary IS NULL, "
            "confirmed_at IS NULL, decided_at IS NULL FROM applications"
        )
    ).one()
    assert tuple(row) == (1, 1, 1, 1, 1)


def test_job_and_profile_nullable_fields_can_be_null(session: Session) -> None:
    """closes_at, location, min_years, confirmed_at and years_experience are optional."""
    job, profile = persisted_job_and_profile(session)
    assert job.closes_at is None
    assert job.location is None
    assert job.min_years is None
    assert profile.confirmed_at is None
    assert profile.years_experience is None


def test_decision_record_with_and_without_application_token(
    session: Session,
) -> None:
    """application_token is optional on a decision record."""
    job = make_job()
    session.add(job)
    session.commit()
    session.add_all(
        [
            make_decision(job.id),
            make_decision(job.id, application_token=FIRST_TOKEN, raw_hash=SECOND_HASH),
        ]
    )
    session.commit()
    tokens = session.scalars(
        select(DecisionRecord.application_token).order_by(DecisionRecord.id)
    ).all()
    assert tokens == [None, FIRST_TOKEN]


def test_decision_record_survives_profile_deletion(session: Session) -> None:
    """A raw_hash-keyed decision record has no foreign key to the profile."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_decision(job.id, raw_hash=profile.raw_hash))
    session.commit()
    session.delete(profile)
    session.commit()
    remaining = session.scalars(select(DecisionRecord)).one()
    assert remaining.raw_hash == FIRST_HASH


def test_section_embedding_and_hash_are_optional(session: Session) -> None:
    """A section can exist before it has been embedded."""
    profile = make_profile()
    profile.sections = [make_section()]
    session.add(profile)
    session.commit()
    section = session.scalars(select(Section)).one()
    assert section.embedding is None
    assert section.text_hash is None
    assert section.model_version is None


def test_json_and_blob_fields_round_trip(session: Session) -> None:
    """JSON structures and float32 embedding bytes come back unchanged."""
    embedding = np.linspace(-1.0, 1.0, EMBEDDING_DIMENSION, dtype=np.float32)
    matrix = [{"requirement_id": 1, "section_id": 2, "sim": 0.81, "evidence": True}]
    profile = make_profile()
    profile.profile_json = {"sections": [{"type": "skills", "text": "c++ c#"}]}
    profile.languages = ["english", "turkish"]
    section = make_section()
    section.embedding = embedding.tobytes()
    section.text_hash = "abc"
    section.model_version = MODEL_VERSION
    profile.sections = [section]
    job = make_job()
    job.required_languages = ["english"]
    session.add_all([job, profile])
    session.commit()
    session.add(
        Application(
            job_id=job.id,
            profile_id=profile.id,
            raw_hash=FIRST_HASH,
            status=ApplicationStatus.RANKED,
            score=0.75,
            matrix_json=matrix,
            summary="Strong fit.",
            draft_token=FIRST_TOKEN,
            created_at=NOW,
        )
    )
    session.commit()
    session.expire_all()
    stored_profile = session.scalars(select(CandidateProfile)).one()
    stored_section = stored_profile.sections[0]
    assert stored_profile.profile_json == {"sections": [{"type": "skills", "text": "c++ c#"}]}
    assert stored_profile.languages == ["english", "turkish"]
    assert stored_section.embedding == embedding.tobytes()
    restored = np.frombuffer(stored_section.embedding, dtype=np.float32)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, embedding)
    assert session.scalars(select(Application)).one().matrix_json == matrix
    assert session.scalars(select(JobPosting)).one().required_languages == ["english"]


def test_datetimes_round_trip_as_utc_aware(session: Session) -> None:
    """Stored datetimes come back timezone-aware in UTC, whatever zone went in."""
    istanbul = timezone(timedelta(hours=3))
    job = make_job()
    job.created_at = datetime(2026, 1, 15, 15, 0, tzinfo=istanbul)
    session.add(job)
    session.commit()
    session.expire_all()
    stored = session.scalars(select(JobPosting)).one().created_at
    assert stored == NOW
    assert stored.tzinfo is not None
    assert stored.utcoffset() == timedelta(0)


def test_naive_datetime_is_rejected(session: Session) -> None:
    """A naive datetime is a caller bug and is refused instead of guessed."""
    job = make_job()
    job.created_at = datetime(2026, 1, 15, 12, 0)
    session.add(job)
    with pytest.raises(StatementError, match="timezone-aware"):
        session.commit()


def test_no_datetime_column_has_a_default() -> None:
    """The clock is injected, so no model may fill a timestamp by itself."""
    for metadata in (DemoBase.metadata, BenchmarkBase.metadata):
        for table in metadata.tables.values():
            for column in table.columns:
                if "datetime" in type(column.type).__name__.lower():
                    assert column.default is None
                    assert column.server_default is None


def test_job_requirements_are_deleted_with_their_job(session: Session) -> None:
    """Requirements are owned by their posting."""
    job = make_job()
    job.requirements = [
        Requirement(text="SQL", weight=2, embedding=b"x", model_version=MODEL_VERSION)
    ]
    session.add(job)
    session.commit()
    session.execute(
        text(f"DELETE FROM {JobPosting.__tablename__} WHERE id = :id"), {"id": job.id}
    )
    session.commit()
    assert count_rows(session, Requirement.__tablename__) == 0


def test_job_with_application_cannot_be_deleted(session: Session) -> None:
    """Application.job_id has no cascade, so a posting with applications stays."""
    job, profile = persisted_job_and_profile(session)
    session.add(make_application(job, profile))
    session.commit()
    with pytest.raises(IntegrityError):
        session.execute(
            text(f"DELETE FROM {JobPosting.__tablename__} WHERE id = :id"),
            {"id": job.id},
        )


def test_indexes_exist(demo_engine: Engine) -> None:
    """The lookup indexes required by the plan are present."""
    inspector = inspect(demo_engine)

    def indexed_columns(table_name: str) -> list[list[str | None]]:
        return [index["column_names"] for index in inspector.get_indexes(table_name)]

    assert ["text_hash"] in indexed_columns(Section.__tablename__)
    assert ["job_id", "status"] in indexed_columns(Application.__tablename__)
    assert ["job_id", "raw_hash"] in indexed_columns(DecisionRecord.__tablename__)
    assert ["application_token"] in indexed_columns(DecisionRecord.__tablename__)


def test_benchmark_tables_work_with_cascade_and_checks(tmp_path: Path) -> None:
    """A run with stage measurements stores, cascades and validates the pipeline."""
    benchmark_engine = create_db_engine(
        f"sqlite:///{(tmp_path / BENCHMARK_DATABASE_FILE).as_posix()}"
    )
    try:
        create_benchmark_schema(benchmark_engine)
        with create_session_factory(benchmark_engine)() as benchmark_session:
            run = make_run(PipelineName.GREEN)
            run.stage_measurements = [
                StageMeasurement(stage="extract", energy_wh=0.1, duration_s=2.0, count=5)
            ]
            benchmark_session.add(run)
            benchmark_session.commit()
            assert run.precision_at_k is None
            assert run.scenario_json == {"n": 50, "seed": 1}
            benchmark_session.execute(
                text(f"DELETE FROM {BenchmarkRun.__tablename__} WHERE id = :id"),
                {"id": run.id},
            )
            benchmark_session.commit()
            assert count_rows(benchmark_session, StageMeasurement.__tablename__) == 0
            benchmark_session.add(make_run(INVALID_VALUE))
            with pytest.raises(IntegrityError):
                benchmark_session.commit()
    finally:
        benchmark_engine.dispose()


def make_run(pipeline: str) -> BenchmarkRun:
    """A valid benchmark run with quality metrics not yet computed."""
    return BenchmarkRun(
        pipeline=pipeline,
        scenario_json={"n": 50, "seed": 1},
        energy_wh_gross=1.0,
        energy_wh_net=0.8,
        co2_g=0.3,
        duration_s=60.0,
        idle_power_w=10.0,
        storage_bytes_active=1000,
        storage_bytes_after_retention=500,
        llm_calls=5,
        llm_prompt_tokens=100,
        llm_completion_tokens=50,
        parse_cache_hits=0,
        embed_cache_hits=0,
        measurement_notes="TDP estimate",
    )
