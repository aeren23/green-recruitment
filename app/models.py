"""ORM entities of SPEC section 5.

There are two independent declarative bases because the benchmark tables live in
`data/bench/bench.db`, not in the demo database (SPEC section 5). No foreign key
crosses from one metadata to the other.

Conventions:
    * Primary keys are integers.
    * Every timestamp is UTC and is passed in by the caller. The clock is injected
      in this project, so no column has a `default=` and this module never reads
      the system clock.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Dialect,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import TypeDecorator

TABLE_JOB_POSTINGS = "job_postings"
TABLE_REQUIREMENTS = "requirements"
TABLE_CANDIDATE_PROFILES = "candidate_profiles"
TABLE_SECTIONS = "sections"
TABLE_CANDIDATE_CONTACTS = "candidate_contacts"
TABLE_APPLICATIONS = "applications"
TABLE_DECISION_RECORDS = "decision_records"
TABLE_GREEN_COUNTER = "green_counter"
TABLE_BENCHMARK_RUNS = "benchmark_runs"
TABLE_STAGE_MEASUREMENTS = "stage_measurements"

DEMO_TABLE_NAMES = frozenset(
    {
        TABLE_JOB_POSTINGS,
        TABLE_REQUIREMENTS,
        TABLE_CANDIDATE_PROFILES,
        TABLE_SECTIONS,
        TABLE_CANDIDATE_CONTACTS,
        TABLE_APPLICATIONS,
        TABLE_DECISION_RECORDS,
        TABLE_GREEN_COUNTER,
    }
)
BENCHMARK_TABLE_NAMES = frozenset({TABLE_BENCHMARK_RUNS, TABLE_STAGE_MEASUREMENTS})

REQUIREMENT_WEIGHT_MIN = 1
REQUIREMENT_WEIGHT_MAX = 3
GREEN_COUNTER_ROW_ID = 1
COUNTER_INITIAL_VALUE = 0

ON_DELETE_CASCADE = "CASCADE"
ORM_CASCADE_OWNED_CHILDREN = "all, delete-orphan"


class JobStatus(StrEnum):
    """Lifecycle of a job posting."""

    OPEN = "open"
    CLOSED = "closed"
    FINALIZED = "finalized"


class ApplicationStatus(StrEnum):
    """Lifecycle of an application while it still exists."""

    DRAFT = "draft"
    RANKED = "ranked"
    SHORTLISTED = "shortlisted"


class SectionType(StrEnum):
    """Embedded CV section types. `languages` is deliberately absent (SPEC 6.3)."""

    EXPERIENCE = "experience"
    PROJECT = "project"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    SUMMARY = "summary"
    SKILLS = "skills"


class PipelineName(StrEnum):
    """The two benchmarked pipelines."""

    BASELINE = "baseline"
    GREEN = "green"


class UtcDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetime that survives SQLite's lack of timezone storage.

    SQLite returns naive datetimes, which would silently break comparisons with the
    injected, timezone-aware `now`. This type refuses naive values on the way in and
    hands back UTC-aware values on the way out.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Convert an aware datetime to UTC, rejecting naive ones."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware (UTC expected).")
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Re-attach UTC to the naive value read back from the database."""
        if value is None:
            return None
        return value.replace(tzinfo=UTC)


def enum_column_type(enum_class: type[StrEnum]) -> Enum:
    """Column type that stores the lowercase enum values as plain strings.

    The CHECK constraint is declared separately on each table, so the database
    rejects bad values even when a caller bypasses the ORM.

    Args:
        enum_class: The `StrEnum` whose values are stored.
    """
    return Enum(
        enum_class,
        native_enum=False,
        create_constraint=False,
        values_callable=lambda members: [member.value for member in members],
        length=max(len(member.value) for member in enum_class),
    )


def values_check(
    column_name: str, enum_class: type[StrEnum], constraint_name: str
) -> CheckConstraint:
    """CHECK constraint restricting a column to the values of a `StrEnum`.

    Args:
        column_name: Column to constrain.
        enum_class: The `StrEnum` listing the allowed values.
        constraint_name: Name of the constraint, for readable errors.
    """
    allowed_values = ", ".join(f"'{member.value}'" for member in enum_class)
    return CheckConstraint(
        f"{column_name} IN ({allowed_values})", name=constraint_name
    )


class DemoBase(DeclarativeBase):
    """Declarative base of the demo database tables."""


class BenchmarkBase(DeclarativeBase):
    """Declarative base of the benchmark tables (kept in `data/bench/bench.db`)."""


class JobPosting(DemoBase):
    """A job posting with its hard filters."""

    __tablename__ = TABLE_JOB_POSTINGS
    __table_args__ = (values_check("status", JobStatus, "ck_job_postings_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[JobStatus] = mapped_column(enum_column_type(JobStatus))
    # Informational only: closing a posting is manual (SPEC 5).
    closes_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    closed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    finalized_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    required_languages: Mapped[list[str]] = mapped_column(JSON)
    location: Mapped[str | None] = mapped_column(String)
    remote_ok: Mapped[bool] = mapped_column()
    min_years: Mapped[float | None] = mapped_column(Float)
    # Used only for the benchmark quality metric, never shown in the UI.
    relevant_categories: Mapped[list[str]] = mapped_column(JSON)

    requirements: Mapped[list["Requirement"]] = relationship(
        back_populates="job",
        cascade=ORM_CASCADE_OWNED_CHILDREN,
        passive_deletes=True,
        order_by="Requirement.id",
    )


class Requirement(DemoBase):
    """One weighted requirement of a job posting, embedded once at creation."""

    __tablename__ = TABLE_REQUIREMENTS
    __table_args__ = (
        CheckConstraint(
            f"weight BETWEEN {REQUIREMENT_WEIGHT_MIN} AND {REQUIREMENT_WEIGHT_MAX}",
            name="ck_requirements_weight",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Requirements are owned by their posting, so they go with it.
    job_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TABLE_JOB_POSTINGS}.id", ondelete=ON_DELETE_CASCADE)
    )
    text: Mapped[str] = mapped_column(Text)
    weight: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    model_version: Mapped[str] = mapped_column(String)

    job: Mapped[JobPosting] = relationship(back_populates="requirements")


class CandidateProfile(DemoBase):
    """The parse cache itself: the confirmed structure of one CV, without contact data."""

    __tablename__ = TABLE_CANDIDATE_PROFILES

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_hash: Mapped[str] = mapped_column(String, unique=True)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    # Also kept as columns so the hard filters can run in SQL.
    years_experience: Mapped[float | None] = mapped_column(Float)
    languages: Mapped[list[str]] = mapped_column(JSON)
    location: Mapped[str | None] = mapped_column(String)
    parser_version: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    confirmed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_used_at: Mapped[datetime] = mapped_column(UtcDateTime)

    sections: Mapped[list["Section"]] = relationship(
        back_populates="profile",
        cascade=ORM_CASCADE_OWNED_CHILDREN,
        passive_deletes=True,
        order_by="Section.ordinal",
    )
    contact: Mapped["CandidateContact | None"] = relationship(
        back_populates="profile",
        cascade=ORM_CASCADE_OWNED_CHILDREN,
        passive_deletes=True,
        uselist=False,
    )
    # There is deliberately no reverse relationship to Application here. With one,
    # deleting a profile through the ORM would try to null out the applications'
    # profile_id instead of failing loudly on the foreign key.


class Section(DemoBase):
    """One embedded piece of a CV (an experience entry, the skills block, ...)."""

    __tablename__ = TABLE_SECTIONS
    __table_args__ = (
        values_check("type", SectionType, "ck_sections_type"),
        # Global embedding reuse looks sections up by this hash (SPEC 6.1).
        Index("ix_sections_text_hash", "text_hash"),
        # Lets the database find the children of a profile for the cascade.
        Index("ix_sections_profile_id", "profile_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TABLE_CANDIDATE_PROFILES}.id", ondelete=ON_DELETE_CASCADE)
    )
    type: Mapped[SectionType] = mapped_column(enum_column_type(SectionType))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # text_hash is SHA-256 of model_version + normalized text (SPEC 5). It is
    # nullable together with embedding and model_version: a section parsed for a
    # draft has no embedding model applied yet, and the hash is only meaningful
    # (and only compared) at confirmation, when the embedding is computed or reused.
    text_hash: Mapped[str | None] = mapped_column(String)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    model_version: Mapped[str | None] = mapped_column(String)

    profile: Mapped[CandidateProfile] = relationship(back_populates="sections")


class CandidateContact(DemoBase):
    """Contact data, kept apart from the profile for blind screening."""

    __tablename__ = TABLE_CANDIDATE_CONTACTS
    __table_args__ = (UniqueConstraint("profile_id", name="uq_candidate_contacts_profile"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TABLE_CANDIDATE_PROFILES}.id", ondelete=ON_DELETE_CASCADE)
    )
    # Nullable because the regex and first-line heuristics of SPEC 6.3 can find
    # nothing, and the SPEC does not require any of the three to be present.
    name: Mapped[str | None] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String)
    phone: Mapped[str | None] = mapped_column(String)

    profile: Mapped[CandidateProfile] = relationship(back_populates="contact")


class Application(DemoBase):
    """One candidate's application to one job posting."""

    __tablename__ = TABLE_APPLICATIONS
    __table_args__ = (
        UniqueConstraint("job_id", "raw_hash", name="uq_applications_job_raw_hash"),
        values_check("status", ApplicationStatus, "ck_applications_status"),
        Index("ix_applications_job_status", "job_id", "status"),
        # Used by the profile reference count (SPEC 8.3).
        Index("ix_applications_profile_id", "profile_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # No ondelete on either foreign key, on purpose. Deleting a profile that still
    # has an application must fail with a foreign-key error, so a reference-count
    # bug surfaces immediately instead of silently losing data. The same holds for
    # a posting, which is closed and finalized but never removed.
    job_id: Mapped[int] = mapped_column(ForeignKey(f"{TABLE_JOB_POSTINGS}.id"))
    profile_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TABLE_CANDIDATE_PROFILES}.id")
    )
    raw_hash: Mapped[str] = mapped_column(String)
    status: Mapped[ApplicationStatus] = mapped_column(
        enum_column_type(ApplicationStatus)
    )
    score: Mapped[float | None] = mapped_column(Float)
    matrix_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON(none_as_null=True)
    )
    summary: Mapped[str | None] = mapped_column(Text)
    draft_token: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime)
    confirmed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    decided_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    profile: Mapped[CandidateProfile] = relationship()


class DecisionRecord(DemoBase):
    """The only trace left after a profile is deleted.

    It is keyed by `raw_hash` and has no foreign key to the profile, so it survives
    the profile's deletion.
    """

    __tablename__ = TABLE_DECISION_RECORDS
    __table_args__ = (
        Index("ix_decision_records_job_raw_hash", "job_id", "raw_hash"),
        Index("ix_decision_records_application_token", "application_token"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey(f"{TABLE_JOB_POSTINGS}.id"))
    raw_hash: Mapped[str] = mapped_column(String)
    reason_code: Mapped[str] = mapped_column(String)
    decided_at: Mapped[datetime] = mapped_column(UtcDateTime)
    expires_at: Mapped[datetime] = mapped_column(UtcDateTime)
    # Copy of the deleted application's draft_token, so its status page still works.
    application_token: Mapped[str | None] = mapped_column(String)


def counter_column() -> Mapped[int]:
    """A non-negative integer counter column starting at zero."""
    return mapped_column(
        Integer,
        nullable=False,
        default=COUNTER_INITIAL_VALUE,
        server_default=text(str(COUNTER_INITIAL_VALUE)),
    )


class GreenCounter(DemoBase):
    """Single-row counter table for the demo panel."""

    __tablename__ = TABLE_GREEN_COUNTER
    __table_args__ = (
        CheckConstraint(f"id = {GREEN_COUNTER_ROW_ID}", name="ck_green_counter_single_row"),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=False, default=GREEN_COUNTER_ROW_ID
    )
    pdf_bytes_not_stored: Mapped[int] = counter_column()
    parse_cache_hits: Mapped[int] = counter_column()
    embed_cache_hits: Mapped[int] = counter_column()
    embeddings_computed: Mapped[int] = counter_column()
    llm_calls: Mapped[int] = counter_column()
    llm_calls_avoided: Mapped[int] = counter_column()
    records_deleted: Mapped[int] = counter_column()


class BenchmarkRun(BenchmarkBase):
    """One benchmark run of one pipeline."""

    __tablename__ = TABLE_BENCHMARK_RUNS
    __table_args__ = (
        values_check("pipeline", PipelineName, "ck_benchmark_runs_pipeline"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    pipeline: Mapped[PipelineName] = mapped_column(enum_column_type(PipelineName))
    scenario_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    # Gross energy and duration are sums over the stages, excluding model_load.
    energy_wh_gross: Mapped[float] = mapped_column(Float)
    energy_wh_net: Mapped[float] = mapped_column(Float)
    co2_g: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    idle_power_w: Mapped[float] = mapped_column(Float)
    storage_bytes_active: Mapped[int] = mapped_column(Integer)
    storage_bytes_after_retention: Mapped[int] = mapped_column(Integer)
    llm_calls: Mapped[int] = mapped_column(Integer)
    llm_prompt_tokens: Mapped[int] = mapped_column(Integer)
    llm_completion_tokens: Mapped[int] = mapped_column(Integer)
    parse_cache_hits: Mapped[int] = mapped_column(Integer)
    embed_cache_hits: Mapped[int] = mapped_column(Integer)
    # Quality metrics are computed from the final rankings, possibly after the run
    # row was written (and overlap needs the other pipeline), so they are nullable.
    precision_at_k: Mapped[float | None] = mapped_column(Float)
    overlap_at_k: Mapped[float | None] = mapped_column(Float)
    measurement_notes: Mapped[str] = mapped_column(Text)

    stage_measurements: Mapped[list["StageMeasurement"]] = relationship(
        back_populates="run",
        cascade=ORM_CASCADE_OWNED_CHILDREN,
        passive_deletes=True,
        order_by="StageMeasurement.id",
    )


class StageMeasurement(BenchmarkBase):
    """Energy and duration of one stage within a benchmark run."""

    __tablename__ = TABLE_STAGE_MEASUREMENTS

    id: Mapped[int] = mapped_column(primary_key=True)
    # Owned by its run, so the measurements go with it.
    run_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TABLE_BENCHMARK_RUNS}.id", ondelete=ON_DELETE_CASCADE)
    )
    stage: Mapped[str] = mapped_column(String)
    energy_wh: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    count: Mapped[int] = mapped_column(Integer)

    run: Mapped[BenchmarkRun] = relationship(back_populates="stage_measurements")
