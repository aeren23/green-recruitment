"""The three demo and benchmark job postings of SPEC 7.2, and the function that seeds them.

The posting data is named module-level data. `seed_postings` creates only the
postings that do not exist yet (matched by title), so a second call changes
nothing and computes no embedding.

Filter values:
    The benchmark runs with the language and location filters off (SPEC 7.2),
    which is the default here: `required_languages=[]`, `location=None`,
    `remote_ok=True`. The demo application uses real filter values; they are
    added in T-12 through `SeedOptions.filters_by_title` without touching this
    module's data or logic.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import JobPosting, JobStatus, Requirement
from app.scoring.embed import (
    EmbeddingProvider,
    UnexpectedEmbeddingShapeError,
    embedding_to_blob,
)

logger = logging.getLogger(__name__)

CATEGORY_INFORMATION_TECHNOLOGY = "INFORMATION-TECHNOLOGY"
CATEGORY_ACCOUNTANT = "ACCOUNTANT"
CATEGORY_FINANCE = "FINANCE"
CATEGORY_HR = "HR"

BACKEND_DEVELOPER_TITLE = "Backend Developer"
ACCOUNTANT_TITLE = "Accountant"
HR_SPECIALIST_TITLE = "HR Specialist"

WEIGHT_HIGH = 3
WEIGHT_MEDIUM = 2
WEIGHT_LOW = 1


@dataclass(frozen=True)
class RequirementSeed:
    """One weighted requirement of a seeded posting (English, since CVs are English)."""

    text: str
    weight: int


@dataclass(frozen=True)
class PostingSeed:
    """The fixed data of one seeded posting, without its hard-filter values."""

    title: str
    description: str
    relevant_categories: tuple[str, ...]
    min_years: float
    requirements: tuple[RequirementSeed, ...]


@dataclass(frozen=True)
class PostingFilters:
    """Language and location hard filters of a posting (SPEC 6.4).

    The defaults switch both filters off, which is the benchmark mode.
    """

    required_languages: tuple[str, ...] = ()
    location: str | None = None
    remote_ok: bool = True


BENCHMARK_FILTERS = PostingFilters()

BACKEND_DEVELOPER_POSTING = PostingSeed(
    title=BACKEND_DEVELOPER_TITLE,
    description=(
        "Design, build and operate backend services and REST APIs on top of "
        "relational databases."
    ),
    relevant_categories=(CATEGORY_INFORMATION_TECHNOLOGY,),
    min_years=2,
    requirements=(
        RequirementSeed("Building REST APIs and backend services", WEIGHT_HIGH),
        RequirementSeed("Relational databases and SQL", WEIGHT_MEDIUM),
        RequirementSeed("Python or Java programming", WEIGHT_MEDIUM),
        RequirementSeed("Linux, cloud or container infrastructure", WEIGHT_LOW),
    ),
)
ACCOUNTANT_POSTING = PostingSeed(
    title=ACCOUNTANT_TITLE,
    description=(
        "Maintain the general ledger, prepare financial statements and handle "
        "tax compliance for the company."
    ),
    relevant_categories=(CATEGORY_ACCOUNTANT, CATEGORY_FINANCE),
    min_years=3,
    requirements=(
        RequirementSeed("Financial statements and general ledger accounting", WEIGHT_HIGH),
        RequirementSeed("Tax preparation and compliance", WEIGHT_MEDIUM),
        RequirementSeed("Accounts payable and receivable", WEIGHT_MEDIUM),
        RequirementSeed("Excel and accounting software", WEIGHT_LOW),
    ),
)
HR_SPECIALIST_POSTING = PostingSeed(
    title=HR_SPECIALIST_TITLE,
    description=(
        "Run recruitment, employee relations and payroll administration for "
        "the company's people team."
    ),
    relevant_categories=(CATEGORY_HR,),
    min_years=2,
    requirements=(
        RequirementSeed("Recruitment and talent acquisition", WEIGHT_HIGH),
        RequirementSeed("Employee relations and HR policies", WEIGHT_MEDIUM),
        RequirementSeed("Payroll and benefits administration", WEIGHT_MEDIUM),
        RequirementSeed("HRIS systems", WEIGHT_LOW),
    ),
)
SEED_POSTINGS = (
    BACKEND_DEVELOPER_POSTING,
    ACCOUNTANT_POSTING,
    HR_SPECIALIST_POSTING,
)


@dataclass(frozen=True)
class EmbeddedSeed:
    """A seed together with the float32 blobs of its requirement embeddings."""

    seed: PostingSeed
    embeddings: tuple[bytes, ...]


@dataclass(frozen=True)
class SeedOptions:
    """How `seed_postings` fills in what the posting data does not define.

    Attributes:
        now: The injected clock, stored as `created_at`.
        default_filters: Hard-filter values of every posting that has no entry
            in `filters_by_title`; the benchmark default switches them off.
        filters_by_title: Extension point for the demo (T-12): filter values
            per posting title.
    """

    now: datetime
    default_filters: PostingFilters = BENCHMARK_FILTERS
    filters_by_title: Mapping[str, PostingFilters] = field(default_factory=dict)

    def filters_for(self, title: str) -> PostingFilters:
        """The filter values of the posting with this title."""
        return self.filters_by_title.get(title, self.default_filters)


def seed_postings(
    session: Session, embedder: EmbeddingProvider, options: SeedOptions
) -> list[JobPosting]:
    """Create the missing seed postings and their embedded requirements.

    The requirement texts of all missing postings are embedded in one
    `embed_queries` batch. Postings that already exist (same title) are
    returned as they are, so calling this twice creates nothing new. Nothing
    is committed; the caller owns the transaction.

    Args:
        session: Session on the demo or benchmark database.
        embedder: The loaded embedding provider.
        options: Clock and hard-filter values.

    Returns:
        The seed postings in the order of `SEED_POSTINGS`.

    Raises:
        UnexpectedEmbeddingShapeError: If the embedder returns the wrong
            number of vectors.
    """
    existing = _postings_by_title(session)
    missing = [seed for seed in SEED_POSTINGS if seed.title not in existing]
    embedded_seeds = _attach_embeddings(embedder, missing)
    created = [
        _build_posting(embedded, options, embedder.model_version)
        for embedded in embedded_seeds
    ]
    session.add_all(created)
    session.flush()
    for seed, posting in zip(missing, created, strict=True):
        existing[seed.title] = posting
    logger.info(
        "Seeded %d new postings, %d already existed",
        len(missing),
        len(existing) - len(missing),
    )
    return [existing[seed.title] for seed in SEED_POSTINGS]


def _postings_by_title(session: Session) -> dict[str, JobPosting]:
    """All stored postings keyed by title."""
    return {posting.title: posting for posting in session.scalars(select(JobPosting))}


def _attach_embeddings(
    embedder: EmbeddingProvider, seeds: list[PostingSeed]
) -> list[EmbeddedSeed]:
    """Embed every requirement of every seed in one query batch (none: no call)."""
    if not seeds:
        return []
    texts = [requirement.text for seed in seeds for requirement in seed.requirements]
    matrix = embedder.embed_queries(texts)
    if len(matrix) != len(texts):
        raise UnexpectedEmbeddingShapeError(
            f"Embedder returned {len(matrix)} vectors for {len(texts)} requirement texts."
        )
    blobs = iter(embedding_to_blob(row) for row in matrix)
    return [
        EmbeddedSeed(seed, tuple(next(blobs) for _ in seed.requirements))
        for seed in seeds
    ]


def _build_posting(
    embedded: EmbeddedSeed, options: SeedOptions, model_version: str
) -> JobPosting:
    """One open posting with its embedded requirements."""
    seed = embedded.seed
    filters = options.filters_for(seed.title)
    posting = JobPosting(
        title=seed.title,
        description=seed.description,
        status=JobStatus.OPEN,
        closes_at=None,
        closed_at=None,
        finalized_at=None,
        created_at=options.now,
        required_languages=list(filters.required_languages),
        location=filters.location,
        remote_ok=filters.remote_ok,
        min_years=seed.min_years,
        relevant_categories=list(seed.relevant_categories),
    )
    posting.requirements = [
        Requirement(
            text=requirement.text,
            weight=requirement.weight,
            embedding=embedding,
            model_version=model_version,
        )
        for requirement, embedding in zip(
            seed.requirements, embedded.embeddings, strict=True
        )
    ]
    return posting
