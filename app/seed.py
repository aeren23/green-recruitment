"""The three demo and benchmark job postings of SPEC 7.2, and the function that seeds them.

The posting data is named module-level data. `seed_postings` creates only the
postings that do not exist yet (matched by title), so a second call changes
nothing and computes no embedding.

Filter values:
    The benchmark runs with the language and location filters off (SPEC 7.2),
    which is the default here: `required_languages=[]`, `location=None`,
    `remote_ok=True`. The demo application uses real filter values
    (`DEMO_FILTERS_BY_TITLE`), supplied through `SeedOptions.filters_by_title`
    by `python -m app.seed` (`app/seed_demo.py`).

Creation itself (embedding in one batch, normalization of the filter values)
is `app.postings.create_postings`, shared with the employer web form.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import JobPosting
from app.postings import PostingInput, RequirementInput, create_postings
from app.scoring.embed import EmbeddingProvider

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

LANGUAGE_ENGLISH = "english"
LANGUAGE_TURKISH = "turkish"
LOCATION_DENIZLI = "denizli"
LOCATION_IZMIR = "izmir"

# The demo's real hard-filter values (T-12). The Backend Developer may work
# remotely, so its location is informational; the other two are on-site.
DEMO_FILTERS_BY_TITLE: Mapping[str, PostingFilters] = {
    BACKEND_DEVELOPER_TITLE: PostingFilters(
        required_languages=(LANGUAGE_ENGLISH,), location=LOCATION_DENIZLI, remote_ok=True
    ),
    ACCOUNTANT_TITLE: PostingFilters(
        required_languages=(LANGUAGE_TURKISH,), location=LOCATION_DENIZLI, remote_ok=False
    ),
    HR_SPECIALIST_TITLE: PostingFilters(
        required_languages=(LANGUAGE_TURKISH, LANGUAGE_ENGLISH),
        location=LOCATION_IZMIR,
        remote_ok=False,
    ),
}

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
    inputs = [_posting_input(seed, options) for seed in missing]
    created = create_postings(session, embedder, inputs)
    for seed, posting in zip(missing, created, strict=True):
        existing[seed.title] = posting
    logger.info(
        "Seeded %d new postings, %d already existed",
        len(missing),
        len(existing) - len(missing),
    )
    return [existing[seed.title] for seed in SEED_POSTINGS]


def missing_seed_titles(session: Session) -> list[str]:
    """Titles of the seed postings the database does not hold yet."""
    existing = _postings_by_title(session)
    return [seed.title for seed in SEED_POSTINGS if seed.title not in existing]


def _postings_by_title(session: Session) -> dict[str, JobPosting]:
    """All stored postings keyed by title."""
    return {posting.title: posting for posting in session.scalars(select(JobPosting))}


def _posting_input(seed: PostingSeed, options: SeedOptions) -> PostingInput:
    """The creation input of one seed, with the filter values of its title."""
    filters = options.filters_for(seed.title)
    return PostingInput(
        title=seed.title,
        description=seed.description,
        requirements=[
            RequirementInput(requirement.text, requirement.weight)
            for requirement in seed.requirements
        ],
        created_at=options.now,
        required_languages=filters.required_languages,
        location=filters.location,
        remote_ok=filters.remote_ok,
        min_years=seed.min_years,
        relevant_categories=seed.relevant_categories,
    )


if __name__ == "__main__":
    from app.seed_demo import main

    main()
