"""Creation of job postings with embedded requirements (SPEC 5, T-12).

The demo seed and the employer web form both create postings through this
module, so they cannot drift apart:

* All requirement texts of all postings in one call are embedded in **one**
  `embed_queries` batch, and the embedder's `model_version` is stored on every
  requirement.
* Hard-filter values are normalized with the very functions the pipeline uses
  for the candidate's values (`normalize_location`, `normalize_languages`). The
  SQL filters of `scoring/filters.py` compare exact strings, so a posting that
  said `DENİZLİ` and a candidate that said `Denizli` must both end up as
  `denizli`.

Nothing is committed here; the caller owns the transaction.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.models import (
    REQUIREMENT_WEIGHT_MAX,
    REQUIREMENT_WEIGHT_MIN,
    JobPosting,
    JobStatus,
    Requirement,
)
from app.pipeline.profile_edits import normalize_languages, normalize_location
from app.scoring.embed import (
    EmbeddingProvider,
    UnexpectedEmbeddingShapeError,
    embedding_to_blob,
)

logger = logging.getLogger(__name__)


class InvalidPostingError(ValueError):
    """A posting breaks an invariant the database or the scorer relies on."""


@dataclass(frozen=True)
class RequirementInput:
    """One weighted requirement (English, since the CVs are English)."""

    text: str
    weight: int


@dataclass(frozen=True)
class PostingInput:
    """Everything needed to create one open posting.

    Attributes:
        title: Posting title.
        description: Free text; may be empty.
        requirements: At least one weighted requirement.
        created_at: The injected clock (timezone-aware UTC).
        required_languages: Languages of which a candidate must know any one;
            Turkish names are mapped to English on creation. May be empty.
        location: Free-text location, folded on creation; `None` or blank means none.
        remote_ok: When true, the location filter does not apply.
        min_years: Minimum years of experience; `None` switches the filter off.
        relevant_categories: Benchmark quality metric only, never shown in the UI.
    """

    title: str
    description: str
    requirements: Sequence[RequirementInput]
    created_at: datetime
    required_languages: Sequence[str] = ()
    location: str | None = None
    remote_ok: bool = True
    min_years: float | None = None
    relevant_categories: Sequence[str] = ()


def create_posting(
    session: Session, embedder: EmbeddingProvider, posting_input: PostingInput
) -> JobPosting:
    """Create one open posting and embed its requirements in one batch.

    Args:
        session: Session on the demo or benchmark database; not committed.
        embedder: The loaded embedding provider.
        posting_input: The posting to create.

    Returns:
        The new posting, flushed so that it and its requirements have ids.

    Raises:
        InvalidPostingError: If the posting has no title, no requirement, a
            blank requirement text or a weight outside 1 to 3.
        UnexpectedEmbeddingShapeError: If the embedder returns the wrong
            number of vectors.
    """
    return create_postings(session, embedder, [posting_input])[0]


def create_postings(
    session: Session, embedder: EmbeddingProvider, inputs: Sequence[PostingInput]
) -> list[JobPosting]:
    """Create several open postings; all requirements share one embedding call.

    An empty `inputs` creates nothing and calls the embedder not at all. When
    validation or embedding fails, nothing is added to the session.

    Returns:
        The new postings in the order of `inputs`.

    Raises:
        InvalidPostingError: See `create_posting`.
        UnexpectedEmbeddingShapeError: See `create_posting`.
    """
    if not inputs:
        return []
    for posting_input in inputs:
        validate_posting_input(posting_input)
    blobs = _embed_requirements(embedder, inputs)
    postings = [
        _build_posting(posting_input, posting_blobs, embedder.model_version)
        for posting_input, posting_blobs in zip(inputs, blobs, strict=True)
    ]
    session.add_all(postings)
    session.flush()
    logger.info("Created %d posting(s): ids=%s", len(postings), [item.id for item in postings])
    return postings


def validate_posting_input(posting_input: PostingInput) -> None:
    """Fail fast on a posting that cannot be stored or scored.

    Raises:
        InvalidPostingError: With a message naming the broken invariant.
    """
    if not posting_input.title.strip():
        raise InvalidPostingError("A posting needs a title.")
    if not posting_input.requirements:
        raise InvalidPostingError("A posting needs at least one requirement.")
    for requirement in posting_input.requirements:
        _validate_requirement(requirement)


def _validate_requirement(requirement: RequirementInput) -> None:
    """One requirement needs text and a weight inside the allowed range."""
    if not requirement.text.strip():
        raise InvalidPostingError("A requirement needs a text.")
    if not REQUIREMENT_WEIGHT_MIN <= requirement.weight <= REQUIREMENT_WEIGHT_MAX:
        raise InvalidPostingError(
            f"Requirement weight {requirement.weight} is outside "
            f"{REQUIREMENT_WEIGHT_MIN} to {REQUIREMENT_WEIGHT_MAX}."
        )


def _embed_requirements(
    embedder: EmbeddingProvider, inputs: Sequence[PostingInput]
) -> list[list[bytes]]:
    """Embed every requirement of every posting in one query batch.

    Returns:
        For each posting, the float32 blobs of its requirements in order.
    """
    texts = [_clean_text(item.text) for posting in inputs for item in posting.requirements]
    matrix = embedder.embed_queries(texts)
    if len(matrix) != len(texts):
        raise UnexpectedEmbeddingShapeError(
            f"Embedder returned {len(matrix)} vectors for {len(texts)} requirement texts."
        )
    blobs = iter(embedding_to_blob(row) for row in matrix)
    return [[next(blobs) for _ in posting.requirements] for posting in inputs]


def _clean_text(text: str) -> str:
    """Surrounding whitespace never belongs to a stored text."""
    return text.strip()


def _build_posting(
    posting_input: PostingInput, blobs: Sequence[bytes], model_version: str
) -> JobPosting:
    """One open posting with normalized filter values and embedded requirements."""
    posting = JobPosting(
        title=_clean_text(posting_input.title),
        description=_clean_text(posting_input.description),
        status=JobStatus.OPEN,
        closes_at=None,
        closed_at=None,
        finalized_at=None,
        created_at=posting_input.created_at,
        required_languages=normalize_languages(posting_input.required_languages),
        location=normalize_location(posting_input.location),
        remote_ok=posting_input.remote_ok,
        min_years=posting_input.min_years,
        relevant_categories=list(posting_input.relevant_categories),
    )
    posting.requirements = [
        Requirement(
            text=_clean_text(requirement.text),
            weight=requirement.weight,
            embedding=blob,
            model_version=model_version,
        )
        for requirement, blob in zip(posting_input.requirements, blobs, strict=True)
    ]
    return posting
