"""Small lookups that several pipeline modules need."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import JobPosting, JobStatus, Requirement, Section
from app.pipeline.context import (
    PipelineContext,
    PostingClosedError,
    PostingWithoutRequirementsError,
)
from app.privacy.retention import PostingNotFoundError, RetentionContext


def require_open_posting(session: Session, job_id: int) -> JobPosting:
    """Load a posting and fail fast unless it still accepts applications."""
    posting = session.get(JobPosting, job_id)
    if posting is None:
        raise PostingNotFoundError(f"Job posting {job_id} does not exist.")
    if posting.status is not JobStatus.OPEN:
        raise PostingClosedError(
            f"Job posting {job_id} is {posting.status.value} and takes no applications."
        )
    return posting


def retention_context(context: PipelineContext) -> RetentionContext:
    """The retention view of the pipeline context."""
    return RetentionContext(
        session=context.session,
        now=context.now,
        decision_retention_days=context.settings.decision_retention_days,
    )


def load_profile_sections(session: Session, profile_id: int) -> list[Section]:
    """The sections of a profile in reading order."""
    statement = (
        select(Section)
        .where(Section.profile_id == profile_id)
        .order_by(Section.ordinal, Section.id)
    )
    return list(session.scalars(statement))


def load_requirements(session: Session, job_id: int) -> list[Requirement]:
    """The requirements of a posting in definition order."""
    statement = (
        select(Requirement).where(Requirement.job_id == job_id).order_by(Requirement.id)
    )
    requirements = list(session.scalars(statement))
    if not requirements:
        raise PostingWithoutRequirementsError(f"Job posting {job_id} has no requirements.")
    return requirements
