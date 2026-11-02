"""Minimal retention of SPEC section 8.3: R2, R3, the R1 rule for one draft and
the profile reference count.

R4, R5 and the `run()` entry point belong to T-14. Every deletion in the
application goes through this module, so the reference count and the decision
records cannot be bypassed.

Conventions:
    * The clock is injected as `RetentionContext.now`; this module never reads
      the system clock.
    * Nothing here commits. The caller owns the transaction, so a rolled-back
      unit of work also rolls back its deletions and counts.
    * Logs and exception messages carry ids only, never CV data (KVKK).
    * Every deleted row (application, profile, sections, contact) adds to
      `GreenCounter.records_deleted`.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app.green.counters import CounterName, increment_counter
from app.models import (
    Application,
    ApplicationStatus,
    CandidateContact,
    CandidateProfile,
    DecisionRecord,
    JobPosting,
    JobStatus,
    Section,
)
from app.privacy.reasons import StoredReasonCode

logger = logging.getLogger(__name__)

PROFILE_ROW_COUNT = 1
APPLICATION_ROW_COUNT = 1
NO_REFERENCES = 0


class PostingNotFoundError(LookupError):
    """The job posting does not exist."""


class ApplicationNotDraftError(RuntimeError):
    """An operation that applies to drafts only was given another application."""


@dataclass(frozen=True)
class RetentionContext:
    """What every retention operation needs.

    Attributes:
        session: Session whose transaction the caller owns; nothing is committed.
        now: The injected clock (timezone-aware UTC).
        decision_retention_days: How long a `DecisionRecord` blocks a new
            application to the same posting (`DECISION_RETENTION_DAYS`).
    """

    session: Session
    now: datetime
    decision_retention_days: int


def is_draft_expired(application: Application, now: datetime, ttl_hours: int) -> bool:
    """The R1 condition: a draft older than its time to live.

    Args:
        application: The application to check.
        now: The injected clock.
        ttl_hours: `DRAFT_TTL_HOURS`.

    Returns:
        True only for a `draft` whose `created_at + ttl_hours` is before `now`.
    """
    if application.status is not ApplicationStatus.DRAFT:
        return False
    return application.created_at + timedelta(hours=ttl_hours) < now


def eliminate_application(
    context: RetentionContext,
    application: Application,
    reason: StoredReasonCode,
) -> None:
    """Rule R2: record the decision, delete the application, release its profile.

    Idempotent: an application that no longer exists in the database is
    ignored, so no second decision record is written.

    Args:
        context: Session, injected clock and decision retention period.
        application: The application to eliminate.
        reason: A stored reason code (a hard-filter code or `NOT_SELECTED`).

    Raises:
        ValueError: If `reason` is not a stored reason code. Nothing is changed.
    """
    stored_reason = StoredReasonCode(reason)
    if not _is_persistent(application):
        return
    context.session.add(_build_decision_record(context, application, stored_reason))
    _remove_application(context, application)
    logger.info(
        "Application %d of posting %d eliminated: reason=%s",
        application.id,
        application.job_id,
        stored_reason.value,
    )


def close_posting(context: RetentionContext, job_id: int) -> int:
    """Rule R3: close a posting and turn its `ranked` applications into decisions.

    Each `ranked` application gets a `NOT_SELECTED` decision record and is
    deleted. Drafts and shortlisted applications are left alone. Idempotent:
    a second call changes nothing and returns 0, and an already closed or
    finalized posting keeps its status and `closed_at`.

    Args:
        context: Session, injected clock and decision retention period.
        job_id: Id of the posting to close.

    Returns:
        The number of applications that were not selected.

    Raises:
        PostingNotFoundError: If the posting does not exist.
    """
    posting = context.session.get(JobPosting, job_id)
    if posting is None:
        raise PostingNotFoundError(f"Job posting {job_id} does not exist.")
    if posting.status is JobStatus.OPEN:
        posting.status = JobStatus.CLOSED
        posting.closed_at = context.now
    ranked = _ranked_applications(context.session, job_id)
    for application in ranked:
        eliminate_application(context, application, StoredReasonCode.NOT_SELECTED)
    logger.info("Posting %d closed: %d ranked applications not selected", job_id, len(ranked))
    return len(ranked)


def expire_draft(context: RetentionContext, application: Application) -> None:
    """Rule R1 for one draft: delete it without a decision record, release its profile.

    The caller decides that the draft has expired (see `is_draft_expired`).
    Idempotent: an application that no longer exists is ignored.

    Args:
        context: Session, injected clock and decision retention period.
        application: The expired draft.

    Raises:
        ApplicationNotDraftError: If the application is not a draft.
    """
    if not _is_persistent(application):
        return
    if application.status is not ApplicationStatus.DRAFT:
        raise ApplicationNotDraftError(
            f"Application {application.id} is not a draft and cannot expire."
        )
    _remove_application(context, application)
    logger.info("Draft %d of posting %d expired", application.id, application.job_id)


def discard_replaced_sections(
    context: RetentionContext, sections: Sequence[Section]
) -> None:
    """Delete sections that a profile edit replaced or removed.

    Args:
        context: Session, injected clock and decision retention period.
        sections: The sections of one profile that are no longer part of it.
    """
    for section in sections:
        context.session.delete(section)
    context.session.flush()
    _count_deleted(context, len(sections))


def _is_persistent(application: Application) -> bool:
    """Whether the application still exists as a database row of its session."""
    return inspect(application).persistent


def _build_decision_record(
    context: RetentionContext,
    application: Application,
    reason: StoredReasonCode,
) -> DecisionRecord:
    """The only trace of an application once it is deleted (SPEC 5)."""
    retention = timedelta(days=context.decision_retention_days)
    return DecisionRecord(
        job_id=application.job_id,
        raw_hash=application.raw_hash,
        reason_code=reason.value,
        decided_at=context.now,
        expires_at=context.now + retention,
        application_token=application.draft_token,
    )


def _ranked_applications(session: Session, job_id: int) -> list[Application]:
    """The `ranked` applications of a posting, oldest first."""
    statement = (
        select(Application)
        .where(
            Application.job_id == job_id,
            Application.status == ApplicationStatus.RANKED,
        )
        .order_by(Application.id)
    )
    return list(session.scalars(statement))


def _remove_application(context: RetentionContext, application: Application) -> None:
    """Delete an application, then its profile if nothing else references it."""
    profile_id = application.profile_id
    context.session.delete(application)
    context.session.flush()
    deleted_rows = APPLICATION_ROW_COUNT + _release_profile(context, profile_id)
    _count_deleted(context, deleted_rows)


def _release_profile(context: RetentionContext, profile_id: int) -> int:
    """Profile reference count (SPEC 8.3): delete the profile when it is unused.

    Returns:
        The number of rows deleted (profile, sections and contact), 0 when the
        profile is still referenced.
    """
    session = context.session
    if _count_where(session, Application.id, Application.profile_id == profile_id):
        return NO_REFERENCES
    profile = session.get(CandidateProfile, profile_id)
    if profile is None:
        return NO_REFERENCES
    section_count = _count_where(session, Section.id, Section.profile_id == profile_id)
    contact_count = _count_where(
        session, CandidateContact.id, CandidateContact.profile_id == profile_id
    )
    session.delete(profile)
    session.flush()
    return PROFILE_ROW_COUNT + section_count + contact_count


def _count_where(session: Session, column: object, condition: object) -> int:
    """Number of rows whose `column` is not null and that satisfy `condition`."""
    return session.execute(select(func.count(column)).where(condition)).scalar_one()


def _count_deleted(context: RetentionContext, deleted_rows: int) -> None:
    """Add deleted rows to the green panel counter."""
    if deleted_rows:
        increment_counter(context.session, CounterName.RECORDS_DELETED, deleted_rows)
