"""Retention of SPEC section 8.3: rules R1 to R5 and the profile reference count.

R2 (elimination) and R3 (closing a posting) are event-driven; R1, R4 and R5 are
time-driven and applied together by `run()`. Every deletion in the application
goes through this module, so the reference count and the decision records
cannot be bypassed.

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

from app.config import Settings
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


class PostingNotClosedError(RuntimeError):
    """A posting that is still open was asked to be finalized (T14-2)."""


class ApplicationNotDraftError(RuntimeError):
    """An operation that applies to drafts only was given another application."""


@dataclass(frozen=True)
class RetentionPolicy:
    """The retention periods that `run()` applies.

    Attributes:
        draft_ttl_hours: `DRAFT_TTL_HOURS` (R1).
        shortlist_retention_days: `SHORTLIST_RETENTION_DAYS` (R4).
        decision_retention_days: `DECISION_RETENTION_DAYS` (sets `expires_at`).
    """

    draft_ttl_hours: int
    shortlist_retention_days: int
    decision_retention_days: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetentionPolicy":
        """The policy configured in the application settings."""
        return cls(
            draft_ttl_hours=settings.draft_ttl_hours,
            shortlist_retention_days=settings.shortlist_retention_days,
            decision_retention_days=settings.decision_retention_days,
        )


@dataclass(frozen=True)
class RetentionReport:
    """What one `run()` deleted, per rule (applications or decision records).

    Attributes:
        expired_drafts: R1, drafts past their time to live.
        released_shortlists: R4, shortlisted applications of finished postings.
        expired_decisions: R5, decision records past `expires_at`.
    """

    expired_drafts: int = 0
    released_shortlists: int = 0
    expired_decisions: int = 0

    @property
    def total(self) -> int:
        """Deleted applications and decision records over all rules."""
        return self.expired_drafts + self.released_shortlists + self.expired_decisions


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
    deleted. Drafts of the posting are deleted without a record (SC-22: they can
    no longer be confirmed). Shortlisted applications are left alone. Idempotent:
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
    _discard_drafts_of_posting(context, job_id)
    logger.info("Posting %d closed: %d ranked applications not selected", job_id, len(ranked))
    return len(ranked)


def finalize_posting(context: RetentionContext, job_id: int) -> None:
    """Mark a closed posting `finalized`, which makes R4 apply to it (T14-2).

    Only a closed posting can be finalized, so R3 has already written the
    `NOT_SELECTED` records. Finalizing a finalized posting changes nothing.

    Args:
        context: Session, injected clock and decision retention period.
        job_id: Id of the posting to finalize.

    Raises:
        PostingNotFoundError: If the posting does not exist.
        PostingNotClosedError: If the posting is still open. Nothing is changed.
    """
    posting = context.session.get(JobPosting, job_id)
    if posting is None:
        raise PostingNotFoundError(f"Job posting {job_id} does not exist.")
    if posting.status is JobStatus.OPEN:
        raise PostingNotClosedError(f"Job posting {job_id} must be closed first.")
    if posting.status is JobStatus.CLOSED:
        posting.status = JobStatus.FINALIZED
        posting.finalized_at = context.now
        logger.info("Posting %d finalized", job_id)


def run(session: Session, now: datetime, policy: RetentionPolicy) -> RetentionReport:
    """Apply the time-driven rules R1, R4 and R5 (R2 and R3 are event-driven).

    Idempotent: a second call with the same `now` deletes nothing. Does not
    commit; the caller owns the transaction.

    Args:
        session: Session whose transaction the caller owns.
        now: The injected clock (timezone-aware UTC).
        policy: The retention periods.

    Returns:
        How many rows each rule deleted.
    """
    context = RetentionContext(session, now, policy.decision_retention_days)
    return RetentionReport(
        expired_drafts=_expire_drafts(context, policy.draft_ttl_hours),
        released_shortlists=_release_shortlists(
            context, policy.shortlist_retention_days
        ),
        expired_decisions=_expire_decisions(context),
    )


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


def _expire_drafts(context: RetentionContext, ttl_hours: int) -> int:
    """Rule R1 for every draft whose `created_at + ttl_hours < now`."""
    cutoff = context.now - timedelta(hours=ttl_hours)
    statement = select(Application).where(
        Application.status == ApplicationStatus.DRAFT,
        Application.created_at < cutoff,
    )
    drafts = list(context.session.scalars(statement))
    for draft in drafts:
        _remove_application(context, draft)
    return len(drafts)


def _discard_drafts_of_posting(context: RetentionContext, job_id: int) -> None:
    """SC-22: delete the drafts of a posting that was just closed, with no record."""
    statement = select(Application).where(
        Application.job_id == job_id,
        Application.status == ApplicationStatus.DRAFT,
    )
    for draft in list(context.session.scalars(statement)):
        _remove_application(context, draft)


def _release_shortlists(context: RetentionContext, retention_days: int) -> int:
    """Rule R4: delete shortlisted applications of finished postings, no record."""
    cutoff = context.now - timedelta(days=retention_days)
    finished_jobs = select(JobPosting.id).where(
        (JobPosting.status == JobStatus.FINALIZED)
        | ((JobPosting.status == JobStatus.CLOSED) & (JobPosting.closed_at < cutoff))
    )
    statement = select(Application).where(
        Application.status == ApplicationStatus.SHORTLISTED,
        Application.job_id.in_(finished_jobs),
    )
    shortlisted = list(context.session.scalars(statement))
    for application in shortlisted:
        _remove_application(context, application)
    return len(shortlisted)


def _expire_decisions(context: RetentionContext) -> int:
    """Rule R5: delete decision records whose `expires_at < now`."""
    statement = select(DecisionRecord).where(DecisionRecord.expires_at < context.now)
    records = list(context.session.scalars(statement))
    for record in records:
        context.session.delete(record)
    context.session.flush()
    _count_deleted(context, len(records))
    return len(records)


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
