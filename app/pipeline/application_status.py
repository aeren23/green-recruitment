"""The candidate's status page lookup (SPEC 9.1, SC-01).

Order: the application by its token first, then the `DecisionRecord` by
`application_token`, and if neither exists the record was deleted. Only the
state of the asker's own token is read; nothing else about other candidates.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Application, ApplicationStatus, DecisionRecord, JobPosting
from app.privacy.reasons import StoredReasonCode


class StatusKind(StrEnum):
    """What the status page tells the candidate."""

    DRAFT = "draft"
    RANKED = "ranked"
    SHORTLISTED = "shortlisted"
    ELIMINATED = "eliminated"
    NOT_SELECTED = "not_selected"
    RECORD_DELETED = "record_deleted"


KIND_BY_APPLICATION_STATUS = {
    ApplicationStatus.DRAFT: StatusKind.DRAFT,
    ApplicationStatus.RANKED: StatusKind.RANKED,
    ApplicationStatus.SHORTLISTED: StatusKind.SHORTLISTED,
}


@dataclass(frozen=True)
class ApplicationStatusView:
    """The answer for one token.

    Attributes:
        kind: The state to show.
        job_title: Posting title; `None` once the record is deleted.
        reason_code: The elimination reason; only for `ELIMINATED`.
    """

    kind: StatusKind
    job_title: str | None = None
    reason_code: StoredReasonCode | None = None


def look_up_status(session: Session, token: str, now: datetime) -> ApplicationStatusView:
    """Find the state behind a token: application, decision record, or deleted.

    Args:
        session: Session to read with; nothing is written.
        token: The application token from the URL.
        now: The injected clock; a decision record past its expiry counts as deleted.

    Returns:
        The status view; `RECORD_DELETED` when nothing carries the token.
    """
    application = session.scalar(
        select(Application).where(Application.draft_token == token)
    )
    if application is not None:
        title = job_title(session, application.job_id)
        return ApplicationStatusView(KIND_BY_APPLICATION_STATUS[application.status], title)
    record = session.scalar(
        select(DecisionRecord).where(DecisionRecord.application_token == token)
    )
    if record is None or record.expires_at < now:
        return ApplicationStatusView(StatusKind.RECORD_DELETED)
    return decision_view(session, record)


def decision_view(session: Session, record: DecisionRecord) -> ApplicationStatusView:
    """The status view of a stored decision."""
    reason = StoredReasonCode(record.reason_code)
    title = job_title(session, record.job_id)
    if reason is StoredReasonCode.NOT_SELECTED:
        return ApplicationStatusView(StatusKind.NOT_SELECTED, title)
    return ApplicationStatusView(StatusKind.ELIMINATED, title, reason)


def job_title(session: Session, job_id: int) -> str | None:
    """The title of a posting, `None` if it does not exist."""
    posting = session.get(JobPosting, job_id)
    return None if posting is None else posting.title
