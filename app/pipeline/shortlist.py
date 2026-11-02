"""Shortlisting (SPEC 3 step 5, SC-04): the only way contact data becomes visible."""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.models import Application, ApplicationStatus, JobPosting, JobStatus

logger = logging.getLogger(__name__)


class ApplicationNotFoundError(LookupError):
    """The application does not exist."""


class ApplicationNotRankedError(RuntimeError):
    """Only a `ranked` application can be shortlisted."""


class PostingNotOpenError(RuntimeError):
    """A closed or finalized posting takes no more shortlisting."""


def shortlist_application(session: Session, application_id: int, now: datetime) -> None:
    """Move a ranked application to `shortlisted`; the caller commits.

    Args:
        session: Session to work on.
        application_id: Id of the application.
        now: The injected clock, stored as `decided_at`.

    Raises:
        ApplicationNotFoundError: If the application does not exist.
        ApplicationNotRankedError: If its status is not `ranked`.
        PostingNotOpenError: If its posting is closed or finalized.
    """
    application = session.get(Application, application_id)
    if application is None:
        raise ApplicationNotFoundError(f"Application {application_id} does not exist.")
    if application.status is not ApplicationStatus.RANKED:
        raise ApplicationNotRankedError(f"Application {application_id} is not ranked.")
    posting = session.get(JobPosting, application.job_id)
    if posting is None or posting.status is not JobStatus.OPEN:
        raise PostingNotOpenError(f"Posting of application {application_id} is not open.")
    application.status = ApplicationStatus.SHORTLISTED
    application.decided_at = now
    logger.info("Application id=%d shortlisted", application_id)
