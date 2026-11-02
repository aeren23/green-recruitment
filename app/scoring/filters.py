"""Hard filters of SPEC 6.4, evaluated in SQL over the profile's columns.

Main principle: an unknown value is never a reason for elimination. A filter
fails only when both sides of the comparison are known.

Stored values are assumed to be normalized already by whoever writes them
(lowercase English language names, a normalized location string, `None` for an
unknown location). The comparisons here are exact and do no normalization.
"""

from sqlalchemy import ColumnElement, and_, exists, func, not_, select, true
from sqlalchemy.orm import Session

from app.models import CandidateProfile, JobPosting
from app.privacy.reasons import StoredReasonCode

SQL_JSON_EACH_VALUE_COLUMN = "value"
REQUIRED_LANGUAGES_ALIAS = "required_language"
CANDIDATE_LANGUAGES_ALIAS = "candidate_language"
EMPTY_LIST_LENGTH = 0


class FilterTargetNotFoundError(LookupError):
    """The job posting or the candidate profile to filter does not exist."""


def _has_items(json_list_column: ColumnElement) -> ColumnElement[bool]:
    """SQL condition: the JSON list column holds at least one item."""
    return func.json_array_length(json_list_column) > EMPTY_LIST_LENGTH


def language_filter_fails() -> ColumnElement[bool]:
    """Fails when languages are required and known, but none of them is shared.

    The rule is "any of": one matching language is enough to pass.
    """
    required = (
        func.json_each(JobPosting.required_languages)
        .table_valued(SQL_JSON_EACH_VALUE_COLUMN)
        .alias(REQUIRED_LANGUAGES_ALIAS)
    )
    known = (
        func.json_each(CandidateProfile.languages)
        .table_valued(SQL_JSON_EACH_VALUE_COLUMN)
        .alias(CANDIDATE_LANGUAGES_ALIAS)
    )
    shares_a_language = exists().where(required.c.value == known.c.value)
    return and_(
        _has_items(JobPosting.required_languages),
        _has_items(CandidateProfile.languages),
        not_(shares_a_language),
    )


def location_filter_fails() -> ColumnElement[bool]:
    """Fails when the posting is not remote and both locations are known and differ."""
    return and_(
        JobPosting.remote_ok.is_(False),
        JobPosting.location.is_not(None),
        CandidateProfile.location.is_not(None),
        JobPosting.location != CandidateProfile.location,
    )


def experience_filter_fails() -> ColumnElement[bool]:
    """Fails when both values are known and the candidate has fewer years than required.

    Exactly the required number of years passes.
    """
    return and_(
        JobPosting.min_years.is_not(None),
        CandidateProfile.years_experience.is_not(None),
        CandidateProfile.years_experience < JobPosting.min_years,
    )


def _failing_conditions() -> tuple[tuple[StoredReasonCode, ColumnElement[bool]], ...]:
    """The filters in evaluation order: language, location, experience."""
    return (
        (StoredReasonCode.LANG_MISSING, language_filter_fails()),
        (StoredReasonCode.LOCATION_MISMATCH, location_filter_fails()),
        (StoredReasonCode.MIN_YEARS, experience_filter_fails()),
    )


def _missing_row_error(
    session: Session, job_id: int, profile_id: int
) -> FilterTargetNotFoundError:
    """Error naming which of the two rows does not exist."""
    if session.get(JobPosting, job_id) is None:
        return FilterTargetNotFoundError(f"Job posting {job_id} does not exist.")
    return FilterTargetNotFoundError(f"Candidate profile {profile_id} does not exist.")


def evaluate_hard_filters(
    session: Session, job_id: int, profile_id: int
) -> StoredReasonCode | None:
    """Run the SPEC 6.4 hard filters of a posting against a profile.

    All three conditions are computed by one SQL statement. They are checked in
    a fixed order (language, location, experience), so a candidate failing
    several filters always gets the first one's reason code.

    Args:
        session: Session on the demo database.
        job_id: Id of the job posting whose filters apply.
        profile_id: Id of the candidate profile to check.

    Returns:
        The stored reason code of the first failing filter, or None if the
        candidate passes all of them.

    Raises:
        FilterTargetNotFoundError: If the posting or the profile does not exist.
    """
    conditions = _failing_conditions()
    statement = (
        select(*(condition for _, condition in conditions))
        .select_from(JobPosting)
        .join(CandidateProfile, true())
        .where(JobPosting.id == job_id, CandidateProfile.id == profile_id)
    )
    row = session.execute(statement).first()
    if row is None:
        raise _missing_row_error(session, job_id, profile_id)
    for (reason_code, _), has_failed in zip(conditions, row, strict=True):
        if has_failed:
            return reason_code
    return None
