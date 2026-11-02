"""`retention.run` (R1, R4, R5), SC-22, finalizing and the idempotency of all rules.

The clock is injected everywhere; objects are built by hand with the helpers of
`test_retention`.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_retention import (
    CREATED,
    DECISION_RETENTION_DAYS,
    DRAFT_TTL_HOURS,
    NOW,
    ROWS_OF_A_FULL_PROFILE,
    count_rows,
    make_application,
    make_context,
    make_posting,
    make_profile,
)

from app.green.counters import read_counters
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
from app.privacy.retention import (
    PostingNotClosedError,
    PostingNotFoundError,
    RetentionPolicy,
    RetentionReport,
    close_posting,
    eliminate_application,
    finalize_posting,
    run,
)

SHORTLIST_RETENTION_DAYS = 30
POLICY = RetentionPolicy(DRAFT_TTL_HOURS, SHORTLIST_RETENTION_DAYS, DECISION_RETENTION_DAYS)
ONE_SECOND = timedelta(seconds=1)
SHORTLIST_PERIOD = timedelta(days=SHORTLIST_RETENTION_DAYS)
DECISION_PERIOD = timedelta(days=DECISION_RETENTION_DAYS)
DRAFT_PERIOD = timedelta(hours=DRAFT_TTL_HOURS)


def close_at(session: Session, posting: JobPosting, closed_at: datetime) -> None:
    """Mark a posting closed at a given time without running R3."""
    posting.status = JobStatus.CLOSED
    posting.closed_at = closed_at
    session.flush()


def add_shortlisted(session: Session, posting: JobPosting, raw_hash: str) -> Application:
    """A shortlisted application on its own profile."""
    profile = make_profile(session, raw_hash)
    return make_application(session, posting, profile, ApplicationStatus.SHORTLISTED)


def add_draft(
    session: Session, posting: JobPosting, raw_hash: str, created_at: datetime
) -> Application:
    """A draft application on its own profile."""
    profile = make_profile(session, raw_hash)
    return make_application(session, posting, profile, ApplicationStatus.DRAFT, created_at)


def add_record(
    session: Session, posting: JobPosting, raw_hash: str, expires_at: datetime
) -> None:
    """A decision record that expires at a given time."""
    session.add(
        DecisionRecord(
            job_id=posting.id,
            raw_hash=raw_hash,
            reason_code=StoredReasonCode.NOT_SELECTED.value,
            decided_at=CREATED,
            expires_at=expires_at,
            application_token=f"token-{raw_hash}",
        )
    )
    session.flush()


# --- R1 ---------------------------------------------------------------------


def test_r1_boundary_exactly_at_the_ttl_is_kept(demo_session: Session) -> None:
    """`created_at + TTL < now` is strict: a draft exactly TTL old stays."""
    posting = make_posting(demo_session, "A")
    add_draft(demo_session, posting, "h1", NOW - DRAFT_PERIOD)

    report = run(demo_session, NOW, POLICY)

    assert report.expired_drafts == 0
    assert count_rows(demo_session, Application) == 1


def test_r1_past_the_ttl_deletes_without_a_record(demo_session: Session) -> None:
    """The expired draft goes with its profile; a fresh draft stays."""
    posting = make_posting(demo_session, "A")
    add_draft(demo_session, posting, "h1", NOW - DRAFT_PERIOD - ONE_SECOND)
    fresh = add_draft(demo_session, posting, "h2", NOW)

    report = run(demo_session, NOW, POLICY)

    assert report == RetentionReport(expired_drafts=1)
    assert [row.id for row in demo_session.scalars(select(Application))] == [fresh.id]
    assert count_rows(demo_session, DecisionRecord) == 0
    assert count_rows(demo_session, CandidateProfile) == 1
    assert read_counters(demo_session).records_deleted == 1 + ROWS_OF_A_FULL_PROFILE


def test_r1_only_touches_drafts(demo_session: Session) -> None:
    """An old ranked application is not a draft and is left alone."""
    posting = make_posting(demo_session, "A")
    make_application(demo_session, posting, make_profile(demo_session, "h1"))

    assert run(demo_session, NOW + timedelta(days=5), POLICY).total == 0
    assert count_rows(demo_session, Application) == 1


# --- SC-22 ------------------------------------------------------------------


def test_closing_a_posting_deletes_its_drafts_only(demo_session: Session) -> None:
    """SC-22: drafts of the closed posting go, with no record; other postings keep theirs."""
    closing = make_posting(demo_session, "A")
    other = make_posting(demo_session, "B")
    add_draft(demo_session, closing, "h1", NOW)
    kept = add_draft(demo_session, other, "h2", NOW)

    close_posting(make_context(demo_session), closing.id)

    assert [row.id for row in demo_session.scalars(select(Application))] == [kept.id]
    assert count_rows(demo_session, DecisionRecord) == 0
    assert count_rows(demo_session, CandidateProfile) == 1


# --- R4 ---------------------------------------------------------------------


def test_r4_finalized_posting_releases_shortlist_with_contact(demo_session: Session) -> None:
    """Finalized: shortlisted application, profile, sections and contact go; no record."""
    posting = make_posting(demo_session, "A")
    add_shortlisted(demo_session, posting, "h1")
    posting.status = JobStatus.FINALIZED
    demo_session.flush()

    report = run(demo_session, NOW, POLICY)

    assert report == RetentionReport(released_shortlists=1)
    for model in (Application, CandidateProfile, Section, CandidateContact, DecisionRecord):
        assert count_rows(demo_session, model) == 0
    assert read_counters(demo_session).records_deleted == 1 + ROWS_OF_A_FULL_PROFILE


def test_r4_closed_posting_exactly_at_the_period_is_kept(demo_session: Session) -> None:
    """`closed_at + period < now` is strict."""
    posting = make_posting(demo_session, "A")
    add_shortlisted(demo_session, posting, "h1")
    close_at(demo_session, posting, NOW - SHORTLIST_PERIOD)

    assert run(demo_session, NOW, POLICY).released_shortlists == 0


def test_r4_closed_posting_past_the_period_is_released(demo_session: Session) -> None:
    """One second past the period the shortlist goes, with no record."""
    posting = make_posting(demo_session, "A")
    add_shortlisted(demo_session, posting, "h1")
    close_at(demo_session, posting, NOW - SHORTLIST_PERIOD - ONE_SECOND)

    assert run(demo_session, NOW, POLICY).released_shortlists == 1
    assert count_rows(demo_session, Application) == 0
    assert count_rows(demo_session, DecisionRecord) == 0


def test_r4_leaves_open_and_recently_closed_postings_alone(demo_session: Session) -> None:
    """An open posting and a closed one inside the period keep their shortlists."""
    open_posting = make_posting(demo_session, "A")
    recent = make_posting(demo_session, "B")
    add_shortlisted(demo_session, open_posting, "h1")
    add_shortlisted(demo_session, recent, "h2")
    close_at(demo_session, recent, NOW - timedelta(days=1))

    assert run(demo_session, NOW, POLICY).released_shortlists == 0
    assert count_rows(demo_session, Application) == 2


# --- R5 ---------------------------------------------------------------------


def test_r5_deletes_only_records_strictly_past_expiry(demo_session: Session) -> None:
    """`expires_at < now`: a record expiring exactly now is kept."""
    posting = make_posting(demo_session, "A")
    add_record(demo_session, posting, "expired", NOW - ONE_SECOND)
    add_record(demo_session, posting, "boundary", NOW)
    add_record(demo_session, posting, "future", NOW + ONE_SECOND)

    report = run(demo_session, NOW, POLICY)

    assert report == RetentionReport(expired_decisions=1)
    kept = {row.raw_hash for row in demo_session.scalars(select(DecisionRecord))}
    assert kept == {"boundary", "future"}
    assert read_counters(demo_session).records_deleted == 1


def test_r5_record_blocks_until_expiry_then_is_deleted(demo_session: Session) -> None:
    """The decision stays through its expiry instant and goes right after."""
    posting = make_posting(demo_session, "A")
    application = make_application(demo_session, posting, make_profile(demo_session, "h1"))
    eliminate_application(make_context(demo_session), application, StoredReasonCode.MIN_YEARS)
    expires_at = NOW + DECISION_PERIOD

    run(demo_session, expires_at, POLICY)
    assert count_rows(demo_session, DecisionRecord) == 1

    run(demo_session, expires_at + ONE_SECOND, POLICY)
    assert count_rows(demo_session, DecisionRecord) == 0


# --- run: idempotency, refcount ---------------------------------------------


def test_run_is_idempotent(demo_session: Session) -> None:
    """The second run deletes nothing and adds nothing to the counter."""
    posting = make_posting(demo_session, "A")
    add_draft(demo_session, posting, "h1", CREATED)
    add_shortlisted(demo_session, posting, "h2")
    posting.status = JobStatus.FINALIZED
    add_record(demo_session, posting, "old", NOW - ONE_SECOND)

    first = run(demo_session, NOW, POLICY)
    counter_after_first = read_counters(demo_session).records_deleted
    second = run(demo_session, NOW, POLICY)

    assert first == RetentionReport(1, 1, 1)
    assert second == RetentionReport()
    assert read_counters(demo_session).records_deleted == counter_after_first


def test_run_does_not_commit(demo_session: Session) -> None:
    """The caller owns the transaction: a rollback restores everything."""
    posting = make_posting(demo_session, "A")
    add_draft(demo_session, posting, "h1", CREATED)
    demo_session.commit()

    run(demo_session, NOW, POLICY)
    demo_session.rollback()

    assert count_rows(demo_session, Application) == 1


def test_profile_of_two_postings_survives_one_r4_and_goes_with_the_second(
    demo_session: Session,
) -> None:
    """The refcount in the time-driven path: deleted only with the last reference."""
    first_posting = make_posting(demo_session, "A")
    second_posting = make_posting(demo_session, "B")
    profile = make_profile(demo_session, "h1")
    make_application(demo_session, first_posting, profile, ApplicationStatus.SHORTLISTED)
    make_application(demo_session, second_posting, profile, ApplicationStatus.SHORTLISTED)
    first_posting.status = JobStatus.FINALIZED
    demo_session.flush()

    run(demo_session, NOW, POLICY)
    assert count_rows(demo_session, CandidateProfile) == 1

    second_posting.status = JobStatus.FINALIZED
    demo_session.flush()
    run(demo_session, NOW, POLICY)
    assert count_rows(demo_session, CandidateProfile) == 0
    assert count_rows(demo_session, Section) == 0
    assert count_rows(demo_session, CandidateContact) == 0


# --- finalize ---------------------------------------------------------------


def test_finalize_requires_a_closed_posting(demo_session: Session) -> None:
    """T14-2: an open posting is rejected and unchanged."""
    posting = make_posting(demo_session, "A")

    with pytest.raises(PostingNotClosedError):
        finalize_posting(make_context(demo_session), posting.id)

    assert posting.status is JobStatus.OPEN
    assert posting.finalized_at is None


def test_finalize_sets_status_and_time_and_is_idempotent(demo_session: Session) -> None:
    """A closed posting becomes finalized once; a repeat keeps `finalized_at`."""
    posting = make_posting(demo_session, "A")
    close_posting(make_context(demo_session), posting.id)

    finalize_posting(make_context(demo_session), posting.id)
    finalize_posting(make_context(demo_session, NOW + timedelta(days=1)), posting.id)

    assert posting.status is JobStatus.FINALIZED
    assert posting.finalized_at == NOW


def test_finalize_unknown_posting_fails(demo_session: Session) -> None:
    """An unknown id is a typed error."""
    with pytest.raises(PostingNotFoundError):
        finalize_posting(make_context(demo_session), 999)
