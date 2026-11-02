"""Retention rules R1 (one draft), R2, R3 and the profile reference count.

All objects are built by hand, so these tests do not depend on the pipeline.
The clock is injected through `RetentionContext.now`.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

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
    SectionType,
)
from app.privacy.reasons import ResponseOnlyReasonCode, StoredReasonCode
from app.privacy.retention import (
    ApplicationNotDraftError,
    PostingNotFoundError,
    RetentionContext,
    close_posting,
    discard_replaced_sections,
    eliminate_application,
    expire_draft,
    is_draft_expired,
)

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
CREATED = NOW - timedelta(hours=30)
DECISION_RETENTION_DAYS = 30
DRAFT_TTL_HOURS = 24
SECTIONS_PER_PROFILE = 2
ROWS_OF_A_FULL_PROFILE = 1 + SECTIONS_PER_PROFILE + 1


def make_context(session: Session, now: datetime = NOW) -> RetentionContext:
    """A retention context with the default decision period."""
    return RetentionContext(session, now, DECISION_RETENTION_DAYS)


def make_posting(session: Session, title: str) -> JobPosting:
    """An open posting without requirements."""
    posting = JobPosting(
        title=title,
        description="d",
        status=JobStatus.OPEN,
        created_at=CREATED,
        required_languages=[],
        remote_ok=True,
        relevant_categories=[],
    )
    session.add(posting)
    session.flush()
    return posting


def make_profile(session: Session, raw_hash: str) -> CandidateProfile:
    """A profile with two sections and a contact row."""
    profile = CandidateProfile(
        raw_hash=raw_hash,
        profile_json={},
        languages=[],
        parser_version="1",
        created_at=CREATED,
        last_used_at=CREATED,
    )
    profile.sections = [
        Section(type=SectionType.EXPERIENCE, ordinal=0, text="first"),
        Section(type=SectionType.SKILLS, ordinal=1, text="second"),
    ]
    profile.contact = CandidateContact(name="n", email="e", phone="p")
    session.add(profile)
    session.flush()
    return profile


def make_application(
    session: Session,
    posting: JobPosting,
    profile: CandidateProfile,
    status: ApplicationStatus = ApplicationStatus.RANKED,
    created_at: datetime = CREATED,
) -> Application:
    """An application with a token derived from its ids."""
    application = Application(
        job_id=posting.id,
        profile_id=profile.id,
        raw_hash=profile.raw_hash,
        status=status,
        draft_token=f"token-{posting.id}-{profile.id}",
        created_at=created_at,
    )
    session.add(application)
    session.flush()
    return application


def count_rows(session: Session, model: type) -> int:
    """Number of rows of a model."""
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_r2_writes_a_decision_record_and_deletes_the_application(
    demo_session: Session,
) -> None:
    """The record carries reason, token, decision time and expiry."""
    posting = make_posting(demo_session, "A")
    profile = make_profile(demo_session, "hash-1")
    application = make_application(demo_session, posting, profile, ApplicationStatus.DRAFT)
    token = application.draft_token

    eliminate_application(make_context(demo_session), application, StoredReasonCode.MIN_YEARS)

    record = demo_session.scalars(select(DecisionRecord)).one()
    assert record.reason_code == StoredReasonCode.MIN_YEARS.value
    assert record.application_token == token
    assert (record.job_id, record.raw_hash) == (posting.id, "hash-1")
    assert record.decided_at == NOW
    assert record.expires_at == NOW + timedelta(days=DECISION_RETENTION_DAYS)
    assert count_rows(demo_session, Application) == 0


def test_r2_deletes_the_profile_with_sections_and_contact_when_unreferenced(
    demo_session: Session,
) -> None:
    """The last application takes the whole profile with it."""
    posting = make_posting(demo_session, "A")
    application = make_application(
        demo_session, posting, make_profile(demo_session, "hash-1")
    )
    eliminate_application(
        make_context(demo_session), application, StoredReasonCode.LANG_MISSING
    )
    assert count_rows(demo_session, CandidateProfile) == 0
    assert count_rows(demo_session, Section) == 0
    assert count_rows(demo_session, CandidateContact) == 0


def test_profile_with_two_postings_survives_the_first_elimination(
    demo_session: Session,
) -> None:
    """Refcount: deleted only after the second application is gone."""
    first_posting = make_posting(demo_session, "A")
    second_posting = make_posting(demo_session, "B")
    profile = make_profile(demo_session, "hash-1")
    first = make_application(demo_session, first_posting, profile)
    second = make_application(demo_session, second_posting, profile)
    context = make_context(demo_session)

    eliminate_application(context, first, StoredReasonCode.MIN_YEARS)
    assert count_rows(demo_session, CandidateProfile) == 1
    assert count_rows(demo_session, Section) == SECTIONS_PER_PROFILE

    eliminate_application(context, second, StoredReasonCode.MIN_YEARS)
    assert count_rows(demo_session, CandidateProfile) == 0
    assert count_rows(demo_session, Section) == 0
    assert count_rows(demo_session, CandidateContact) == 0


def test_r2_rejects_a_response_only_reason_and_changes_nothing(
    demo_session: Session,
) -> None:
    """`DUPLICATE_APPLICATION` is never stored."""
    posting = make_posting(demo_session, "A")
    application = make_application(
        demo_session, posting, make_profile(demo_session, "hash-1")
    )
    with pytest.raises(ValueError):
        eliminate_application(
            make_context(demo_session),
            application,
            ResponseOnlyReasonCode.DUPLICATE_APPLICATION,  # type: ignore[arg-type]
        )
    assert count_rows(demo_session, Application) == 1
    assert count_rows(demo_session, DecisionRecord) == 0


def test_r3_not_selected_for_ranked_only_and_draft_is_untouched(
    demo_session: Session,
) -> None:
    """Closing turns ranked applications into decisions; drafts and shortlisted stay."""
    posting = make_posting(demo_session, "A")
    ranked = make_application(demo_session, posting, make_profile(demo_session, "h1"))
    draft = make_application(
        demo_session, posting, make_profile(demo_session, "h2"), ApplicationStatus.DRAFT
    )
    shortlisted = make_application(
        demo_session, posting, make_profile(demo_session, "h3"), ApplicationStatus.SHORTLISTED
    )
    ranked_token = ranked.draft_token

    not_selected = close_posting(make_context(demo_session), posting.id)

    assert not_selected == 1
    assert posting.status is JobStatus.CLOSED
    assert posting.closed_at == NOW
    record = demo_session.scalars(select(DecisionRecord)).one()
    assert record.reason_code == StoredReasonCode.NOT_SELECTED.value
    assert record.application_token == ranked_token
    remaining = {application.id for application in demo_session.scalars(select(Application))}
    assert remaining == {draft.id, shortlisted.id}
    assert count_rows(demo_session, CandidateProfile) == 2


def test_close_posting_is_idempotent(demo_session: Session) -> None:
    """A second close changes nothing: no new record, same `closed_at`."""
    posting = make_posting(demo_session, "A")
    make_application(demo_session, posting, make_profile(demo_session, "h1"))
    context = make_context(demo_session)
    close_posting(context, posting.id)

    later = make_context(demo_session, NOW + timedelta(days=1))
    assert close_posting(later, posting.id) == 0
    assert posting.closed_at == NOW
    assert count_rows(demo_session, DecisionRecord) == 1


def test_close_posting_of_unknown_id_fails(demo_session: Session) -> None:
    """A typed error names the missing posting."""
    with pytest.raises(PostingNotFoundError):
        close_posting(make_context(demo_session), 999)


def test_eliminating_an_already_eliminated_application_is_a_no_op(
    demo_session: Session,
) -> None:
    """No second decision record and no double counting."""
    posting = make_posting(demo_session, "A")
    application = make_application(
        demo_session, posting, make_profile(demo_session, "h1")
    )
    context = make_context(demo_session)
    eliminate_application(context, application, StoredReasonCode.MIN_YEARS)
    deleted_once = read_counters(demo_session).records_deleted

    eliminate_application(context, application, StoredReasonCode.MIN_YEARS)

    assert count_rows(demo_session, DecisionRecord) == 1
    assert read_counters(demo_session).records_deleted == deleted_once


def test_every_deleted_row_adds_to_records_deleted(demo_session: Session) -> None:
    """Application, profile, its sections and its contact are each counted."""
    posting = make_posting(demo_session, "A")
    application = make_application(
        demo_session, posting, make_profile(demo_session, "h1")
    )
    eliminate_application(
        make_context(demo_session), application, StoredReasonCode.MIN_YEARS
    )
    assert read_counters(demo_session).records_deleted == 1 + ROWS_OF_A_FULL_PROFILE


def test_a_shared_profile_counts_only_the_application(demo_session: Session) -> None:
    """While the profile survives, only the application row is counted."""
    first_posting = make_posting(demo_session, "A")
    second_posting = make_posting(demo_session, "B")
    profile = make_profile(demo_session, "h1")
    first = make_application(demo_session, first_posting, profile)
    make_application(demo_session, second_posting, profile)
    eliminate_application(make_context(demo_session), first, StoredReasonCode.MIN_YEARS)
    assert read_counters(demo_session).records_deleted == 1


@pytest.mark.parametrize(
    ("age_hours", "expected"),
    [(DRAFT_TTL_HOURS - 1, False), (DRAFT_TTL_HOURS, False), (DRAFT_TTL_HOURS + 1, True)],
)
def test_draft_expiry_condition(
    demo_session: Session, age_hours: int, expected: bool
) -> None:
    """`created_at + ttl < now`; exactly at the limit is not yet expired."""
    posting = make_posting(demo_session, "A")
    application = make_application(
        demo_session,
        posting,
        make_profile(demo_session, "h1"),
        ApplicationStatus.DRAFT,
        created_at=NOW - timedelta(hours=age_hours),
    )
    assert is_draft_expired(application, NOW, DRAFT_TTL_HOURS) is expected


def test_only_drafts_expire(demo_session: Session) -> None:
    """A ranked application is never expired by the draft rule."""
    posting = make_posting(demo_session, "A")
    ranked = make_application(demo_session, posting, make_profile(demo_session, "h1"))
    assert is_draft_expired(ranked, NOW, DRAFT_TTL_HOURS) is False
    with pytest.raises(ApplicationNotDraftError):
        expire_draft(make_context(demo_session), ranked)


def test_expiring_a_draft_writes_no_record_and_releases_the_profile(
    demo_session: Session,
) -> None:
    """R1: no decision record; the unreferenced profile goes; idempotent."""
    posting = make_posting(demo_session, "A")
    draft = make_application(
        demo_session, posting, make_profile(demo_session, "h1"), ApplicationStatus.DRAFT
    )
    context = make_context(demo_session)

    expire_draft(context, draft)
    expire_draft(context, draft)

    assert count_rows(demo_session, Application) == 0
    assert count_rows(demo_session, DecisionRecord) == 0
    assert count_rows(demo_session, CandidateProfile) == 0
    assert read_counters(demo_session).records_deleted == 1 + ROWS_OF_A_FULL_PROFILE


def test_expiring_a_draft_keeps_a_profile_another_application_uses(
    demo_session: Session,
) -> None:
    """The refcount also protects the profile on the R1 path."""
    first_posting = make_posting(demo_session, "A")
    second_posting = make_posting(demo_session, "B")
    profile = make_profile(demo_session, "h1")
    draft = make_application(demo_session, first_posting, profile, ApplicationStatus.DRAFT)
    make_application(demo_session, second_posting, profile)
    expire_draft(make_context(demo_session), draft)
    assert count_rows(demo_session, CandidateProfile) == 1


def test_discarding_replaced_sections_counts_each_row(demo_session: Session) -> None:
    """Sections removed by an edit are deleted here and counted."""
    profile = make_profile(demo_session, "h1")
    removed = list(profile.sections)[:1]
    discard_replaced_sections(make_context(demo_session), removed)
    assert count_rows(demo_session, Section) == SECTIONS_PER_PROFILE - 1
    assert read_counters(demo_session).records_deleted == 1
