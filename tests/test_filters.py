"""Hard filters of SPEC 6.4 on a real temp SQLite database.

Each filter has an elimination, a pass and an "unknown" case. The "unknown is
not a reason for elimination" principle is tested on every filter.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.db import create_demo_schema, create_session_factory
from app.models import CandidateProfile, JobPosting, JobStatus
from app.privacy.reasons import StoredReasonCode
from app.scoring.filters import FilterTargetNotFoundError, evaluate_hard_filters

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
UNUSED_ID = 9999


@dataclass
class JobSpec:
    """Hard-filter fields of a posting, with permissive defaults."""

    required_languages: list[str] = field(default_factory=list)
    location: str | None = None
    remote_ok: bool = True
    min_years: float | None = None


@dataclass
class ProfileSpec:
    """Hard-filter fields of a profile, with "unknown" defaults."""

    languages: list[str] = field(default_factory=list)
    location: str | None = None
    years_experience: float | None = None


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session on a fresh demo database, closed after the test."""
    create_demo_schema(engine)
    with create_session_factory(engine)() as test_session:
        yield test_session


def add_job(session: Session, spec: JobSpec) -> int:
    """Insert a posting and return its id."""
    job = JobPosting(
        title="Backend Developer",
        description="Build services.",
        status=JobStatus.OPEN,
        created_at=NOW,
        required_languages=spec.required_languages,
        location=spec.location,
        remote_ok=spec.remote_ok,
        min_years=spec.min_years,
        relevant_categories=[],
    )
    session.add(job)
    session.commit()
    return job.id


def add_profile(session: Session, spec: ProfileSpec, raw_hash: str = "hash-a") -> int:
    """Insert a profile and return its id."""
    profile = CandidateProfile(
        raw_hash=raw_hash,
        profile_json={"sections": []},
        languages=spec.languages,
        location=spec.location,
        years_experience=spec.years_experience,
        parser_version="1",
        created_at=NOW,
        last_used_at=NOW,
    )
    session.add(profile)
    session.commit()
    return profile.id


def run_filters(session: Session, job_spec: JobSpec, profile_spec: ProfileSpec):
    """Insert one posting and one profile, then evaluate the filters."""
    job_id = add_job(session, job_spec)
    profile_id = add_profile(session, profile_spec)
    return evaluate_hard_filters(session, job_id, profile_id)


# --- language ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("required", "candidate", "expected"),
    [
        (["english"], ["german"], StoredReasonCode.LANG_MISSING),
        (["english", "german"], ["french", "turkish"], StoredReasonCode.LANG_MISSING),
        (["english"], ["english"], None),
        (["english"], ["turkish", "english"], None),
        (["english", "german"], ["german"], None),
        (["english", "german"], ["english", "german"], None),
        (["english"], [], None),
        ([], ["german"], None),
        ([], [], None),
    ],
    ids=[
        "eliminated-single",
        "eliminated-none-of-several",
        "pass-exact",
        "pass-among-several",
        "pass-any-of-required",
        "pass-all-required",
        "unknown-candidate-languages",
        "no-requirement",
        "nothing-known-or-required",
    ],
)
def test_language_filter(
    session: Session,
    required: list[str],
    candidate: list[str],
    expected: StoredReasonCode | None,
) -> None:
    """Fails only when requirements and candidate languages are both non-empty and disjoint."""
    result = run_filters(
        session, JobSpec(required_languages=required), ProfileSpec(languages=candidate)
    )
    assert result == expected


# --- location ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("remote_ok", "job_location", "candidate_location", "expected"),
    [
        (False, "denizli", "izmir", StoredReasonCode.LOCATION_MISMATCH),
        (False, "denizli", "denizli", None),
        (False, "denizli", None, None),
        (False, None, "izmir", None),
        (False, None, None, None),
        (True, "denizli", "izmir", None),
        (True, "denizli", "denizli", None),
    ],
    ids=[
        "eliminated-different-city",
        "pass-same-city",
        "unknown-candidate-location",
        "unknown-posting-location",
        "both-unknown",
        "remote-ok-differing",
        "remote-ok-same",
    ],
)
def test_location_filter(
    session: Session,
    remote_ok: bool,
    job_location: str | None,
    candidate_location: str | None,
    expected: StoredReasonCode | None,
) -> None:
    """Fails only for an on-site posting whose and candidate's known locations differ."""
    result = run_filters(
        session,
        JobSpec(remote_ok=remote_ok, location=job_location),
        ProfileSpec(location=candidate_location),
    )
    assert result == expected


# --- experience -------------------------------------------------------------


@pytest.mark.parametrize(
    ("min_years", "years_experience", "expected"),
    [
        (5.0, 4.9, StoredReasonCode.MIN_YEARS),
        (5.0, 0.0, StoredReasonCode.MIN_YEARS),
        (5.0, 5.0, None),
        (5.0, 5.1, None),
        (5.0, 12.0, None),
        (5.0, None, None),
        (None, 1.0, None),
        (None, None, None),
        (0.0, 0.0, None),
    ],
    ids=[
        "eliminated-just-below",
        "eliminated-zero-years",
        "pass-exactly-equal",
        "pass-just-above",
        "pass-far-above",
        "unknown-candidate-years",
        "no-requirement",
        "nothing-known-or-required",
        "zero-required-zero-known",
    ],
)
def test_experience_filter(
    session: Session,
    min_years: float | None,
    years_experience: float | None,
    expected: StoredReasonCode | None,
) -> None:
    """Fails only when both values are known and the candidate has fewer years."""
    result = run_filters(
        session,
        JobSpec(min_years=min_years),
        ProfileSpec(years_experience=years_experience),
    )
    assert result == expected


# --- combinations and order ---------------------------------------------------

FAILS_EVERYTHING_JOB = JobSpec(
    required_languages=["english"], location="denizli", remote_ok=False, min_years=5.0
)
FAILS_EVERYTHING_PROFILE = ProfileSpec(
    languages=["german"], location="izmir", years_experience=1.0
)


def test_all_three_filters_failing_returns_the_language_code(session: Session) -> None:
    """Evaluation order is language, then location, then experience."""
    result = run_filters(session, FAILS_EVERYTHING_JOB, FAILS_EVERYTHING_PROFILE)
    assert result == StoredReasonCode.LANG_MISSING


def test_location_and_experience_failing_returns_the_location_code(
    session: Session,
) -> None:
    """Without a language failure the location code comes before the experience one."""
    profile = ProfileSpec(
        languages=["english"], location="izmir", years_experience=1.0
    )
    assert run_filters(session, FAILS_EVERYTHING_JOB, profile) == (
        StoredReasonCode.LOCATION_MISMATCH
    )


def test_only_experience_failing_returns_the_experience_code(session: Session) -> None:
    """The last filter reports when it is the only one failing."""
    profile = ProfileSpec(
        languages=["english"], location="denizli", years_experience=1.0
    )
    assert run_filters(session, FAILS_EVERYTHING_JOB, profile) == StoredReasonCode.MIN_YEARS


def test_candidate_passing_every_filter_returns_none(session: Session) -> None:
    """A fully matching candidate passes."""
    profile = ProfileSpec(
        languages=["english", "turkish"], location="denizli", years_experience=5.0
    )
    assert run_filters(session, FAILS_EVERYTHING_JOB, profile) is None


def test_candidate_with_everything_unknown_passes_strict_posting(
    session: Session,
) -> None:
    """Unknown is never a reason for elimination, whatever the posting demands."""
    assert run_filters(session, FAILS_EVERYTHING_JOB, ProfileSpec()) is None


def test_result_is_deterministic_and_a_stored_reason_code(session: Session) -> None:
    """Repeated evaluation gives the same value, and it equals the stored string."""
    job_id = add_job(session, FAILS_EVERYTHING_JOB)
    profile_id = add_profile(session, FAILS_EVERYTHING_PROFILE)
    results = {evaluate_hard_filters(session, job_id, profile_id) for _ in range(5)}
    assert results == {StoredReasonCode.LANG_MISSING}
    assert next(iter(results)) == "LANG_MISSING"


def test_filters_use_the_given_job_and_profile_ids(session: Session) -> None:
    """With several rows present, only the requested pair is evaluated."""
    strict_job = add_job(session, JobSpec(min_years=10.0))
    lenient_job = add_job(session, JobSpec())
    junior = add_profile(session, ProfileSpec(years_experience=2.0), "hash-junior")
    senior = add_profile(session, ProfileSpec(years_experience=15.0), "hash-senior")
    assert evaluate_hard_filters(session, strict_job, junior) == StoredReasonCode.MIN_YEARS
    assert evaluate_hard_filters(session, strict_job, senior) is None
    assert evaluate_hard_filters(session, lenient_job, junior) is None


def test_language_match_is_exact_on_stored_values(session: Session) -> None:
    """Normalization is the writer's job: differing case does not match here."""
    result = run_filters(
        session,
        JobSpec(required_languages=["english"]),
        ProfileSpec(languages=["English"]),
    )
    assert result == StoredReasonCode.LANG_MISSING


# --- missing rows ---------------------------------------------------------------


def test_missing_job_raises_clear_error(session: Session) -> None:
    """An unknown posting id is an error, not a silent pass."""
    profile_id = add_profile(session, ProfileSpec())
    with pytest.raises(FilterTargetNotFoundError, match=f"Job posting {UNUSED_ID}"):
        evaluate_hard_filters(session, UNUSED_ID, profile_id)


def test_missing_profile_raises_clear_error(session: Session) -> None:
    """An unknown profile id is an error, not a silent pass."""
    job_id = add_job(session, JobSpec())
    with pytest.raises(FilterTargetNotFoundError, match=f"Candidate profile {UNUSED_ID}"):
        evaluate_hard_filters(session, job_id, UNUSED_ID)


def test_both_missing_names_the_job_first(session: Session) -> None:
    """When both rows are absent the error names the posting."""
    with pytest.raises(FilterTargetNotFoundError, match="Job posting"):
        evaluate_hard_filters(session, UNUSED_ID, UNUSED_ID)


def test_missing_row_error_is_a_lookup_error() -> None:
    """Callers may catch the standard `LookupError`."""
    assert issubclass(FilterTargetNotFoundError, LookupError)
