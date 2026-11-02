"""Close and finalize endpoints, repeat applications after R5, and start-up retention (T-14).

Fake embedder and fake LLM; no GPU, no Ollama.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from candidate_support import (
    create_open_posting,
    posting_status,
    stored_applications,
    stored_decisions,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import ControlledEmbedder, build_cv_pdf
from sqlalchemy import select
from test_candidate_flow import confirmed_application
from web_support import log_in

import app.main as main_module
from app.config import Settings
from app.green.counters import read_counters
from app.main import create_app
from app.models import Application, ApplicationStatus, CandidateProfile, JobPosting
from app.privacy.retention import RetentionPolicy, run
from app.web import messages, paths
from app.web.dependencies import get_now
from conftest import FakeEmbedder

CONFLICT = 409
NOT_FOUND = 404
REDIRECT = 303
LONG_AGO = datetime(2020, 1, 1, tzinfo=UTC)
BEYOND_RETENTION = timedelta(days=31)
UNKNOWN_JOB_ID = 9999


@pytest.fixture
def employer(candidate_app: FastAPI, candidate_client: TestClient) -> Iterator[TestClient]:
    """A logged-in employer client."""
    log_in(candidate_client)
    yield candidate_client
    candidate_app.dependency_overrides.clear()


@pytest.fixture
def backend_job(candidate_app: FastAPI, candidate_embedder: ControlledEmbedder) -> int:
    """An open posting without hard filters."""
    return create_open_posting(candidate_app, candidate_embedder, "Backend Developer")


def shortlist_all(application: FastAPI) -> None:
    """Turn every ranked application into a shortlisted one (T-15 does this in the UI)."""
    with application.state.session_factory() as session:
        for row in session.scalars(select(Application)):
            row.status = ApplicationStatus.SHORTLISTED
        session.commit()


def count_profiles(application: FastAPI) -> int:
    """Number of stored candidate profiles."""
    with application.state.session_factory() as session:
        return len(list(session.scalars(select(CandidateProfile))))


# --- authentication ---------------------------------------------------------


@pytest.mark.parametrize("path_builder", [paths.employer_close_path, paths.employer_finalize_path])
def test_actions_without_login_are_redirected_and_change_nothing(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int, path_builder
) -> None:
    """The guard covers both endpoints."""
    response = candidate_client.post(path_builder(backend_job))

    assert response.status_code == REDIRECT
    assert response.headers["location"] == paths.EMPLOYER_LOGIN_PATH
    assert posting_status(candidate_app, backend_job) == "open"


# --- close ------------------------------------------------------------------


def test_close_runs_r3_and_sc22_and_redirects_with_a_notice(
    employer: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Ranked becomes NOT_SELECTED, the draft goes with no record, the list says so."""
    confirmed_application(employer, backend_job, build_cv_pdf("alpha"))
    upload_cv(employer, backend_job, build_cv_pdf("beta"))  # stays a draft

    response = employer.post(paths.employer_close_path(backend_job))

    assert response.status_code == REDIRECT
    assert response.headers["location"] == paths.employer_home_with_notice("closed")
    assert posting_status(candidate_app, backend_job) == "closed"
    assert stored_applications(candidate_app) == []
    assert [record.reason_code for record in stored_decisions(candidate_app)] == ["NOT_SELECTED"]
    assert count_profiles(candidate_app) == 0
    assert messages.POSTING_NOTICES["closed"] in employer.get(response.headers["location"]).text


def test_close_unknown_posting_is_not_found(employer: TestClient) -> None:
    """No such posting, no redirect."""
    assert employer.post(paths.employer_close_path(UNKNOWN_JOB_ID)).status_code == NOT_FOUND


# --- finalize ---------------------------------------------------------------


def test_finalize_of_an_open_posting_is_rejected_and_unchanged(
    employer: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """T14-2: a Turkish error, status code 409, and nothing changes."""
    confirmed_application(employer, backend_job, build_cv_pdf("alpha"))

    response = employer.post(paths.employer_finalize_path(backend_job))

    assert response.status_code == CONFLICT
    assert messages.FINALIZE_REQUIRES_CLOSED in response.text
    assert posting_status(candidate_app, backend_job) == "open"
    assert len(stored_applications(candidate_app)) == 1


def test_finalize_marks_the_posting_and_releases_the_shortlist(
    employer: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Closed then finalized: status and time set, shortlisted rows gone, counter up."""
    confirmed_application(employer, backend_job, build_cv_pdf("alpha"))
    shortlist_all(candidate_app)
    employer.post(paths.employer_close_path(backend_job))
    assert len(stored_applications(candidate_app)) == 1  # shortlisted survive R3
    before = records_deleted(candidate_app)

    response = employer.post(paths.employer_finalize_path(backend_job))

    assert response.status_code == REDIRECT
    assert posting_status(candidate_app, backend_job) == "finalized"
    with candidate_app.state.session_factory() as session:
        assert session.get(JobPosting, backend_job).finalized_at is not None
    assert stored_applications(candidate_app) == []
    assert count_profiles(candidate_app) == 0
    assert stored_decisions(candidate_app) == []
    assert records_deleted(candidate_app) > before


def test_buttons_show_only_when_the_action_is_valid(
    employer: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Open: close button; closed: finalize button; finalized: neither."""
    close_action = f'action="{paths.employer_close_path(backend_job)}"'
    finalize_action = f'action="{paths.employer_finalize_path(backend_job)}"'

    page = employer.get(paths.EMPLOYER_HOME_PATH).text
    assert close_action in page and finalize_action not in page

    employer.post(paths.employer_close_path(backend_job))
    page = employer.get(paths.EMPLOYER_HOME_PATH).text
    assert finalize_action in page and close_action not in page

    employer.post(paths.employer_finalize_path(backend_job))
    page = employer.get(paths.EMPLOYER_HOME_PATH).text
    assert finalize_action not in page and close_action not in page


# --- status pages and repeat applications -----------------------------------


def test_status_page_says_record_deleted_after_r4(
    employer: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """T14-5: the shortlisted application's page shows the deletion message."""
    token = confirmed_application(employer, backend_job, build_cv_pdf("alpha"))
    shortlist_all(candidate_app)
    employer.post(paths.employer_close_path(backend_job))
    employer.post(paths.employer_finalize_path(backend_job))

    assert "silindi" in employer.get(f"/applications/{token}").text


def test_repeat_application_blocked_until_the_decision_expires(
    candidate_client: TestClient,
    candidate_app: FastAPI,
    candidate_embedder: ControlledEmbedder,
) -> None:
    """R5: blocked within the period, allowed after `run` deletes the record."""
    job_id = create_open_posting(
        candidate_app, candidate_embedder, "Senior Backend", min_years=10.0
    )
    pdf = build_cv_pdf("alpha")
    token = confirmed_application(candidate_client, job_id, pdf, years_experience="2")
    assert len(stored_decisions(candidate_app)) == 1
    assert upload_cv(candidate_client, job_id, pdf).status_code == CONFLICT

    policy = RetentionPolicy.from_settings(candidate_app.state.settings)
    after_expiry = datetime.now(UTC) + BEYOND_RETENTION
    with candidate_app.state.session_factory() as session:
        report = run(session, after_expiry, policy)
        session.commit()

    assert report.expired_decisions == 1
    assert stored_decisions(candidate_app) == []
    candidate_app.dependency_overrides[get_now] = lambda: after_expiry
    assert upload_cv(candidate_client, job_id, pdf).status_code == REDIRECT
    assert "silindi" in candidate_client.get(f"/applications/{token}").text


# --- start-up ---------------------------------------------------------------


def seed_old_draft(settings: Settings, embedder: FakeEmbedder) -> None:
    """Create the database with one ancient draft, as a previous run would leave it."""
    application = create_app(settings, embedder)
    with TestClient(application):
        job_id = create_open_posting(application, embedder, "Old")  # type: ignore[arg-type]
        with application.state.session_factory() as session:
            profile = CandidateProfile(
                raw_hash="h",
                profile_json={},
                languages=[],
                parser_version="1",
                created_at=LONG_AGO,
                last_used_at=LONG_AGO,
            )
            session.add(profile)
            session.flush()
            session.add(
                Application(
                    job_id=job_id,
                    profile_id=profile.id,
                    raw_hash="h",
                    status=ApplicationStatus.DRAFT,
                    draft_token="old-token",
                    created_at=LONG_AGO,
                )
            )
            session.commit()


def test_startup_runs_retention(test_settings: Settings, fake_embedder: FakeEmbedder) -> None:
    """An expired draft left in the database is gone once the app has started."""
    seed_old_draft(test_settings, fake_embedder)

    application = create_app(test_settings, fake_embedder)
    with TestClient(application):
        assert stored_applications(application) == []
        assert records_deleted(application) > 0


def test_startup_failure_does_not_stop_the_app(
    test_settings: Settings,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T14-3: the app serves requests; the log has the exception class only."""
    secret_message = "cv text of a candidate"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(secret_message)

    monkeypatch.setattr(main_module, "run_retention", fail)
    application = create_app(test_settings, fake_embedder)

    with caplog.at_level("ERROR"), TestClient(application) as client:
        assert client.get("/health").status_code == 200

    assert "RuntimeError" in caplog.text
    assert secret_message not in caplog.text


def records_deleted(application: FastAPI) -> int:
    """The `records_deleted` counter of the app's database."""
    with application.state.session_factory() as session:
        return read_counters(session).records_deleted
