"""The background summary refresh survives concurrent deletes and stale rows.

Everything runs in the calling thread: the fake LLM's hook changes the database
from a second session on the same SQLite file, at a fixed call number, so the
outcome is deterministic. No GPU and no Ollama are needed.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest
from pipeline_support import (
    SUMMARY_REPLY,
    ControlledEmbedder,
    FakeLlm,
    RecordingTracker,
    build_cv_pdf,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import StaleDataError

from app.config import Settings
from app.models import Application, ApplicationStatus
from app.pipeline import ConfirmInput, PipelineContext, UploadInput, confirm_draft, submit_upload
from app.privacy.retention import RetentionContext, close_posting
from app.seed import ACCOUNTANT_TITLE, BACKEND_DEVELOPER_TITLE, SeedOptions, seed_postings
from app.web import background
from app.web.background import SummaryRefreshJob, refresh_summaries

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
REFERENCE_DATE = date(2026, 1, 1)
TOP_K = 3
DECISION_RETENTION_DAYS = 30
# The refresh calls the LLM for the two backend applications first, then for the
# accountant application; the hook closes the accountant posting on that third call.
CLOSING_CALL_NUMBER = 3
EXPECTED_WRITES = 2
STALE_MESSAGE = "CV TEXT THAT MUST NOT BE LOGGED"
RETRY_LOG_TEXT = "retrying after StaleDataError"
FAILURE_LOG_TEXT = "Summary refresh failed for postings"
WROTE_TWO_LOG_TEXT = f"wrote {EXPECTED_WRITES} summaries"


class HookedLlm(FakeLlm):
    """A fake LLM that runs `before_call(call_number)` before each answer."""

    def __init__(self, before_call: Callable[[int], None] | None = None) -> None:
        super().__init__()
        self.before_call = before_call

    def generate(self, prompt: str, expect_json: bool = False):
        """Run the hook with the number of this call, then answer as the fake does."""
        if self.before_call is not None:
            self.before_call(self.attempts + 1)
        return super().generate(prompt, expect_json)


@dataclass
class RefreshWorld:
    """Two postings with top-k applications waiting for a summary."""

    session_factory: sessionmaker[Session]
    settings: Settings
    embedder: ControlledEmbedder
    backend_id: int
    accountant_id: int
    backend_application_ids: list[int]
    accountant_application_id: int

    def refresh_job(self, job_ids: tuple[int, ...], llm: FakeLlm) -> SummaryRefreshJob:
        """A background job over the given postings, using the fake LLM."""
        return SummaryRefreshJob(
            session_factory=self.session_factory,
            settings=self.settings,
            embedder=self.embedder,
            tracker=RecordingTracker(),
            llm_client=llm,
            now=NOW,
            job_ids=job_ids,
        )

    def summary_of(self, application_id: int) -> str | None:
        """The summary as the database holds it now, `None` if none or the row is gone."""
        with self.session_factory() as session:
            application = session.get(Application, application_id)
            return None if application is None else application.summary

    def exists(self, application_id: int) -> bool:
        """Whether the application row is still in the database."""
        with self.session_factory() as session:
            return session.get(Application, application_id) is not None


def pipeline_context(session: Session, settings: Settings, embedder: ControlledEmbedder):
    """A pipeline context without an LLM, so uploads are ranked with no summary."""
    return PipelineContext(
        session=session,
        settings=settings,
        embedder=embedder,
        tracker=RecordingTracker(),
        now=NOW,
        reference_date=REFERENCE_DATE,
        llm_client=None,
    )


def apply_cv(
    session: Session, settings: Settings, embedder: ControlledEmbedder, job_id: int, pdf: bytes
) -> int:
    """Upload and auto-confirm one CV; returns its application id."""
    context = pipeline_context(session, settings, embedder)
    outcome = submit_upload(context, UploadInput(job_id, pdf))
    result = confirm_draft(context, ConfirmInput(outcome.draft_token))
    session.commit()
    return result.application_id


@pytest.fixture
def refresh_world(demo_session: Session, engine: Engine, test_settings: Settings) -> RefreshWorld:
    """Backend (two CVs, both top-k) and accountant (one CV), all unsummarised."""
    settings = test_settings.model_copy(update={"top_k": TOP_K})
    embedder = ControlledEmbedder()
    postings = {
        posting.title: posting
        for posting in seed_postings(demo_session, embedder, SeedOptions(now=NOW))
    }
    backend_id = postings[BACKEND_DEVELOPER_TITLE].id
    accountant_id = postings[ACCOUNTANT_TITLE].id
    backend_ids = [
        apply_cv(demo_session, settings, embedder, backend_id, build_cv_pdf("alpha", 50)),
        apply_cv(demo_session, settings, embedder, backend_id, build_cv_pdf("gamma", 30)),
    ]
    accountant_application_id = apply_cv(
        demo_session, settings, embedder, accountant_id, build_cv_pdf("beta", 60)
    )
    for application_id in [*backend_ids, accountant_application_id]:
        assert demo_session.get(Application, application_id).status is ApplicationStatus.RANKED
    return RefreshWorld(
        session_factory=sessionmaker(bind=engine),
        settings=settings,
        embedder=embedder,
        backend_id=backend_id,
        accountant_id=accountant_id,
        backend_application_ids=backend_ids,
        accountant_application_id=accountant_application_id,
    )


def close_accountant_posting(world: RefreshWorld) -> None:
    """Close the accountant posting through retention, from another session."""
    with world.session_factory() as other:
        close_posting(RetentionContext(other, NOW, DECISION_RETENTION_DAYS), world.accountant_id)
        other.commit()


def test_application_deleted_mid_refresh_is_skipped_without_a_retry(
    refresh_world: RefreshWorld, caplog: pytest.LogCaptureFixture
) -> None:
    """A row deleted during its own LLM call gets no summary; the others are written once."""
    world = refresh_world

    def close_on_third_call(call_number: int) -> None:
        if call_number == CLOSING_CALL_NUMBER:
            close_accountant_posting(world)

    llm = HookedLlm(before_call=close_on_third_call)
    job = world.refresh_job((world.backend_id, world.accountant_id), llm)
    with caplog.at_level(logging.INFO):
        refresh_summaries(job)

    assert llm.attempts == CLOSING_CALL_NUMBER
    assert RETRY_LOG_TEXT not in caplog.text
    assert FAILURE_LOG_TEXT not in caplog.text
    assert WROTE_TWO_LOG_TEXT in caplog.text
    assert not world.exists(world.accountant_application_id)
    for application_id in world.backend_application_ids:
        assert world.summary_of(application_id) == SUMMARY_REPLY


def test_stale_error_on_the_first_attempt_triggers_exactly_one_retry(
    refresh_world: RefreshWorld,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first attempt raises a stale-row error; the second one finishes the work."""
    world = refresh_world
    real_refresh = background.refresh_top_k_summaries
    attempts: list[int] = []

    def stale_on_first_attempt(context, job_ids):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise StaleDataError(STALE_MESSAGE)
        return real_refresh(context, job_ids)

    monkeypatch.setattr(background, "refresh_top_k_summaries", stale_on_first_attempt)
    llm = HookedLlm()
    with caplog.at_level(logging.INFO):
        refresh_summaries(world.refresh_job((world.backend_id,), llm))

    assert len(attempts) == 2
    assert RETRY_LOG_TEXT in caplog.text
    assert FAILURE_LOG_TEXT not in caplog.text
    assert WROTE_TWO_LOG_TEXT in caplog.text
    for application_id in world.backend_application_ids:
        assert world.summary_of(application_id) == SUMMARY_REPLY


def test_a_second_stale_error_gives_up_and_logs_only_the_class_name(
    refresh_world: RefreshWorld,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two stale-row errors in a row: one retry, then a log line with the class name and ids."""
    world = refresh_world
    attempts: list[int] = []

    def always_stale(context, job_ids):
        attempts.append(len(attempts) + 1)
        raise StaleDataError(STALE_MESSAGE)

    monkeypatch.setattr(background, "refresh_top_k_summaries", always_stale)
    llm = HookedLlm()
    with caplog.at_level(logging.INFO):
        refresh_summaries(world.refresh_job((world.backend_id,), llm))

    assert len(attempts) == 2
    assert llm.attempts == 0
    assert FAILURE_LOG_TEXT in caplog.text
    assert "StaleDataError" in caplog.text
    assert str(world.backend_id) in caplog.text
    assert "MUST NOT BE LOGGED" not in caplog.text
