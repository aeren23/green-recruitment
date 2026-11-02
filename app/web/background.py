"""The post-confirmation job: top-k summaries written after the response (T13-1).

A candidate must not wait for the local LLM. The confirmation commits without
it; this job then opens its own session and writes the missing summaries of
every posting the confirmation touched. An unreachable Ollama leaves the
summaries empty and is not an error.

KVKK: the job logs ids and counts only, and never the text of an exception.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import ObjectDeletedError, StaleDataError
from starlette.background import BackgroundTask

from app.config import Settings
from app.green.tracker import StageTracker
from app.llm.ollama import LlmClient
from app.pipeline import PipelineContext
from app.pipeline.summaries import refresh_top_k_summaries
from app.scoring.embed import EmbeddingProvider
from app.web.dependencies import WebContext

logger = logging.getLogger(__name__)

MAX_REFRESH_ATTEMPTS = 2
RETRYABLE_REFRESH_ERRORS = (StaleDataError, ObjectDeletedError)


@dataclass(frozen=True)
class SummaryRefreshJob:
    """Everything the background job needs, captured while the request is alive."""

    session_factory: sessionmaker[Session]
    settings: Settings
    embedder: EmbeddingProvider
    tracker: StageTracker
    llm_client: LlmClient
    now: datetime
    job_ids: Sequence[int]


def refresh_summaries(job: SummaryRefreshJob) -> None:
    """Write the missing top-k summaries of the job's postings, in a new session.

    A concurrent delete or re-score can make a row stale mid-refresh; the whole
    refresh is then retried once. Summaries already committed are skipped by the retry.
    """
    with job.session_factory() as session:
        context = build_refresh_context(session, job)
        written = run_refresh_attempts(session, context, job)
    if written is not None:
        logger.info("Summary refresh for postings %s wrote %d summaries", list(job.job_ids), written)


def build_refresh_context(session: Session, job: SummaryRefreshJob) -> PipelineContext:
    """The pipeline context of one refresh, bound to the job's own session."""
    return PipelineContext(
        session=session,
        settings=job.settings,
        embedder=job.embedder,
        tracker=job.tracker,
        now=job.now,
        reference_date=job.now.date(),
        llm_client=job.llm_client,
    )


def run_refresh_attempts(
    session: Session, context: PipelineContext, job: SummaryRefreshJob
) -> int | None:
    """Run the refresh, retrying once after a stale-row conflict.

    Returns:
        The number of summaries written, or `None` when the refresh gave up.
    """
    for attempt in range(1, MAX_REFRESH_ATTEMPTS + 1):
        try:
            return refresh_top_k_summaries(context, job.job_ids)
        except RETRYABLE_REFRESH_ERRORS as error:
            session.rollback()
            if attempt < MAX_REFRESH_ATTEMPTS:
                logger.warning(
                    "Summary refresh for postings %s retrying after %s",
                    list(job.job_ids),
                    type(error).__name__,
                )
                continue
            log_refresh_failure(job, error)
            return None
        except Exception as error:  # noqa: BLE001 - after the response nobody can be told
            session.rollback()
            log_refresh_failure(job, error)
            return None
    return None


def log_refresh_failure(job: SummaryRefreshJob, error: Exception) -> None:
    """Log the failed refresh by posting ids and exception class only (KVKK)."""
    logger.error(
        "Summary refresh failed for postings %s: %s", list(job.job_ids), type(error).__name__
    )


def schedule_summary_refresh(
    context: WebContext, job_ids: Sequence[int]
) -> BackgroundTask | None:
    """A background task for the postings, or `None` when there is nothing to do."""
    if not job_ids:
        return None
    state = context.request.app.state
    job = SummaryRefreshJob(
        session_factory=state.session_factory,
        settings=context.settings,
        embedder=context.embedder,
        tracker=context.tracker,
        llm_client=state.llm_client,
        now=context.now,
        job_ids=tuple(job_ids),
    )
    return BackgroundTask(refresh_summaries, job)
