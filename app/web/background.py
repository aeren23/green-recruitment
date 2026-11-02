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
from starlette.background import BackgroundTask

from app.config import Settings
from app.green.tracker import StageTracker
from app.llm.ollama import LlmClient
from app.pipeline import PipelineContext, fill_top_k_summaries
from app.scoring.embed import EmbeddingProvider
from app.web.dependencies import WebContext

logger = logging.getLogger(__name__)


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
    """Write the missing top-k summaries of the job's postings, in a new session."""
    with job.session_factory() as session:
        context = PipelineContext(
            session=session,
            settings=job.settings,
            embedder=job.embedder,
            tracker=job.tracker,
            now=job.now,
            reference_date=job.now.date(),
            llm_client=job.llm_client,
        )
        try:
            written = fill_top_k_summaries(context, job.job_ids)
            session.commit()
        except Exception as error:  # noqa: BLE001 - after the response nobody can be told
            session.rollback()
            logger.error("Summary refresh failed: %s", type(error).__name__)
            return
    logger.info("Summary refresh for postings %s wrote %d summaries", list(job.job_ids), written)


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
