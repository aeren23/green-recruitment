"""Top-k summaries: the only place the green pipeline calls the LLM (SPEC 6.6).

After a confirmation, the top-k of every affected posting is checked: score
descending, id ascending. Each top-k application without a summary gets one,
written from the matrix-selected sections only.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.green.counters import CounterName, increment_counter
from app.llm.prompts import CvSection, SummaryPromptInput
from app.llm.tasks import summarize_candidate
from app.models import Application
from app.pipeline.context import SCORED_STATUSES, STAGE_LLM_SUMMARY, PipelineContext
from app.pipeline.scoring import MATRIX_SECTION_ID
from app.pipeline.shared import load_profile_sections, load_requirements


def top_k_applications(context: PipelineContext, job_id: int) -> list[Application]:
    """The top-k of a posting: score descending, id ascending."""
    statement = (
        select(Application)
        .where(Application.job_id == job_id, Application.status.in_(SCORED_STATUSES))
        .order_by(Application.score.desc(), Application.id)
        .limit(context.settings.top_k)
    )
    return list(context.session.scalars(statement))


def fill_top_k_summaries(context: PipelineContext, job_ids: Sequence[int]) -> int:
    """Summarise every top-k application that has no summary yet.

    Stops after the first failed summary, so an unreachable LLM costs one
    attempt per confirmation, not one per candidate. Updates the `llm_calls`
    counter from the usage accumulator.

    Returns:
        The number of summaries written.
    """
    if context.llm_client is None:
        return 0
    calls_before = context.llm_usage.calls
    pending = [
        application
        for job_id in job_ids
        for application in top_k_applications(context, job_id)
        if application.summary is None
    ]
    written = summarize_until_failure(context, pending)
    new_calls = context.llm_usage.calls - calls_before
    if new_calls:
        increment_counter(context.session, CounterName.LLM_CALLS, new_calls)
    return written


def summarize_until_failure(
    context: PipelineContext, applications: Sequence[Application]
) -> int:
    """Write summaries one by one; leave `None` where the LLM gave nothing."""
    written = 0
    for application in applications:
        failures_before = context.llm_usage.failures
        summary = summarize_in_stage(context, application)
        if summary is not None:
            application.summary = summary
            written += 1
        if context.llm_usage.failures > failures_before:
            break
    return written


def summarize_in_stage(context: PipelineContext, application: Application) -> str | None:
    """Stage `llm_summary`: the matrix-selected sections of one candidate only."""
    prompt_input = summary_prompt_input(context, application)
    if prompt_input is None or context.llm_client is None:
        return None
    with context.tracker.stage(STAGE_LLM_SUMMARY):
        return summarize_candidate(context.llm_client, prompt_input, context.llm_usage)


def summary_prompt_input(
    context: PipelineContext, application: Application
) -> SummaryPromptInput | None:
    """Prompt input with the matrix-selected section ids; `None` if none exists."""
    sections = load_profile_sections(context.session, application.profile_id)
    selected = {entry[MATRIX_SECTION_ID] for entry in application.matrix_json or []}
    if not selected & {section.id for section in sections}:
        return None
    requirements = load_requirements(context.session, application.job_id)
    return SummaryPromptInput(
        requirements=[requirement.text for requirement in requirements],
        sections=[CvSection(item.id, item.type.value, item.text) for item in sections],
        selected_section_ids=selected,
    )


def refresh_top_k_summaries(context: PipelineContext, job_ids: Sequence[int]) -> int:
    """The background variant of `fill_top_k_summaries`: one application per commit.

    Each unit commits on its own, so a conflict with a concurrent request costs
    at most the unit in progress, and a retry skips the summaries already
    written. Stops after the first failed summary, as the confirmation does.

    Returns:
        The number of summaries written.
    """
    if context.llm_client is None:
        return 0
    written = 0
    for application_id in pending_top_k_ids(context, job_ids):
        failures_before = context.llm_usage.failures
        written += summarize_unit(context, application_id)
        if context.llm_usage.failures > failures_before:
            break
    return written


def pending_top_k_ids(context: PipelineContext, job_ids: Sequence[int]) -> list[int]:
    """Ids of the top-k applications without a summary, posting by posting."""
    return [
        application.id
        for job_id in job_ids
        for application in top_k_applications(context, job_id)
        if application.summary is None
    ]


def summarize_unit(context: PipelineContext, application_id: int) -> int:
    """One unit of the background refresh: summarise, count the calls, commit.

    Returns:
        1 when a summary was written, else 0.
    """
    calls_before = context.llm_usage.calls
    written = summarize_stored_application(context, application_id)
    new_calls = context.llm_usage.calls - calls_before
    if new_calls:
        increment_counter(context.session, CounterName.LLM_CALLS, new_calls)
    context.session.commit()
    return int(written)


def summarize_stored_application(context: PipelineContext, application_id: int) -> bool:
    """Write one summary, re-loading the application before the LLM call and before the write.

    The LLM call can take seconds, and a concurrent request may delete the
    application or re-score it meanwhile. Only an application that is still
    top-k and without a summary is written; a vanished or changed row is skipped.

    Returns:
        Whether a summary was written.
    """
    application = load_application_now(context.session, application_id)
    if application is None or not is_summary_due(context, application):
        return False
    summary = summarize_in_stage(context, application)
    if summary is None:
        return False
    current = load_application_now(context.session, application_id)
    if current is None or not is_summary_due(context, current):
        return False
    current.summary = summary
    return True


def load_application_now(session: Session, application_id: int) -> Application | None:
    """The application as the database holds it now, or `None` once it is gone.

    `populate_existing` overwrites what this session loaded earlier, so a row
    deleted or re-scored by another request is seen in its current state.
    """
    statement = (
        select(Application)
        .where(Application.id == application_id)
        .execution_options(populate_existing=True)
    )
    return session.scalar(statement)


def is_summary_due(context: PipelineContext, application: Application) -> bool:
    """True while the application is top-k (ranked or shortlisted) and has no summary."""
    if application.summary is not None:
        return False
    top_k_ids = {entry.id for entry in top_k_applications(context, application.job_id)}
    return application.id in top_k_ids
