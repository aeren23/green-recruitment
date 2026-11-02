"""Top-k summaries: the only place the green pipeline calls the LLM (SPEC 6.6).

After a confirmation, the top-k of every affected posting is checked: score
descending, id ascending. Each top-k application without a summary gets one,
written from the matrix-selected sections only.
"""

from collections.abc import Sequence

from sqlalchemy import select

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
