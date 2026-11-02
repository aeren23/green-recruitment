"""Bridge from the web request to the pipeline's injected context."""

from app.llm.ollama import LlmClient
from app.pipeline import PipelineContext
from app.web.dependencies import WebContext


def pipeline_context_of(
    context: WebContext, llm_client: LlmClient | None = None
) -> PipelineContext:
    """The pipeline context of a request.

    The web layer always runs with the no-op tracker (D3) and without an LLM:
    summaries are written afterwards by a background task, not in the request.

    Args:
        context: The request's web context.
        llm_client: Client for the top-k summaries; `None` switches them off.
    """
    return PipelineContext(
        session=context.session,
        settings=context.settings,
        embedder=context.embedder,
        tracker=context.tracker,
        now=context.now,
        reference_date=context.now.date(),
        llm_client=llm_client,
    )
