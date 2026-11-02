"""FastAPI application factory and lifespan."""

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager

import httpx
from fastapi import APIRouter, FastAPI

from app.config import Settings
from app.db import create_db_engine, create_demo_schema, create_session_factory
from app.green.tracker import NoOpStageTracker
from app.llm.ollama import LlmClient, OllamaClient, OllamaConfig
from app.scoring.embed import EmbeddingProvider, create_embedder
from app.web.access_log import install_access_log_masking
from app.web.setup import install_web

HEALTH_PATH = "/health"
HEALTH_RESPONSE = {"status": "ok"}

health_router = APIRouter()


@health_router.get(HEALTH_PATH)
def health() -> dict[str, str]:
    """Report that the service is up."""
    return HEALTH_RESPONSE


def resolve_embedder(
    settings: Settings, injected: EmbeddingProvider | None
) -> EmbeddingProvider:
    """Return the embedder for this process, loading the real model exactly once.

    An injected embedder is used as is (it is the caller's job to have it ready).
    Otherwise the model named in the settings is built and loaded here, at
    start-up, never inside a request.
    """
    if injected is not None:
        return injected
    embedder = create_embedder(settings)
    embedder.load()
    return embedder


def resolve_llm_client(
    settings: Settings, injected: LlmClient | None, resources: ExitStack
) -> LlmClient:
    """Return the client for the top-k summaries.

    An injected client is used as is. Otherwise an Ollama client is built; its
    HTTP client is closed with `resources` at shutdown. Nothing connects here:
    Ollama is only contacted when a summary is written.
    """
    if injected is not None:
        return injected
    http_client = resources.enter_context(httpx.Client())
    return OllamaClient(http_client, OllamaConfig(settings.ollama_url, settings.llm_model))


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    llm_client: LlmClient | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Settings are resolved inside the lifespan, so importing this module never
    requires a `.env` file, and the embedding model is loaded there too.

    Args:
        settings: Explicit settings, mainly for tests. When omitted they are
            loaded from the environment and `.env` at start-up.
        embedder: Embedding provider to use as is, mainly for tests. When omitted
            the real model is loaded once during start-up.
        llm_client: LLM client to use as is, mainly for tests. When omitted an
            Ollama client for the configured server is built.

    Returns:
        The configured application.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings if settings is not None else Settings()
        application.state.embedder = resolve_embedder(resolved_settings, embedder)
        install_access_log_masking()
        with ExitStack() as resources:
            application.state.llm_client = resolve_llm_client(
                resolved_settings, llm_client, resources
            )
            engine = create_db_engine(resolved_settings.database_url)
            resources.callback(engine.dispose)
            application.state.settings = resolved_settings
            # Decision D3: the web layer never measures, whatever GREEN_TRACKING says.
            application.state.tracker = NoOpStageTracker()
            application.state.engine = engine
            application.state.session_factory = create_session_factory(engine)
            create_demo_schema(engine)
            yield

    application = FastAPI(lifespan=lifespan)
    application.include_router(health_router)
    install_web(application)
    return application


app = create_app()
