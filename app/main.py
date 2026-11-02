"""FastAPI application factory and lifespan."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.config import Settings
from app.db import create_db_engine, create_demo_schema, create_session_factory
from app.scoring.embed import EmbeddingProvider, create_embedder

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


def create_app(
    settings: Settings | None = None, embedder: EmbeddingProvider | None = None
) -> FastAPI:
    """Build the FastAPI application.

    Settings are resolved inside the lifespan, so importing this module never
    requires a `.env` file, and the embedding model is loaded there too.

    Args:
        settings: Explicit settings, mainly for tests. When omitted they are
            loaded from the environment and `.env` at start-up.
        embedder: Embedding provider to use as is, mainly for tests. When omitted
            the real model is loaded once during start-up.

    Returns:
        The configured application.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings if settings is not None else Settings()
        application.state.embedder = resolve_embedder(resolved_settings, embedder)
        engine = create_db_engine(resolved_settings.database_url)
        application.state.settings = resolved_settings
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        try:
            create_demo_schema(engine)
            yield
        finally:
            engine.dispose()

    application = FastAPI(lifespan=lifespan)
    application.include_router(health_router)
    return application


app = create_app()
