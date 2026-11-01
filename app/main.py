"""FastAPI application factory and lifespan."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.config import Settings
from app.db import create_db_engine, create_demo_schema, create_session_factory

HEALTH_PATH = "/health"
HEALTH_RESPONSE = {"status": "ok"}

health_router = APIRouter()


@health_router.get(HEALTH_PATH)
def health() -> dict[str, str]:
    """Report that the service is up."""
    return HEALTH_RESPONSE


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Settings are resolved inside the lifespan, so importing this module never
    requires a `.env` file.

    Args:
        settings: Explicit settings, mainly for tests. When omitted they are
            loaded from the environment and `.env` at start-up.

    Returns:
        The configured application.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings if settings is not None else Settings()
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
