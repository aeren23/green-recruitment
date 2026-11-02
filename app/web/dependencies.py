"""FastAPI dependencies: the per-request session, the clock and the web context."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings
from app.green.tracker import StageTracker
from app.scoring.embed import EmbeddingProvider


@dataclass(frozen=True)
class WebContext:
    """What a route handler needs, bundled so handlers stay short.

    Attributes:
        request: The current request.
        session: The request's database session; the handler commits.
        now: The injected clock (timezone-aware UTC).
        embedder: The embedding model loaded once in the lifespan.
        settings: The application settings.
        tracker: The stage tracker; always the no-op one in the web layer (D3).
    """

    request: Request
    session: Session
    now: datetime
    embedder: EmbeddingProvider
    settings: Settings
    tracker: StageTracker


def get_settings(request: Request) -> Settings:
    """The settings the lifespan resolved at start-up."""
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    """One database session per request, closed afterwards."""
    with request.app.state.session_factory() as session:
        yield session


def get_now() -> datetime:
    """The current UTC time; tests override this dependency."""
    return datetime.now(UTC)


def get_web_context(
    request: Request,
    session: Session = Depends(get_session),
    now: datetime = Depends(get_now),
) -> WebContext:
    """Bundle the request, session, clock, embedder and settings."""
    return WebContext(
        request=request,
        session=session,
        now=now,
        embedder=request.app.state.embedder,
        settings=request.app.state.settings,
        tracker=request.app.state.tracker,
    )
