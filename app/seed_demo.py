"""`python -m app.seed`: load the three demo postings into the demo database.

The database is the one named by `DATABASE_URL`. The embedding model is loaded
once and only when a posting is actually missing, so a second run is idempotent
and costs no model load and no embedding.
"""

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.config import Settings
from app.db import create_db_engine, create_demo_schema, create_session_factory
from app.scoring.embed import create_embedder
from app.seed import (
    DEMO_FILTERS_BY_TITLE,
    SEED_POSTINGS,
    SeedOptions,
    missing_seed_titles,
    seed_postings,
)

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def seed_demo_postings(session: Session, settings: Settings, now: datetime) -> int:
    """Seed the missing demo postings, loading the model only if needed.

    Args:
        session: Session on the demo database; committed on success.
        settings: Settings naming the embedding model and device.
        now: The injected clock, stored as `created_at`.

    Returns:
        The number of postings created by this call.
    """
    missing = missing_seed_titles(session)
    if not missing:
        return 0
    embedder = create_embedder(settings)
    embedder.load()
    options = SeedOptions(now=now, filters_by_title=DEMO_FILTERS_BY_TITLE)
    seed_postings(session, embedder, options)
    session.commit()
    return len(missing)


def main() -> None:
    """Command-line entry point: seed the database named by `DATABASE_URL`."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    settings = Settings()
    engine = create_db_engine(settings.database_url)
    try:
        create_demo_schema(engine)
        with create_session_factory(engine)() as session:
            created = seed_demo_postings(session, settings, datetime.now(UTC))
    finally:
        engine.dispose()
    print(f"Demo postings created: {created}, already present: {len(SEED_POSTINGS) - created}")
