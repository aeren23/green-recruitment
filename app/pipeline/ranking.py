"""The ranked list of a posting and its relative score (SPEC 6.5, 9.1).

Ranking uses the raw score. The score shown to the employer is min-max scaled
over all scored candidates of the posting (not just the visible page) and is
only meaningful relative to them, because e5 similarities cluster in a narrow
band.
"""

from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, defer, selectinload

from app.models import Application, CandidateProfile
from app.pipeline.context import SCORED_STATUSES

RELATIVE_SCORE_MIN = 0.0
RELATIVE_SCORE_MAX = 100.0


@dataclass(frozen=True)
class ScoreBounds:
    """The lowest and highest raw score among a posting's ranked candidates."""

    lowest: float
    highest: float

    def relative(self, score: float) -> float:
        """The score scaled to 0..100; 100 when every candidate scored the same.

        Args:
            score: A raw score within the bounds.

        Returns:
            A value from `RELATIVE_SCORE_MIN` to `RELATIVE_SCORE_MAX`.
        """
        span = self.highest - self.lowest
        if span <= 0:
            return RELATIVE_SCORE_MAX
        share = (score - self.lowest) / span
        return RELATIVE_SCORE_MIN + share * (RELATIVE_SCORE_MAX - RELATIVE_SCORE_MIN)


def ranked_statement(job_id: int) -> Select[tuple[Application]]:
    """Ranked and shortlisted applications: raw score descending, then id."""
    return (
        select(Application)
        .where(Application.job_id == job_id, Application.status.in_(SCORED_STATUSES))
        .options(selectinload(Application.profile).options(defer(CandidateProfile.profile_json)))
        .order_by(Application.score.desc(), Application.id)
    )


def score_bounds(session: Session, job_id: int) -> ScoreBounds | None:
    """Score range over all ranked candidates of a posting; `None` without any."""
    statement = select(func.min(Application.score), func.max(Application.score)).where(
        Application.job_id == job_id,
        Application.status.in_(SCORED_STATUSES),
        Application.score.is_not(None),
    )
    lowest, highest = session.execute(statement).one()
    if lowest is None or highest is None:
        return None
    return ScoreBounds(lowest=lowest, highest=highest)
