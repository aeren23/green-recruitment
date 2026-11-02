"""`confirm_draft`: apply edits, filter, embed, score, summarise (SPEC 6.1, second half)."""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Application, ApplicationStatus
from app.pipeline.context import (
    SCORED_STATUSES,
    STAGE_FILTER,
    STAGE_RETENTION,
    ConfirmInput,
    ConfirmOutcome,
    ConfirmOutcomeKind,
    DraftNotFoundError,
    PipelineContext,
)
from app.pipeline.profile_edits import apply_edits
from app.pipeline.scoring import embed_and_score, invalidate_summaries
from app.pipeline.shared import require_open_posting, retention_context
from app.pipeline.summaries import fill_top_k_summaries
from app.privacy.reasons import StoredReasonCode
from app.privacy.retention import ApplicationNotDraftError, eliminate_application
from app.scoring.filters import evaluate_hard_filters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfirmCounts:
    """What a confirmation did besides ranking its own application."""

    rescored: int
    summaries: int


@dataclass(frozen=True)
class ApplicationIdentity:
    """Ids read before an elimination deletes the application row."""

    application_id: int
    draft_token: str
    profile_id: int

    @classmethod
    def of(cls, application: Application) -> "ApplicationIdentity":
        """Capture the ids of an application."""
        return cls(application.id, application.draft_token, application.profile_id)


def confirm_draft(context: PipelineContext, confirm_input: ConfirmInput) -> ConfirmOutcome:
    """Confirm a draft: apply edits, filter, embed, score, summarise (SPEC 6.1).

    The hard filter runs on the current posting only. A candidate who fails it
    is eliminated through retention (R2) before anything is embedded. When the
    edits changed the profile's sections, the profile's other `ranked` and
    `shortlisted` applications are re-scored against the new sections and
    lose their summaries (SC-03); their status is unchanged. Finally the
    top-k of every affected posting is checked and each top-k application
    without a summary gets one. The session is flushed but not committed.

    Args:
        context: Injected dependencies.
        confirm_input: The draft token and the optional corrections.

    Returns:
        Whether the application was ranked or eliminated, with the details.

    Raises:
        DraftNotFoundError: If no application carries the token.
        ApplicationNotDraftError: If the application is already confirmed.
        PostingClosedError: If the posting is closed or finalized.
    """
    application = load_draft(context.session, confirm_input.draft_token)
    require_open_posting(context.session, application.job_id)
    identity = ApplicationIdentity.of(application)
    sections_changed = apply_edits(context, application.profile, confirm_input.edits)
    siblings = scored_siblings(context.session, application) if sections_changed else []
    reason = filter_in_stage(context, application)
    if reason is not None:
        eliminate_in_stage(context, application, reason)
    targets = siblings if reason is not None else [application, *siblings]
    embed_and_score(context, identity.profile_id, targets)
    invalidate_summaries(siblings)
    if reason is None:
        application.status = ApplicationStatus.RANKED
        application.confirmed_at = context.now
    job_ids = sorted({target.job_id for target in targets})
    summaries = fill_top_k_summaries(context, job_ids)
    counts = ConfirmCounts(rescored=len(siblings), summaries=summaries)
    return build_confirm_outcome(identity, application, reason, counts)


def build_confirm_outcome(
    identity: ApplicationIdentity,
    application: Application,
    reason: StoredReasonCode | None,
    counts: ConfirmCounts,
) -> ConfirmOutcome:
    """Assemble the outcome DTO and log the result (ids only)."""
    eliminated = reason is not None
    logger.info(
        "Application %d confirmed: %s rescored=%d summaries=%d",
        identity.application_id,
        reason.value if eliminated else ConfirmOutcomeKind.RANKED.value,
        counts.rescored,
        counts.summaries,
    )
    return ConfirmOutcome(
        kind=ConfirmOutcomeKind.ELIMINATED if eliminated else ConfirmOutcomeKind.RANKED,
        draft_token=identity.draft_token,
        application_id=None if eliminated else identity.application_id,
        score=None if eliminated else application.score,
        reason_code=reason,
        rescored_applications=counts.rescored,
        summaries_generated=counts.summaries,
    )


def load_draft(session: Session, draft_token: str) -> Application:
    """The draft application behind a token; the token is never put in a message."""
    application = session.scalar(
        select(Application).where(Application.draft_token == draft_token)
    )
    if application is None:
        raise DraftNotFoundError("No application matches the given draft token.")
    if application.status is not ApplicationStatus.DRAFT:
        raise ApplicationNotDraftError(
            f"Application {application.id} is already confirmed."
        )
    return application


def scored_siblings(session: Session, application: Application) -> list[Application]:
    """The other `ranked` and `shortlisted` applications sharing this profile."""
    statement = (
        select(Application)
        .where(
            Application.profile_id == application.profile_id,
            Application.id != application.id,
            Application.status.in_(SCORED_STATUSES),
        )
        .order_by(Application.id)
    )
    return list(session.scalars(statement))


def filter_in_stage(
    context: PipelineContext, application: Application
) -> StoredReasonCode | None:
    """Stage `filter`: the hard filters of the current posting only (SPEC 6.4)."""
    context.session.flush()
    with context.tracker.stage(STAGE_FILTER):
        return evaluate_hard_filters(
            context.session, application.job_id, application.profile_id
        )


def eliminate_in_stage(
    context: PipelineContext, application: Application, reason: StoredReasonCode
) -> None:
    """Stage `retention`: rule R2 for a candidate who failed a hard filter."""
    with context.tracker.stage(STAGE_RETENTION):
        eliminate_application(retention_context(context), application, reason)
