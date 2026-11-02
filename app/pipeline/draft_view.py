"""Opening a draft for the candidate's preview page (SPEC 3 step 2, SC-06).

A live draft is read into a plain DTO so the web layer needs no ORM
knowledge. A draft past its time to live is removed through retention (rule
R1) and reported as expired: `DRAFT_EXPIRED` is a response only and writes no
`DecisionRecord`. The session is flushed but not committed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from app.models import Application, CandidateContact, JobPosting, SectionType
from app.pipeline.confirm import load_draft
from app.pipeline.context import STAGE_RETENTION, ContactEdit, PipelineContext
from app.pipeline.shared import load_profile_sections, retention_context
from app.privacy.retention import expire_draft, is_draft_expired


class DraftState(StrEnum):
    """Whether a draft can still be edited and confirmed."""

    LIVE = "live"
    EXPIRED = "expired"


@dataclass(frozen=True)
class PreviewSection:
    """One section as shown in the preview form; the text is hidden from `repr`."""

    type: SectionType
    text: str = field(repr=False)


@dataclass(frozen=True)
class DraftPreview:
    """What the preview page shows about a draft.

    Attributes:
        state: Live, or expired (the draft was just removed).
        job_id: Posting the candidate applied to.
        job_title: Title of that posting.
        expires_at: When the draft stops being usable.
        sections: The parsed sections in reading order (empty when expired).
        years_experience: Merged years of experience, `None` when unknown.
        languages: Spoken languages as stored (lowercase English names).
        location: Stored (folded) location, `None` when unknown.
        contact: The candidate's own contact details.
    """

    state: DraftState
    job_id: int
    job_title: str
    expires_at: datetime
    sections: tuple[PreviewSection, ...] = field(default=(), repr=False)
    years_experience: float | None = field(default=None, repr=False)
    languages: tuple[str, ...] = field(default=(), repr=False)
    location: str | None = field(default=None, repr=False)
    contact: ContactEdit = field(default_factory=ContactEdit, repr=False)


def open_draft(context: PipelineContext, draft_token: str) -> DraftPreview:
    """Read a draft for the preview, or remove it when it has expired.

    Args:
        context: Injected dependencies.
        draft_token: The unguessable token from the URL.

    Returns:
        The preview; `state` is `EXPIRED` when the draft was removed now.

    Raises:
        DraftNotFoundError: If no application carries the token.
        ApplicationNotDraftError: If the application is already confirmed.
    """
    application = load_draft(context.session, draft_token)
    posting = context.session.get(JobPosting, application.job_id)
    ttl_hours = context.settings.draft_ttl_hours
    expires_at = application.created_at + timedelta(hours=ttl_hours)
    if is_draft_expired(application, context.now, ttl_hours):
        return expire_and_describe(context, application, posting, expires_at)
    return live_preview(context, application, posting, expires_at)


def expire_and_describe(
    context: PipelineContext,
    application: Application,
    posting: JobPosting,
    expires_at: datetime,
) -> DraftPreview:
    """Stage `retention` (R1): delete the stale draft and report it as expired."""
    preview = DraftPreview(DraftState.EXPIRED, posting.id, posting.title, expires_at)
    with context.tracker.stage(STAGE_RETENTION):
        expire_draft(retention_context(context), application)
    return preview


def live_preview(
    context: PipelineContext,
    application: Application,
    posting: JobPosting,
    expires_at: datetime,
) -> DraftPreview:
    """The editable content of a draft that is still alive."""
    profile = application.profile
    stored = load_profile_sections(context.session, profile.id)
    return DraftPreview(
        state=DraftState.LIVE,
        job_id=posting.id,
        job_title=posting.title,
        expires_at=expires_at,
        sections=tuple(PreviewSection(item.type, item.text) for item in stored),
        years_experience=profile.years_experience,
        languages=tuple(profile.languages),
        location=profile.location,
        contact=contact_edit_of(profile.contact),
    )


def contact_edit_of(contact: CandidateContact | None) -> ContactEdit:
    """The stored contact as an edit DTO; empty when none is stored."""
    if contact is None:
        return ContactEdit()
    return ContactEdit(contact.name, contact.email, contact.phone)
