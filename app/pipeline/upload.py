"""`submit_upload`: PDF bytes in, a draft application out (SPEC 6.1, first half).

Order: extract, hash (with the duplicate check and the parse-cache lookup),
parse on a cache miss, draft. The lookups run inside the `hash` stage, so the
green pipeline has no unmeasured lookup work. A stale draft is expired after
the `hash` stage in its own `retention` stage, because stages never nest.
"""

import logging
import secrets
from dataclasses import dataclass

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.green.counters import CounterName, increment_counter
from app.ingest.contact import extract_contact, remove_contact
from app.ingest.normalize import compute_raw_hash, normalize, short_hash
from app.ingest.pdf import extract_text
from app.ingest.sections import PARSER_VERSION, ParsedSection, parse_cv
from app.models import (
    Application,
    ApplicationStatus,
    CandidateContact,
    CandidateProfile,
    DecisionRecord,
    Section,
)
from app.pipeline.context import (
    STAGE_EXTRACT,
    STAGE_HASH,
    STAGE_PARSE,
    STAGE_RETENTION,
    PipelineContext,
    UploadInput,
    UploadOutcome,
    UploadOutcomeKind,
)
from app.pipeline.shared import require_open_posting, retention_context
from app.privacy.retention import expire_draft, is_draft_expired

logger = logging.getLogger(__name__)

DRAFT_TOKEN_BYTES = 32
EMPTY_PDF = b""


@dataclass(frozen=True)
class HashLookup:
    """What the `hash` stage found out about an upload.

    Attributes:
        raw_hash: The keyed hash of the normalized text.
        blocked: A duplicate or redirect outcome that ends the upload, if any.
        expired_draft: A stale draft of this CV to remove before a new one.
        cached_profile: The cached profile of this CV, if any.
    """

    raw_hash: str
    blocked: UploadOutcome | None = None
    expired_draft: Application | None = None
    cached_profile: CandidateProfile | None = None


@dataclass(frozen=True)
class ProfileLookup:
    """The profile an upload resolved to, and how it was obtained."""

    profile: CandidateProfile
    cache_hit: bool
    no_headings: bool


def submit_upload(context: PipelineContext, upload: UploadInput) -> UploadOutcome:
    """Turn an uploaded PDF into a draft application (SPEC 6.1, first half).

    Order: extract, hash and lookups, profile cache, draft. A draft that is
    older than `DRAFT_TTL_HOURS` is expired and replaced by a new one. The
    session is flushed but not committed.

    Args:
        context: Injected dependencies.
        upload: The posting id and the PDF bytes; the bytes are emptied.

    Returns:
        A new draft, a redirect to the live draft, or `DUPLICATE_APPLICATION`.

    Raises:
        PostingNotFoundError: If the posting does not exist.
        PostingClosedError: If the posting is closed or finalized.
        UnreadablePdfError: If the PDF is corrupt, empty or encrypted.
        TextTooShortError: If the PDF yields too little text.
    """
    posting = require_open_posting(context.session, upload.job_id)
    text, pdf_size = extract_text_in_stage(context, upload)
    found = hash_and_look_up_in_stage(context, text, posting.id)
    if found.blocked is not None:
        log_upload(posting.id, found.raw_hash, found.blocked)
        return found.blocked
    cached = release_expired_draft(context, found)
    lookup = obtain_profile(context, text, found.raw_hash, cached)
    draft = create_draft(context, posting.id, lookup.profile)
    increment_counter(context.session, CounterName.PDF_BYTES_NOT_STORED, pdf_size)
    outcome = UploadOutcome(
        UploadOutcomeKind.DRAFT_CREATED,
        draft.draft_token,
        cache_hit=lookup.cache_hit,
        no_headings=lookup.no_headings,
    )
    log_upload(posting.id, found.raw_hash, outcome)
    return outcome


def extract_text_in_stage(context: PipelineContext, upload: UploadInput) -> tuple[str, int]:
    """Stage `extract`: read the text in memory, then drop the bytes reference."""
    pdf_size = len(upload.pdf_bytes)
    try:
        with context.tracker.stage(STAGE_EXTRACT):
            text = extract_text(upload.pdf_bytes)
    finally:
        upload.pdf_bytes = EMPTY_PDF
    return text, pdf_size


def hash_and_look_up_in_stage(
    context: PipelineContext, text: str, job_id: int
) -> HashLookup:
    """Stage `hash`: the keyed hash (SPEC 6.2), the duplicate check, the cache lookup."""
    with context.tracker.stage(STAGE_HASH):
        raw_hash = compute_raw_hash(normalize(text), context.settings.hash_secret)
        return look_up_existing(context, job_id, raw_hash)


def look_up_existing(context: PipelineContext, job_id: int, raw_hash: str) -> HashLookup:
    """The duplicate check (SPEC 6.1) followed by the parse-cache lookup.

    A non-draft application or an unexpired decision blocks the upload, a live
    draft redirects to itself (SC-07), a stale draft is reported for removal.
    """
    existing = find_application(context.session, job_id, raw_hash)
    expired: Application | None = None
    if existing is not None:
        if existing.status is not ApplicationStatus.DRAFT:
            return HashLookup(raw_hash, blocked=duplicate_outcome())
        if not is_draft_expired(existing, context.now, context.settings.draft_ttl_hours):
            return HashLookup(raw_hash, blocked=draft_exists_outcome(existing))
        expired = existing
    if has_active_decision(context, job_id, raw_hash):
        return HashLookup(raw_hash, blocked=duplicate_outcome())
    cached = find_cached_profile(context.session, raw_hash)
    return HashLookup(raw_hash, expired_draft=expired, cached_profile=cached)


def duplicate_outcome() -> UploadOutcome:
    """The `DUPLICATE_APPLICATION` outcome (a response code, never stored)."""
    return UploadOutcome(UploadOutcomeKind.DUPLICATE_APPLICATION, None)


def draft_exists_outcome(draft: Application) -> UploadOutcome:
    """The redirect to a draft that is still alive."""
    return UploadOutcome(UploadOutcomeKind.DRAFT_EXISTS, draft.draft_token)


def find_application(session: Session, job_id: int, raw_hash: str) -> Application | None:
    """The application of this CV to this posting, if any."""
    return session.scalar(
        select(Application).where(
            Application.job_id == job_id, Application.raw_hash == raw_hash
        )
    )


def has_active_decision(context: PipelineContext, job_id: int, raw_hash: str) -> bool:
    """Whether an unexpired `DecisionRecord` blocks this CV for this posting."""
    statement = select(DecisionRecord.id).where(
        DecisionRecord.job_id == job_id,
        DecisionRecord.raw_hash == raw_hash,
        DecisionRecord.expires_at >= context.now,
    )
    return context.session.scalar(statement.limit(1)) is not None


def find_cached_profile(session: Session, raw_hash: str) -> CandidateProfile | None:
    """The cached profile of this CV (the parse cache, keyed by `raw_hash`)."""
    return session.scalar(
        select(CandidateProfile).where(CandidateProfile.raw_hash == raw_hash)
    )


def release_expired_draft(
    context: PipelineContext, found: HashLookup
) -> CandidateProfile | None:
    """Stage `retention` (rule R1) for a stale draft; return the usable cached profile.

    The expiry may delete the very profile that was found in the cache (when
    the stale draft was its last application), so a profile that is gone after
    it counts as a cache miss.
    """
    if found.expired_draft is None:
        return found.cached_profile
    with context.tracker.stage(STAGE_RETENTION):
        expire_draft(retention_context(context), found.expired_draft)
    profile = found.cached_profile
    if profile is None or not inspect(profile).persistent:
        return None
    return profile


def obtain_profile(
    context: PipelineContext,
    text: str,
    raw_hash: str,
    cached: CandidateProfile | None,
) -> ProfileLookup:
    """Reuse the cached profile of this CV, or parse it into a new one."""
    if cached is not None:
        cached.last_used_at = context.now
        increment_counter(context.session, CounterName.PARSE_CACHE_HITS)
        return ProfileLookup(cached, cache_hit=True, no_headings=False)
    with context.tracker.stage(STAGE_PARSE):
        return parse_into_new_profile(context, text, raw_hash)


def parse_into_new_profile(
    context: PipelineContext, text: str, raw_hash: str
) -> ProfileLookup:
    """Stage `parse` body: split contact and sections, store a new profile."""
    contact = extract_contact(text)
    parsed = parse_cv(remove_contact(text, contact), context.reference_date)
    profile = CandidateProfile(
        raw_hash=raw_hash,
        profile_json=parsed.profile_json,
        years_experience=parsed.years_experience,
        languages=parsed.languages,
        location=None,
        parser_version=PARSER_VERSION,
        created_at=context.now,
        confirmed_at=None,
        last_used_at=context.now,
    )
    profile.sections = [section_row(section) for section in parsed.sections]
    profile.contact = CandidateContact(
        name=contact.name, email=contact.email, phone=contact.phone
    )
    context.session.add(profile)
    context.session.flush()
    return ProfileLookup(profile, cache_hit=False, no_headings=parsed.no_headings_found)


def section_row(section: ParsedSection) -> Section:
    """A section row without hash or embedding; those come at confirmation."""
    return Section(type=section.type, ordinal=section.ordinal, text=section.text)


def create_draft(
    context: PipelineContext, job_id: int, profile: CandidateProfile
) -> Application:
    """Create the draft application with an unguessable token."""
    draft = Application(
        job_id=job_id,
        profile_id=profile.id,
        raw_hash=profile.raw_hash,
        status=ApplicationStatus.DRAFT,
        draft_token=secrets.token_urlsafe(DRAFT_TOKEN_BYTES),
        created_at=context.now,
    )
    context.session.add(draft)
    context.session.flush()
    return draft


def log_upload(job_id: int, raw_hash: str, outcome: UploadOutcome) -> None:
    """Log an upload with ids and the short hash only."""
    logger.info(
        "Upload to posting %d: outcome=%s hash=%s cache_hit=%s",
        job_id,
        outcome.kind.value,
        short_hash(raw_hash),
        outcome.cache_hit,
    )
