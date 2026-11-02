"""Stage names, errors and the DTOs the pipeline functions take and return."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy.orm import Session

from app.config import Settings
from app.green.tracker import StageTracker
from app.llm.ollama import LlmClient
from app.llm.usage import LlmUsage
from app.models import ApplicationStatus, SectionType
from app.privacy.reasons import StoredReasonCode
from app.scoring.embed import EmbeddingProvider

STAGE_EXTRACT = "extract"
STAGE_HASH = "hash"
STAGE_PARSE = "parse"
STAGE_FILTER = "filter"
STAGE_EMBED = "embed"
STAGE_SCORE = "score"
STAGE_LLM_SUMMARY = "llm_summary"
STAGE_RETENTION = "retention"

# A shortlisted application still sits in the ranking, so it counts for top-k
# and is re-scored together with the ranked ones (SC-03).
SCORED_STATUSES = (ApplicationStatus.RANKED, ApplicationStatus.SHORTLISTED)


class PostingClosedError(RuntimeError):
    """The posting is closed or finalized and accepts no uploads or confirmations."""


class DraftNotFoundError(LookupError):
    """No application carries the given draft token."""


class PostingWithoutRequirementsError(ValueError):
    """The posting has no requirements, so nothing can be scored against it."""


class UploadOutcomeKind(StrEnum):
    """What `submit_upload` did with an upload."""

    DRAFT_CREATED = "draft_created"
    DRAFT_EXISTS = "draft_exists"
    DUPLICATE_APPLICATION = "duplicate_application"


class ConfirmOutcomeKind(StrEnum):
    """What `confirm_draft` did with a draft."""

    RANKED = "ranked"
    ELIMINATED = "eliminated"


@dataclass(frozen=True)
class PipelineContext:
    """Everything the pipeline functions depend on, injected by the caller.

    Attributes:
        session: Session whose transaction the caller owns; nothing is committed.
        settings: Application settings (`HASH_SECRET`, `TOP_K`, ...).
        embedder: The loaded embedding provider.
        tracker: Stage tracker; the web layer passes the no-op one.
        now: The injected clock (timezone-aware UTC).
        reference_date: Date that `Current` ends in a CV are clipped to; the
            web layer passes today, the benchmark `BENCH_CV_REFERENCE_DATE`.
        llm_client: Client for the top-k summaries; `None` switches summaries off.
        llm_usage: Accumulates the LLM calls and tokens of the whole run.
    """

    session: Session
    settings: Settings
    embedder: EmbeddingProvider
    tracker: StageTracker
    now: datetime
    reference_date: date
    llm_client: LlmClient | None = None
    llm_usage: LlmUsage = field(default_factory=LlmUsage)


@dataclass
class UploadInput:
    """One uploaded PDF. The pipeline empties `pdf_bytes` after extraction.

    The CV bytes are hidden from `repr` (KVKK). Callers must not keep their
    own reference to the bytes after the call.
    """

    job_id: int
    pdf_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class UploadOutcome:
    """Result of `submit_upload`.

    Attributes:
        kind: A new draft, a redirect to a live draft, or a duplicate.
        draft_token: Token of the new or existing draft; `None` for a duplicate.
        cache_hit: The profile came from the parse cache and parsing was skipped.
        no_headings: The CV had no heading and became one summary section (the
            candidate sees a warning). Only known after a fresh parse, so it is
            `False` on a cache hit.
    """

    kind: UploadOutcomeKind
    draft_token: str | None
    cache_hit: bool = False
    no_headings: bool = False


@dataclass(frozen=True)
class SectionEdit:
    """One section as the candidate left it in the preview form."""

    type: SectionType
    text: str = field(repr=False)


@dataclass(frozen=True)
class ContactEdit:
    """Contact details as the candidate left them; they replace the stored ones."""

    name: str | None = field(default=None, repr=False)
    email: str | None = field(default=None, repr=False)
    phone: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ProfileEdits:
    """The candidate's corrections to the parsed profile (T-13 preview form).

    Attributes:
        sections: The complete list of sections after the edit, in order.
        years_experience: Years of experience, `None` when unknown.
        languages: Spoken languages; Turkish names are mapped to English.
        location: Free-text location, `None` when unknown.
        contact: New contact details; `None` keeps the stored ones.
    """

    sections: Sequence[SectionEdit]
    years_experience: float | None
    languages: Sequence[str]
    location: str | None
    contact: ContactEdit | None = None


@dataclass(frozen=True)
class ConfirmInput:
    """Input of `confirm_draft`; no edits means auto-confirm (the benchmark)."""

    draft_token: str
    edits: ProfileEdits | None = None


@dataclass(frozen=True)
class ConfirmOutcome:
    """Result of `confirm_draft`.

    Attributes:
        kind: Ranked, or eliminated by a hard filter.
        draft_token: Token of the application; after an elimination the status
            page finds the `DecisionRecord` through it.
        application_id: Id of the ranked application; `None` when eliminated.
        score: The raw score; `None` when eliminated.
        reason_code: The elimination reason; `None` when ranked.
        rescored_applications: Other applications (ranked or shortlisted) of
            the same profile that were re-scored because its sections changed
            (SC-03).
        summaries_generated: Top-k summaries written by this call.
        affected_job_ids: Postings whose ranking this call touched; a caller
            that confirmed without an LLM refreshes their top-k summaries later.
    """

    kind: ConfirmOutcomeKind
    draft_token: str
    application_id: int | None
    score: float | None
    reason_code: StoredReasonCode | None
    rescored_applications: int = 0
    summaries_generated: int = 0
    affected_job_ids: tuple[int, ...] = ()
