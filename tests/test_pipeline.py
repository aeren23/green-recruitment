"""The green pipeline flow of SPEC 6.1: upload, duplicate check, cache, confirm, rank.

No GPU and no Ollama: embeddings come from `ControlledEmbedder`, summaries from
`FakeLlm`, stages from `RecordingTracker`. CVs are real PDFs built in memory.
"""

import builtins
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pdf_fixtures import build_pdf
from pipeline_support import (
    CANDIDATE_EMAIL,
    CANDIDATE_NAME,
    CANDIDATE_PHONE,
    EXPERIENCE_MARKER,
    SUMMARY_REPLY,
    UNSELECTED_MARKER,
    ControlledEmbedder,
    FakeLlm,
    RecordingTracker,
    build_claim_cv_pdf,
    build_cv_pdf,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.green.counters import read_counters
from app.ingest.normalize import short_hash
from app.ingest.pdf import TextTooShortError, UnreadablePdfError
from app.llm.usage import LlmUsage
from app.models import (
    Application,
    ApplicationStatus,
    CandidateContact,
    CandidateProfile,
    DecisionRecord,
    JobPosting,
    Section,
    SectionType,
)
from app.pipeline import (
    ConfirmInput,
    ConfirmOutcomeKind,
    ContactEdit,
    DraftNotFoundError,
    PipelineContext,
    PostingClosedError,
    ProfileEdits,
    SectionEdit,
    UploadInput,
    UploadOutcome,
    UploadOutcomeKind,
    confirm_draft,
    submit_upload,
)
from app.pipeline import upload as upload_module
from app.privacy.reasons import StoredReasonCode
from app.privacy.retention import ApplicationNotDraftError, RetentionContext, close_posting
from app.seed import (
    ACCOUNTANT_TITLE,
    BACKEND_DEVELOPER_TITLE,
    HR_SPECIALIST_TITLE,
    SeedOptions,
    seed_postings,
)

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
REFERENCE_DATE = date(2026, 1, 1)
TOP_K = 2
DRAFT_TTL_HOURS = 24
DECISION_RETENTION_DAYS = 30
SECTIONS_PER_CV = 4
REQUIREMENTS_PER_POSTING = 4
APPROX_TOLERANCE = 1e-4
SKILLS_COEFFICIENT = 0.6
TWO_YEARS = (2022, 2024)
ONE_YEAR = (2023, 2024)
EXTRACT_HASH_PARSE = ["extract", "hash", "parse"]
EXTRACT_HASH = ["extract", "hash"]
FILTER_EMBED_SCORE = ["filter", "embed", "score"]
WRITE_MODE_FLAGS = "wax+"
OTHER_EMAIL = "someone.else@example.org"
OTHER_PHONE = "0532 987 65 43"
NEW_CONTACT_EMAIL = "new.mail@example.org"


@dataclass
class World:
    """Everything a pipeline test needs, with a helper to build contexts."""

    session: Session
    settings: Settings
    embedder: ControlledEmbedder
    tracker: RecordingTracker
    llm: FakeLlm
    usage: LlmUsage
    postings: dict[str, JobPosting]
    log_records: list[str] = field(default_factory=list)

    def context(self, now: datetime = NOW, with_llm: bool = False) -> PipelineContext:
        """A pipeline context sharing the world's tracker, embedder and usage."""
        return PipelineContext(
            session=self.session,
            settings=self.settings,
            embedder=self.embedder,
            tracker=self.tracker,
            now=now,
            reference_date=REFERENCE_DATE,
            llm_client=self.llm if with_llm else None,
            llm_usage=self.usage,
        )

    def submit(
        self, title: str, pdf: bytes, now: datetime = NOW, with_llm: bool = False
    ) -> UploadOutcome:
        """Upload a PDF to the posting with this title."""
        upload = UploadInput(self.postings[title].id, pdf)
        return submit_upload(self.context(now, with_llm), upload)

    def confirm(
        self,
        token: str | None,
        edits: ProfileEdits | None = None,
        now: datetime = NOW,
        with_llm: bool = False,
    ):
        """Confirm the draft with this token."""
        assert token is not None
        return confirm_draft(self.context(now, with_llm), ConfirmInput(token, edits))

    def apply(self, title: str, pdf: bytes, with_llm: bool = False):
        """Upload and auto-confirm a CV; returns the confirm outcome."""
        outcome = self.submit(title, pdf, with_llm=with_llm)
        return self.confirm(outcome.draft_token, with_llm=with_llm)

    def count(self, model: type) -> int:
        """Number of rows of a model."""
        return self.session.execute(select(func.count()).select_from(model)).scalar_one()

    def application(self, token: str | None) -> Application:
        """The application with this draft token."""
        return self.session.scalars(
            select(Application).where(Application.draft_token == token)
        ).one()

    def sections_of(self, profile_id: int) -> list[Section]:
        """The sections of a profile in reading order."""
        statement = select(Section).where(Section.profile_id == profile_id).order_by(Section.ordinal)
        return list(self.session.scalars(statement))


@pytest.fixture
def world(demo_session: Session, test_settings: Settings) -> World:
    """A seeded demo database with `TOP_K=2` and recording test doubles."""
    settings = test_settings.model_copy(update={"top_k": TOP_K})
    embedder = ControlledEmbedder()
    seeded = seed_postings(demo_session, embedder, SeedOptions(now=NOW))
    embedder.query_calls.clear()
    return World(
        session=demo_session,
        settings=settings,
        embedder=embedder,
        tracker=RecordingTracker(),
        llm=FakeLlm(),
        usage=LlmUsage(),
        postings={posting.title: posting for posting in seeded},
    )


def edits_from(world: World, profile: CandidateProfile, **changes: object) -> ProfileEdits:
    """Edits that reproduce the profile as stored, with fields overridden."""
    fields = {
        "sections": [SectionEdit(s.type, s.text) for s in world.sections_of(profile.id)],
        "years_experience": profile.years_experience,
        "languages": list(profile.languages),
        "location": profile.location,
    }
    fields.update(changes)
    return ProfileEdits(**fields)  # type: ignore[arg-type]


# --- upload -----------------------------------------------------------------


def test_new_cv_creates_profile_sections_contact_and_draft(world: World) -> None:
    """A fresh CV is parsed once; contact is stored apart and scrubbed from texts."""
    pdf = build_cv_pdf("alpha")
    pdf_size = len(pdf)
    upload = UploadInput(world.postings[BACKEND_DEVELOPER_TITLE].id, pdf)

    outcome = submit_upload(world.context(), upload)

    assert outcome.kind is UploadOutcomeKind.DRAFT_CREATED
    assert outcome.cache_hit is False
    assert outcome.no_headings is False
    assert world.tracker.stage_names == EXTRACT_HASH_PARSE
    application = world.application(outcome.draft_token)
    assert application.status is ApplicationStatus.DRAFT
    profile = application.profile
    assert profile.confirmed_at is None
    assert len(world.sections_of(profile.id)) == SECTIONS_PER_CV
    assert profile.contact.email == CANDIDATE_EMAIL
    all_text = " ".join(section.text for section in world.sections_of(profile.id))
    assert CANDIDATE_EMAIL not in all_text
    assert CANDIDATE_NAME not in all_text
    assert CANDIDATE_EMAIL not in str(profile.profile_json)
    assert read_counters(world.session).pdf_bytes_not_stored == pdf_size
    assert upload.pdf_bytes == b""


def test_same_cv_to_a_second_posting_is_a_parse_cache_hit(world: World) -> None:
    """The profile is reused: parse is skipped and the counter increments."""
    pdf = build_cv_pdf("alpha")
    world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    world.tracker.clear()

    outcome = world.submit(ACCOUNTANT_TITLE, pdf)

    assert outcome.kind is UploadOutcomeKind.DRAFT_CREATED
    assert outcome.cache_hit is True
    assert world.tracker.stage_names == EXTRACT_HASH
    assert read_counters(world.session).parse_cache_hits == 1
    assert world.count(CandidateProfile) == 1
    assert world.count(Application) == 2


def test_same_cv_to_the_same_posting_is_a_duplicate(world: World) -> None:
    """A confirmed application blocks a repeat application."""
    pdf = build_cv_pdf("alpha")
    world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    world.tracker.clear()

    outcome = world.submit(BACKEND_DEVELOPER_TITLE, pdf)

    assert outcome.kind is UploadOutcomeKind.DUPLICATE_APPLICATION
    assert outcome.draft_token is None
    assert world.tracker.stage_names == EXTRACT_HASH
    assert world.count(Application) == 1


def test_a_live_draft_redirects_to_itself(world: World) -> None:
    """SC-07: a repeat upload while the draft is alive returns the same token."""
    pdf = build_cv_pdf("alpha")
    first = world.submit(BACKEND_DEVELOPER_TITLE, pdf)

    second = world.submit(BACKEND_DEVELOPER_TITLE, pdf, now=NOW + timedelta(hours=DRAFT_TTL_HOURS))

    assert second.kind is UploadOutcomeKind.DRAFT_EXISTS
    assert second.draft_token == first.draft_token
    assert world.count(Application) == 1


def test_an_expired_draft_is_replaced_by_a_new_one(world: World) -> None:
    """R1 for one draft: the old draft goes without a decision, a new one is made."""
    pdf = build_cv_pdf("alpha")
    first = world.submit(BACKEND_DEVELOPER_TITLE, pdf)
    world.tracker.clear()

    later = NOW + timedelta(hours=DRAFT_TTL_HOURS + 1)
    second = world.submit(BACKEND_DEVELOPER_TITLE, pdf, now=later)

    assert second.kind is UploadOutcomeKind.DRAFT_CREATED
    assert second.draft_token != first.draft_token
    assert world.tracker.stage_names == ["extract", "hash", "retention", "parse"]
    assert world.count(Application) == 1
    assert world.application(second.draft_token).created_at == later
    assert world.count(DecisionRecord) == 0


def test_closed_posting_rejects_uploads_and_confirmations(world: World) -> None:
    """A closed posting takes no new uploads; its draft is gone (SC-22), so it cannot be confirmed."""
    posting = world.postings[BACKEND_DEVELOPER_TITLE]
    draft = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    close_posting(RetentionContext(world.session, NOW, DECISION_RETENTION_DAYS), posting.id)

    with pytest.raises(PostingClosedError):
        world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("beta"))
    with pytest.raises(DraftNotFoundError):
        world.confirm(draft.draft_token)


def test_unreadable_and_too_short_pdfs_pass_through_typed(world: World) -> None:
    """The PDF errors are not wrapped, and the bytes are dropped either way."""
    job_id = world.postings[BACKEND_DEVELOPER_TITLE].id
    corrupt = UploadInput(job_id, b"this is not a pdf")
    short = UploadInput(job_id, build_pdf(["Too short."]))

    with pytest.raises(UnreadablePdfError):
        submit_upload(world.context(), corrupt)
    with pytest.raises(TextTooShortError):
        submit_upload(world.context(), short)

    assert corrupt.pdf_bytes == b"" and short.pdf_bytes == b""
    assert world.count(Application) == 0
    assert world.count(CandidateProfile) == 0


def test_unknown_draft_token_and_confirmed_application_are_rejected(world: World) -> None:
    """Confirmation needs a live draft."""
    with pytest.raises(DraftNotFoundError):
        world.confirm("no-such-token")
    outcome = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    world.confirm(outcome.draft_token)
    with pytest.raises(ApplicationNotDraftError):
        world.confirm(outcome.draft_token)


# --- confirm, filter, rank --------------------------------------------------


def test_confirmation_ranks_the_application_through_filter_embed_score(world: World) -> None:
    """A passing candidate is embedded, scored and ranked."""
    outcome = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha", quality=80))
    world.tracker.clear()

    result = world.confirm(outcome.draft_token)

    assert result.kind is ConfirmOutcomeKind.RANKED
    assert world.tracker.stage_names == FILTER_EMBED_SCORE
    application = world.application(outcome.draft_token)
    assert application.status is ApplicationStatus.RANKED
    assert application.confirmed_at == NOW
    assert application.score == pytest.approx(0.8, abs=APPROX_TOLERANCE)
    assert application.profile.confirmed_at == NOW
    assert read_counters(world.session).embeddings_computed == SECTIONS_PER_CV


def test_matrix_json_marks_evidence_and_claim_sections(world: World) -> None:
    """Evidence (experience) and claim (skills) are flagged; adjusted uses alpha."""
    evidence = world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha", quality=80))
    claim = world.apply(BACKEND_DEVELOPER_TITLE, build_claim_cv_pdf("beta", 80))

    evidence_application = world.session.get(Application, evidence.application_id)
    claim_application = world.session.get(Application, claim.application_id)
    requirement_ids = [r.id for r in world.postings[BACKEND_DEVELOPER_TITLE].requirements]

    evidence_matrix = evidence_application.matrix_json
    assert [entry["requirement_id"] for entry in evidence_matrix] == requirement_ids
    section_ids = {s.id for s in world.sections_of(evidence_application.profile_id)}
    for entry in evidence_matrix:
        assert entry["section_type"] == SectionType.EXPERIENCE.value
        assert entry["evidence"] is True
        assert entry["section_id"] in section_ids
        assert entry["sim"] == pytest.approx(0.8, abs=APPROX_TOLERANCE)
        assert entry["adjusted"] == pytest.approx(0.8, abs=APPROX_TOLERANCE)
    for entry in claim_application.matrix_json:
        assert entry["section_type"] == SectionType.SKILLS.value
        assert entry["evidence"] is False
        assert entry["adjusted"] == pytest.approx(0.8 * SKILLS_COEFFICIENT, abs=APPROX_TOLERANCE)
    assert claim_application.score == pytest.approx(0.8 * SKILLS_COEFFICIENT, abs=APPROX_TOLERANCE)


def test_filter_elimination_stores_the_reason_and_deletes_the_application(world: World) -> None:
    """MIN_YEARS: record with token, application and profile gone, nothing embedded."""
    outcome = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha", years=ONE_YEAR))
    world.tracker.clear()

    result = world.confirm(outcome.draft_token)

    assert result.kind is ConfirmOutcomeKind.ELIMINATED
    assert result.reason_code is StoredReasonCode.MIN_YEARS
    assert result.application_id is None and result.score is None
    assert world.tracker.stage_names == ["filter", "retention"]
    record = world.session.scalars(select(DecisionRecord)).one()
    assert record.reason_code == StoredReasonCode.MIN_YEARS.value
    assert record.application_token == outcome.draft_token
    assert world.count(Application) == 0
    assert world.count(CandidateProfile) == 0
    assert world.embedder.passage_texts_embedded == 0


def test_a_decision_record_blocks_reapplication_until_it_expires(world: World) -> None:
    """Within the retention period a repeat application is a duplicate."""
    pdf = build_cv_pdf("alpha", years=ONE_YEAR)
    world.confirm(world.submit(BACKEND_DEVELOPER_TITLE, pdf).draft_token)
    within = NOW + timedelta(days=DECISION_RETENTION_DAYS)
    after = NOW + timedelta(days=DECISION_RETENTION_DAYS, seconds=1)

    assert world.submit(BACKEND_DEVELOPER_TITLE, pdf, now=within).kind is (
        UploadOutcomeKind.DUPLICATE_APPLICATION
    )
    assert world.submit(BACKEND_DEVELOPER_TITLE, pdf, now=after).kind is (
        UploadOutcomeKind.DRAFT_CREATED
    )


def test_elimination_from_one_posting_keeps_the_profile_of_another(world: World) -> None:
    """Two years pass Backend (min 2) and fail Accountant (min 3)."""
    pdf = build_cv_pdf("alpha", years=TWO_YEARS)
    backend = world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    accountant_draft = world.submit(ACCOUNTANT_TITLE, pdf)

    result = world.confirm(accountant_draft.draft_token)

    assert backend.kind is ConfirmOutcomeKind.RANKED
    assert result.reason_code is StoredReasonCode.MIN_YEARS
    assert world.count(CandidateProfile) == 1
    assert world.count(Application) == 1


# --- summaries --------------------------------------------------------------


def test_top_k_summaries_are_written_only_for_new_entrants(world: World) -> None:
    """With k=2, an application is summarised when it enters, never twice."""
    qualities = {"a": 50, "b": 70, "c": 30, "d": 90}
    cumulative_attempts = []
    outcomes = {}
    for tag, quality in qualities.items():
        outcomes[tag] = world.apply(
            BACKEND_DEVELOPER_TITLE, build_cv_pdf(tag, quality), with_llm=True
        )
        cumulative_attempts.append(world.llm.attempts)

    assert cumulative_attempts == [1, 2, 2, 3]
    summaries = {
        tag: world.session.get(Application, outcome.application_id).summary
        for tag, outcome in outcomes.items()
    }
    assert summaries["c"] is None
    assert [summaries[tag] for tag in "abd"] == [SUMMARY_REPLY] * 3
    assert world.usage.calls == 3
    assert read_counters(world.session).llm_calls == 3

    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("e", 10), with_llm=True)
    assert world.llm.attempts == 3


def test_summary_prompt_contains_only_matrix_selected_sections(world: World) -> None:
    """The unselected education marker and the contact details never reach the LLM."""
    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha", 60), with_llm=True)

    (prompt,) = world.llm.prompts
    assert EXPERIENCE_MARKER in prompt
    assert UNSELECTED_MARKER not in prompt
    for private in (CANDIDATE_NAME, CANDIDATE_EMAIL, CANDIDATE_PHONE):
        assert private not in prompt


def test_unreachable_llm_leaves_the_summary_empty_and_the_flow_running(world: World) -> None:
    """Ollama down: ranked without summary, one attempt per confirmation."""
    world.llm.unavailable = True
    first = world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha", 60), with_llm=True)
    attempts_after_first = world.llm.attempts
    second = world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("beta", 70), with_llm=True)

    assert first.kind is second.kind is ConfirmOutcomeKind.RANKED
    assert world.session.get(Application, first.application_id).summary is None
    assert attempts_after_first == 1
    assert world.llm.attempts == 2
    assert world.usage.failures == 2
    assert read_counters(world.session).llm_calls == 0


def test_no_llm_client_means_no_summary_stage(world: World) -> None:
    """Without a client the summary step does not even open a stage."""
    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    assert "llm_summary" not in world.tracker.stage_names


def test_stage_names_are_recorded_in_order_and_never_nest(world: World) -> None:
    """The full green flow, one stage after another."""
    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"), with_llm=True)
    assert world.tracker.stage_names == [
        "extract", "hash", "parse", "filter", "embed", "score", "llm_summary",
    ]


# --- SC-03 and edits --------------------------------------------------------


def test_reconfirming_with_an_edited_section_keeps_ids_and_rescores_the_other(
    world: World,
) -> None:
    """SC-03: unchanged sections keep their ids; the other ranked application is re-scored."""
    pdf = build_cv_pdf("alpha", quality=40)
    first = world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    first_application = world.session.get(Application, first.application_id)
    profile = first_application.profile
    ids_before = {s.type: s.id for s in world.sections_of(profile.id)}
    assert first_application.score == pytest.approx(0.4, abs=APPROX_TOLERANCE)
    draft = world.submit(ACCOUNTANT_TITLE, pdf)
    embedded_before = world.embedder.passage_texts_embedded
    world.tracker.clear()

    edited = [
        SectionEdit(s.type, s.text.replace("QSCORE40", "QSCORE80"))
        for s in world.sections_of(profile.id)
    ]
    result = world.confirm(
        draft.draft_token,
        edits_from(world, profile, sections=edited, languages=["Ingilizce"], location="Denizli"),
    )

    ids_after = {s.type: s.id for s in world.sections_of(profile.id)}
    changed = SectionType.EXPERIENCE
    assert {t: i for t, i in ids_after.items() if t is not changed} == {
        t: i for t, i in ids_before.items() if t is not changed
    }
    assert ids_after[changed] != ids_before[changed]
    assert result.rescored_applications == 1
    assert world.tracker.stage_names == FILTER_EMBED_SCORE
    assert world.embedder.passage_texts_embedded == embedded_before + 1
    assert first_application.score == pytest.approx(0.8, abs=APPROX_TOLERANCE)
    current_ids = set(ids_after.values())
    assert {e["section_id"] for e in first_application.matrix_json} <= current_ids
    assert result.score == pytest.approx(0.8, abs=APPROX_TOLERANCE)
    assert profile.languages == ["english"]
    assert profile.location == "denizli"
    assert profile.profile_json["location"] == "denizli"
    assert "QSCORE80" in str(profile.profile_json["sections"])


def test_unchanged_edits_rescore_nothing(world: World) -> None:
    """Edits that reproduce the sections do not trigger SC-03 re-scoring."""
    pdf = build_cv_pdf("alpha")
    first = world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    profile = world.session.get(Application, first.application_id).profile
    draft = world.submit(ACCOUNTANT_TITLE, pdf)
    embedded_before = world.embedder.passage_texts_embedded

    result = world.confirm(draft.draft_token, edits_from(world, profile))

    assert result.rescored_applications == 0
    assert world.embedder.passage_texts_embedded == embedded_before


def test_edits_re_scrub_contact_data_from_section_text(world: World) -> None:
    """An email or phone the candidate typed into a section is removed again."""
    draft = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    profile = world.application(draft.draft_token).profile
    leaky = [
        SectionEdit(
            SectionType.SKILLS,
            f"Python and SQL. Reach me at {OTHER_EMAIL} or {OTHER_PHONE} "
            f"or {NEW_CONTACT_EMAIL}.",
        ),
        SectionEdit(SectionType.EXPERIENCE, f"Built services {EXPERIENCE_MARKER}"),
    ]
    contact = ContactEdit(name="Ada Lovelace", email=NEW_CONTACT_EMAIL, phone=None)

    world.confirm(draft.draft_token, edits_from(world, profile, sections=leaky, contact=contact))

    texts = " ".join(s.text for s in world.sections_of(profile.id))
    assert "Python and SQL" in texts
    for private in (OTHER_EMAIL, OTHER_PHONE, NEW_CONTACT_EMAIL, "@"):
        assert private not in texts
        assert private not in str(profile.profile_json)
    stored = world.session.scalars(select(CandidateContact)).one()
    assert (stored.email, stored.phone) == (NEW_CONTACT_EMAIL, None)


def test_sections_that_normalize_to_empty_are_dropped(world: World) -> None:
    """Empty or punctuation-only sections never reach the embedder."""
    draft = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    profile = world.application(draft.draft_token).profile
    blanks = [
        SectionEdit(SectionType.SKILLS, "   "),
        SectionEdit(SectionType.EDUCATION, " -- ,, !! "),
    ]
    stored = [SectionEdit(s.type, s.text) for s in world.sections_of(profile.id)]

    result = world.confirm(
        draft.draft_token, edits_from(world, profile, sections=[*blanks, *stored])
    )

    assert result.kind is ConfirmOutcomeKind.RANKED
    assert len(world.sections_of(profile.id)) == SECTIONS_PER_CV


def test_removed_sections_are_deleted_through_retention_and_counted(world: World) -> None:
    """A section the candidate removed is gone and shows in `records_deleted`."""
    draft = world.submit(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    profile = world.application(draft.draft_token).profile
    kept = [SectionEdit(s.type, s.text) for s in world.sections_of(profile.id)][:-1]

    world.confirm(draft.draft_token, edits_from(world, profile, sections=kept))

    assert len(world.sections_of(profile.id)) == SECTIONS_PER_CV - 1
    assert read_counters(world.session).records_deleted == 1


def test_elimination_after_an_edit_still_refreshes_the_other_application(world: World) -> None:
    """The edit changed a shared profile, so the sibling is re-scored even if this one fails."""
    pdf = build_cv_pdf("alpha", quality=40, years=TWO_YEARS)
    first = world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    first_application = world.session.get(Application, first.application_id)
    profile = first_application.profile
    draft = world.submit(ACCOUNTANT_TITLE, pdf)
    edited = [
        SectionEdit(s.type, s.text.replace("QSCORE40", "QSCORE90"))
        for s in world.sections_of(profile.id)
    ]

    result = world.confirm(draft.draft_token, edits_from(world, profile, sections=edited))

    assert result.kind is ConfirmOutcomeKind.ELIMINATED
    assert result.rescored_applications == 1
    assert first_application.score == pytest.approx(0.9, abs=APPROX_TOLERANCE)
    current_ids = {s.id for s in world.sections_of(profile.id)}
    assert {e["section_id"] for e in first_application.matrix_json} <= current_ids


def test_hr_posting_is_scored_against_its_own_requirements(world: World) -> None:
    """Each application is scored against the requirements of its own posting."""
    result = world.apply(HR_SPECIALIST_TITLE, build_cv_pdf("alpha", 70))
    application = world.session.get(Application, result.application_id)
    hr_ids = [r.id for r in world.postings[HR_SPECIALIST_TITLE].requirements]
    assert [e["requirement_id"] for e in application.matrix_json] == hr_ids


# --- privacy and green rules ------------------------------------------------


def test_pdf_bytes_are_never_written_to_disk(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No write-mode open, no temp file, nothing new in the temp directory."""
    watched = tmp_path / "watched-temp"
    watched.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(watched))
    writes: list[str] = []
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def watch(opener):
        def watching(file, mode="r", *args, **kwargs):
            if any(flag in str(mode) for flag in WRITE_MODE_FLAGS):
                writes.append(f"open:{mode}")
            return opener(file, mode, *args, **kwargs)
        return watching

    def watching_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(f"os.open:{flags}")
        return real_os_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        writes.append("tempfile")
        raise AssertionError("tempfile used")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", watch(real_open))
        patch.setattr(io, "open", watch(real_io_open))
        patch.setattr(os, "open", watching_os_open)
        for name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile", "mkstemp"):
            patch.setattr(tempfile, name, forbidden)
        world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"), with_llm=True)
        world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("beta", years=ONE_YEAR))

    assert writes == []
    assert list(watched.iterdir()) == []


def test_pdf_bytes_do_not_reach_the_database_file(world: World) -> None:
    """After a commit the SQLite file holds sections, not the PDF."""
    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("alpha"))
    world.session.commit()
    database_file = Path(world.settings.database_url.removeprefix("sqlite:///"))
    content = database_file.read_bytes()
    assert b"%PDF" not in content
    assert b"EXPERIENCETEXTMARKER" in content


def test_logs_carry_no_cv_text_only_ids_and_short_hashes(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """Upload, confirm, elimination, duplicate and summary leave no personal data."""
    caplog.set_level(logging.DEBUG)
    pdf = build_cv_pdf("alpha", 70)
    world.apply(BACKEND_DEVELOPER_TITLE, pdf, with_llm=True)
    world.submit(BACKEND_DEVELOPER_TITLE, pdf)
    world.apply(BACKEND_DEVELOPER_TITLE, build_cv_pdf("beta", years=ONE_YEAR))
    raw_hash = world.session.scalars(select(Application.raw_hash)).first()

    logged = caplog.text
    assert short_hash(raw_hash) in logged
    assert raw_hash not in logged
    forbidden = (
        CANDIDATE_NAME, CANDIDATE_EMAIL, "532 123", EXPERIENCE_MARKER,
        UNSELECTED_MARKER, SUMMARY_REPLY, "alpha", "beta", "Candidate profile",
    )
    for text in forbidden:
        assert text not in logged


# --- F1: lookups are measured -------------------------------------------------

LOOKUP_FUNCTIONS = ("find_application", "has_active_decision", "find_cached_profile")


def test_duplicate_check_and_cache_lookup_run_inside_the_hash_stage(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Green rule 6: the lookups are measured work, so they run while `hash` is active."""
    seen: list[tuple[str, str | None]] = []

    def spy(name: str):
        original = getattr(upload_module, name)

        def spied(*args, **kwargs):
            seen.append((name, world.tracker.active_stage))
            return original(*args, **kwargs)

        return spied

    for name in LOOKUP_FUNCTIONS:
        monkeypatch.setattr(upload_module, name, spy(name))
    pdf = build_cv_pdf("alpha")
    world.apply(BACKEND_DEVELOPER_TITLE, pdf)
    world.submit(ACCOUNTANT_TITLE, pdf)
    world.submit(BACKEND_DEVELOPER_TITLE, pdf)

    assert {name for name, _ in seen} == set(LOOKUP_FUNCTIONS)
    assert all(active == "hash" for _, active in seen)


# --- F2: every sibling is re-scored, summaries are refreshed -------------------


def test_after_an_edit_every_sibling_matrix_points_at_existing_sections(world: World) -> None:
    """Ranked and shortlisted siblings are re-scored; no matrix keeps a deleted id."""
    pdf = build_cv_pdf("alpha", quality=40)
    ranked = world.session.get(
        Application, world.apply(BACKEND_DEVELOPER_TITLE, pdf).application_id
    )
    shortlisted = world.session.get(
        Application, world.apply(HR_SPECIALIST_TITLE, pdf).application_id
    )
    shortlisted.status = ApplicationStatus.SHORTLISTED
    profile = ranked.profile
    draft = world.submit(ACCOUNTANT_TITLE, pdf)
    edited = [
        SectionEdit(s.type, s.text.replace("QSCORE40", "QSCORE80"))
        for s in world.sections_of(profile.id)
    ]

    result = world.confirm(draft.draft_token, edits_from(world, profile, sections=edited))

    assert result.rescored_applications == 2
    assert shortlisted.status is ApplicationStatus.SHORTLISTED
    assert ranked.status is ApplicationStatus.RANKED
    existing_ids = set(world.session.scalars(select(Section.id)))
    for sibling in (ranked, shortlisted):
        assert sibling.score == pytest.approx(0.8, abs=APPROX_TOLERANCE)
        assert sibling.matrix_json
        assert {e["section_id"] for e in sibling.matrix_json} <= existing_ids


def test_a_rescored_top_k_sibling_gets_a_fresh_summary_from_the_new_texts(
    world: World,
) -> None:
    """The old summary is dropped; the LLM sees the edited section, not the old one."""
    pdf = build_cv_pdf("alpha", quality=40)
    first = world.apply(BACKEND_DEVELOPER_TITLE, pdf, with_llm=True)
    sibling = world.session.get(Application, first.application_id)
    assert sibling.summary == SUMMARY_REPLY
    profile = sibling.profile
    draft = world.submit(ACCOUNTANT_TITLE, pdf)
    edited = [
        SectionEdit(s.type, s.text.replace("QSCORE40", "QSCORE80 FRESHEDITMARKER"))
        for s in world.sections_of(profile.id)
    ]
    world.llm.reply = "Yeni ozet."
    prompts_before = len(world.llm.prompts)

    result = world.confirm(
        draft.draft_token, edits_from(world, profile, sections=edited), with_llm=True
    )

    new_prompts = world.llm.prompts[prompts_before:]
    assert result.rescored_applications == 1
    assert len(new_prompts) == 2
    assert all("FRESHEDITMARKER" in prompt for prompt in new_prompts)
    assert all("QSCORE40" not in prompt for prompt in new_prompts)
    assert sibling.summary == "Yeni ozet."
    assert world.session.get(Application, result.application_id).summary == "Yeni ozet."
