"""`GreenRunner`: the green pipeline of SPEC 6.1 measured as a benchmark run.

The runner drives the same `submit_upload` and `confirm_draft` functions the web
layer will use, against a scratch database that is deleted and recreated for
every run (`data/bench/bench_green.db`). Per application it commits twice, once
after the upload and once after the confirmation, because the web flow commits
once per request.

Reading the PDF bytes, the commits and the ranking lookups are harness work
outside any stage; the baseline reads PDFs the same way. The share of wall time
that no stage covers is reported as `unstaged_wall_s`.

KVKK: the harness map from `raw_hash` to dataset record id holds ids only. No
CV text, name, email or phone is logged or stored.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import create_db_engine, create_demo_schema, create_session_factory
from app.green.counters import CounterName, CounterSnapshot, increment_counter, read_counters
from app.ingest.pdf import PdfIngestError
from app.llm.ollama import LlmUnavailableError
from app.llm.usage import LlmUsage
from app.models import Application, PipelineName
from app.pipeline import (
    ConfirmInput,
    ConfirmOutcome,
    ConfirmOutcomeKind,
    PipelineContext,
    UploadInput,
    UploadOutcomeKind,
    confirm_draft,
    submit_upload,
)
from app.pipeline.context import SCORED_STATUSES, STAGE_RETENTION
from app.privacy.retention import RetentionContext, close_posting
from app.scoring.embed import EmbeddingProvider
from app.seed import SeedOptions, seed_postings
from bench.llm_timing import TimedLlm, WarmLlmClient
from bench.results import (
    LLM_STATUS_AVAILABLE,
    LLM_STATUS_UNAVAILABLE,
    MODEL_LOAD_STAGE,
    PipelineResult,
    RankedEntry,
    Rankings,
    RunPlan,
    WorkloadCounts,
)
from bench.runner import BenchmarkRunner, RunnerServices, read_cv_bytes
from bench.scenario import ScenarioApplication
from bench.store import BenchmarkStore, sqlite_url

logger = logging.getLogger(__name__)

DATABASE_FILE_SUFFIXES: tuple[str, ...] = ("", "-journal", "-wal", "-shm")
VACUUM_STATEMENT = "VACUUM"
NO_ELIMINATIONS = "none"
REASON_SEPARATOR = ","
REASON_COUNT_FORMAT = "{reason}:{count}"


@dataclass(frozen=True)
class StorageSizes:
    """Database size in bytes at the two measurement points (SPEC 7.6)."""

    active: int
    after_retention: int


class LoadableEmbedder(EmbeddingProvider, Protocol):
    """An embedding provider whose model is loaded explicitly, once."""

    def load(self) -> None:
        """Load the model into memory; later calls do nothing."""
        ...


def database_files(database_path: Path) -> list[Path]:
    """The SQLite file and the journal, WAL and shared-memory files beside it."""
    return [
        database_path.with_name(database_path.name + suffix)
        for suffix in DATABASE_FILE_SUFFIXES
    ]


def database_size_bytes(database_path: Path) -> int:
    """Total size of the database file and any journal or WAL sibling."""
    return sum(
        file.stat().st_size for file in database_files(database_path) if file.exists()
    )


def format_eliminations(counts: WorkloadCounts) -> str:
    """`MIN_YEARS:3,...` for the notes, or `none`."""
    if not counts.eliminated_by_reason:
        return NO_ELIMINATIONS
    return REASON_SEPARATOR.join(
        REASON_COUNT_FORMAT.format(reason=reason, count=count)
        for reason, count in sorted(counts.eliminated_by_reason.items())
    )


class GreenRunner(BenchmarkRunner):
    """Runs the green pipeline over a scenario and measures it.

    Args:
        services: Injected runner dependencies.
        embedder: The embedding model; loaded in stage `model_load`.
        llm: The LLM client, warmed up before measurement. If Ollama is
            unreachable at warm-up the run continues without summaries.
        store: Result storage; defaults to the one for `services.paths`.
    """

    def __init__(
        self,
        services: RunnerServices,
        embedder: LoadableEmbedder,
        llm: WarmLlmClient,
        store: BenchmarkStore | None = None,
    ) -> None:
        super().__init__(services, store)
        self._embedder = embedder
        self._raw_llm = llm
        self._engine: Engine | None = None
        self._sessions: sessionmaker[Session] | None = None
        self._job_ids: dict[str, int] = {}
        self._llm: TimedLlm | None = None
        self._llm_usage = LlmUsage()
        self._record_by_hash: dict[str, str] = {}

    @property
    def pipeline(self) -> PipelineName:
        """The green pipeline."""
        return PipelineName.GREEN

    def prepare(self, plan: RunPlan) -> None:
        """Recreate the scratch database, load the models and seed the postings."""
        self._llm_usage = LlmUsage()
        self._record_by_hash = {}
        self._open_fresh_database()
        self._load_models()
        self._seed_postings()

    def run_applications(self, plan: RunPlan) -> WorkloadCounts:
        """Upload and confirm every application, each in its own transactions."""
        counts = WorkloadCounts()
        for application in plan.applications:
            counts.submitted += 1
            self._apply(plan, application, counts)
        self._record_llm_calls_avoided(counts)
        return counts

    def finalize(self, plan: RunPlan, counts: WorkloadCounts) -> PipelineResult:
        """Measure storage, save the rankings, close the postings, vacuum, report."""
        storage_active = database_size_bytes(self._database_path)
        rankings = self._collect_rankings()
        self._run_retention()
        storage_after = database_size_bytes(self._database_path)
        sizes = StorageSizes(storage_active, storage_after)
        return self._build_result(counts, sizes, rankings)

    def close(self) -> None:
        """Dispose the engine so the database file can be deleted next run."""
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._sessions = None

    @property
    def _database_path(self) -> Path:
        """Location of the scratch database."""
        return self._services.paths.green_db

    def _open_fresh_database(self) -> None:
        """Delete any old scratch database and create an empty one."""
        for file in database_files(self._database_path):
            file.unlink(missing_ok=True)
        self._engine = create_db_engine(sqlite_url(self._database_path))
        create_demo_schema(self._engine)
        self._sessions = create_session_factory(self._engine)

    def _load_models(self) -> None:
        """Stage `model_load`: the embedding model and the LLM warm-up."""
        with self._services.tracker.stage(MODEL_LOAD_STAGE):
            self._embedder.load()
            self._warm_up_llm()

    def _warm_up_llm(self) -> None:
        """Warm Ollama up; continue without summaries when it is unreachable."""
        timed = TimedLlm(self._raw_llm)
        try:
            timed.warm_up()
        except LlmUnavailableError:
            logger.warning("LLM warm-up failed; the run continues without summaries")
            self._llm = None
            return
        self._llm = timed

    def _seed_postings(self) -> None:
        """Seed the three postings in benchmark mode (language and location filters off)."""
        with self._open_session() as session:
            options = SeedOptions(now=self._services.clock())
            postings = seed_postings(session, self._embedder, options)
            session.commit()
            self._job_ids = {posting.title: posting.id for posting in postings}

    def _open_session(self) -> Session:
        """A new session on the scratch database."""
        if self._sessions is None:
            raise RuntimeError("prepare() must run before the database is used.")
        return self._sessions()

    def _context(self, session: Session) -> PipelineContext:
        """Pipeline dependencies for one application."""
        settings = self._services.settings
        return PipelineContext(
            session=session,
            settings=settings,
            embedder=self._embedder,
            tracker=self._services.tracker,
            now=self._services.clock(),
            reference_date=settings.bench_cv_reference_date,
            llm_client=self._llm,
            llm_usage=self._llm_usage,
        )

    def _apply(
        self, plan: RunPlan, application: ScenarioApplication, counts: WorkloadCounts
    ) -> None:
        """Process one application and count what happened to it."""
        with self._open_session() as session:
            try:
                token = self._submit(session, plan, application)
                if token is None:
                    counts.blocked += 1
                    return
                outcome = self._confirm(session, token)
            except PdfIngestError as error:
                session.rollback()
                counts.unreadable += 1
                logger.warning(
                    "Record %s is unreadable: %s", application.record_id, type(error).__name__
                )
                return
        self._count_outcome(counts, outcome)

    def _submit(
        self, session: Session, plan: RunPlan, application: ScenarioApplication
    ) -> str | None:
        """Upload and commit; return the draft token, or `None` when blocked."""
        upload = UploadInput(
            self._job_ids[application.posting_title],
            read_cv_bytes(plan, application.record_id),
        )
        outcome = submit_upload(self._context(session), upload)
        session.commit()
        if outcome.kind is not UploadOutcomeKind.DRAFT_CREATED or outcome.draft_token is None:
            return None
        self._remember_record(session, outcome.draft_token, application.record_id)
        return outcome.draft_token

    def _remember_record(self, session: Session, draft_token: str, record_id: str) -> None:
        """Map the draft's `raw_hash` to its dataset record id (ids only)."""
        raw_hash = session.scalar(
            select(Application.raw_hash).where(Application.draft_token == draft_token)
        )
        self._record_by_hash.setdefault(raw_hash, record_id)

    def _confirm(self, session: Session, draft_token: str) -> ConfirmOutcome:
        """Auto-confirm a draft without corrections and commit."""
        outcome = confirm_draft(self._context(session), ConfirmInput(draft_token))
        session.commit()
        return outcome

    @staticmethod
    def _count_outcome(counts: WorkloadCounts, outcome: ConfirmOutcome) -> None:
        """Add a confirmation result to the counts."""
        if outcome.kind is ConfirmOutcomeKind.RANKED:
            counts.ranked += 1
        elif outcome.reason_code is not None:
            counts.eliminated_by_reason[outcome.reason_code.value] += 1

    def _record_llm_calls_avoided(self, counts: WorkloadCounts) -> None:
        """Decision T9-3: confirmed applications minus LLM calls, written once."""
        avoided = max(0, counts.confirmed - self._llm_usage.calls)
        with self._open_session() as session:
            increment_counter(session, CounterName.LLM_CALLS_AVOIDED, avoided)
            session.commit()

    def _collect_rankings(self) -> Rankings:
        """Decision T9-1: the final ranking per posting, as record ids and scores."""
        with self._open_session() as session:
            return {
                title: self._ranking_of(session, job_id)
                for title, job_id in self._job_ids.items()
            }

    def _ranking_of(self, session: Session, job_id: int) -> list[RankedEntry]:
        """Scored applications of a posting: score descending, id ascending."""
        statement = (
            select(Application.raw_hash, Application.score)
            .where(Application.job_id == job_id, Application.status.in_(SCORED_STATUSES))
            .order_by(Application.score.desc(), Application.id)
        )
        return [
            RankedEntry(self._record_by_hash[raw_hash], float(score))
            for raw_hash, score in session.execute(statement)
        ]

    def _run_retention(self) -> None:
        """Stage `retention`: close every posting (R3), commit, then `VACUUM`."""
        settings = self._services.settings
        with self._services.tracker.stage(STAGE_RETENTION):
            with self._open_session() as session:
                context = RetentionContext(
                    session, self._services.clock(), settings.decision_retention_days
                )
                for job_id in self._job_ids.values():
                    close_posting(context, job_id)
                session.commit()
            self._vacuum()

    def _vacuum(self) -> None:
        """Shrink the file; `VACUUM` cannot run inside a transaction."""
        if self._engine is None:
            raise RuntimeError("prepare() must run before the database is used.")
        with self._engine.connect() as connection:
            connection.exec_driver_sql(VACUUM_STATEMENT)

    def _build_result(
        self, counts: WorkloadCounts, sizes: StorageSizes, rankings: Rankings
    ) -> PipelineResult:
        """Gather the counters, LLM usage and notes into the pipeline result."""
        with self._open_session() as session:
            counters = read_counters(session)
        usage = self._llm_usage
        return PipelineResult(
            counts=counts,
            storage_bytes_active=sizes.active,
            storage_bytes_after_retention=sizes.after_retention,
            llm_calls=usage.calls,
            llm_prompt_tokens=usage.prompt_tokens,
            llm_completion_tokens=usage.completion_tokens,
            parse_cache_hits=counters.parse_cache_hits,
            embed_cache_hits=counters.embed_cache_hits,
            llm_status=LLM_STATUS_UNAVAILABLE if self._llm is None else LLM_STATUS_AVAILABLE,
            llm_call_seconds=tuple(self._llm.call_seconds) if self._llm else (),
            rankings=rankings,
            notes=self._pipeline_notes(counts, counters),
        )

    def _pipeline_notes(
        self, counts: WorkloadCounts, counters: CounterSnapshot
    ) -> dict[str, str]:
        """Aggregate counts for `measurement_notes`."""
        return {
            "applications": str(counts.submitted),
            "ranked": str(counts.ranked),
            "duplicates_blocked": str(counts.blocked),
            "unreadable": str(counts.unreadable),
            "eliminated_by_reason": format_eliminations(counts),
            "embeddings_computed": str(counters.embeddings_computed),
            "records_deleted": str(counters.records_deleted),
            "llm_failures": str(self._llm_usage.failures),
        }

