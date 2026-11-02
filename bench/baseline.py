"""`BaselineRunner`: the conventional pipeline of SPEC 7.4 measured as a benchmark run.

For every application of the scenario, resubmits included and with no cache at
all, the baseline runs three stages:

* `store_pdf`: copy the PDF bytes into the run's folder under
  `data/bench/baseline_store/`;
* `extract`: extract the text from scratch;
* `llm_score`: send the full text (cut to `MAX_CV_CHARS`) and the posting's
  requirements to the LLM and append text, score and reason to a JSON Lines
  file in the same folder.

After the last application it ranks each posting once (stage `rank`, decision
T10-3), by LLM score descending with unscored applications last. Nothing is
ever deleted or filtered.

Fairness: the runner shares `BenchmarkRunner`, the scenario, the harness PDF
read and the LLM options with the green runner, so the two differ only in the
pipeline itself. An unreachable Ollama, at warm-up or mid-run, aborts the run
with no stored result (decision T10-1).

KVKK: the store intentionally holds CV bytes, texts and LLM output (that is
what a conventional pipeline keeps). None of it reaches a log, the console, the
rankings file, the notes or a report: those carry ids, scores and counts only.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.ingest.pdf import PdfIngestError, extract_text
from app.llm.prompts import BaselinePromptInput
from app.llm.tasks import score_cv_with_llm
from app.llm.usage import LlmUsage
from app.models import PipelineName
from app.pipeline.context import STAGE_EXTRACT
from app.seed import SEED_POSTINGS
from bench.baseline_files import BaselineFiles, StoredResult
from bench.llm_timing import TimedLlm, WarmLlmClient
from bench.pipeline_notes import PipelineNoteFacts, build_pipeline_notes
from bench.results import (
    LLM_STATUS_AVAILABLE,
    MODEL_LOAD_STAGE,
    PipelineResult,
    RankedEntry,
    Rankings,
    RunPlan,
    WorkloadCounts,
)
from bench.runner import BenchmarkRunner, RunnerServices, RunOutcome, read_cv_bytes
from bench.scenario import POSTING_TITLES, ScenarioApplication
from bench.store import BenchmarkStore

logger = logging.getLogger(__name__)

STAGE_STORE_PDF = "store_pdf"
STAGE_LLM_SCORE = "llm_score"
STAGE_RANK = "rank"
PIPELINE_NOTE_KEY = "pipeline"
REQUIREMENTS_BY_TITLE: dict[str, tuple[str, ...]] = {
    seed.title: tuple(requirement.text for requirement in seed.requirements)
    for seed in SEED_POSTINGS
}
NO_SCORE_SORT_VALUE = 0.0


@dataclass(frozen=True)
class NumberedApplication:
    """An application together with its position in the scenario."""

    ordinal: int
    application: ScenarioApplication


def score_sort_key(entry: RankedEntry) -> tuple[bool, float]:
    """Sort key: scored entries by score descending, unscored entries last."""
    if entry.score is None:
        return (True, NO_SCORE_SORT_VALUE)
    return (False, -entry.score)


def rank_entries(entries: Sequence[RankedEntry]) -> list[RankedEntry]:
    """Order entries by LLM score descending, `None` last, stable by input order."""
    return sorted(entries, key=score_sort_key)


class BaselineRunner(BenchmarkRunner):
    """Runs the conventional pipeline over a scenario and measures it.

    Args:
        services: Injected runner dependencies.
        llm: The LLM client, warmed up before measurement. It must be
            reachable: otherwise the run aborts.
        store: Result storage; defaults to the one for `services.paths`.
    """

    def __init__(
        self,
        services: RunnerServices,
        llm: WarmLlmClient,
        store: BenchmarkStore | None = None,
    ) -> None:
        super().__init__(services, store)
        self._raw_llm = llm
        self._files = BaselineFiles(services.paths.baseline_store)
        self._llm = TimedLlm(llm)
        self._llm_usage = LlmUsage()
        self._scored: dict[str, list[RankedEntry]] = {}

    @property
    def pipeline(self) -> PipelineName:
        """The baseline pipeline."""
        return PipelineName.BASELINE

    def run(self, plan: RunPlan) -> RunOutcome:
        """Run the plan once; an aborted run leaves no folder and no stored row.

        Raises:
            LlmUnavailableError: If Ollama is unreachable at warm-up or mid-run.
            IdleDataError: If the idle power file is missing or broken.
        """
        try:
            outcome = super().run(plan)
        except BaseException:
            self._files.discard()
            raise
        self._files.publish(outcome.run_id)
        return outcome

    def prepare(self, plan: RunPlan) -> None:
        """Clear the earlier stores, open a fresh one and warm the LLM up."""
        self._llm = TimedLlm(self._raw_llm)
        self._llm_usage = LlmUsage()
        self._scored = {title: [] for title in POSTING_TITLES}
        self._files.begin()
        with self._services.tracker.stage(MODEL_LOAD_STAGE):
            self._llm.warm_up()

    def run_applications(self, plan: RunPlan) -> WorkloadCounts:
        """Store, extract and score every application, resubmits included."""
        counts = WorkloadCounts()
        for ordinal, application in enumerate(plan.applications):
            counts.submitted += 1
            self._process(plan, NumberedApplication(ordinal, application), counts)
        return counts

    def finalize(self, plan: RunPlan, counts: WorkloadCounts) -> PipelineResult:
        """Rank every posting once, measure the store and report."""
        rankings = self._rank_postings()
        return self._build_result(counts, self._files.size_bytes(), rankings)

    def _process(
        self, plan: RunPlan, numbered: NumberedApplication, counts: WorkloadCounts
    ) -> None:
        """One application through `store_pdf`, `extract` and `llm_score`."""
        application = numbered.application
        pdf_bytes = read_cv_bytes(plan, application.record_id)
        with self._services.tracker.stage(STAGE_STORE_PDF):
            self._files.write_pdf(numbered.ordinal, application.record_id, pdf_bytes)
        text = self._extract(pdf_bytes, application.record_id)
        if text is None:
            counts.unreadable += 1
            return
        self._score(numbered, text)
        counts.ranked += 1

    def _extract(self, pdf_bytes: bytes, record_id: str) -> str | None:
        """Stage `extract`: the text, or `None` when the PDF cannot be read."""
        try:
            with self._services.tracker.stage(STAGE_EXTRACT):
                return extract_text(pdf_bytes)
        except PdfIngestError as error:
            logger.warning("Record %s is unreadable: %s", record_id, type(error).__name__)
            return None

    def _score(self, numbered: NumberedApplication, text: str) -> None:
        """Stage `llm_score`: ask the LLM, append the result, remember the score."""
        application = numbered.application
        prompt_input = BaselinePromptInput(
            REQUIREMENTS_BY_TITLE[application.posting_title],
            text,
            self._services.settings.max_cv_chars,
        )
        with self._services.tracker.stage(STAGE_LLM_SCORE):
            answer = score_cv_with_llm(self._llm, prompt_input, self._llm_usage)
            self._files.append_result(
                StoredResult(
                    numbered.ordinal,
                    application.record_id,
                    application.posting_title,
                    text,
                    answer.score,
                    answer.reason,
                )
            )
        entry = RankedEntry(application.record_id, answer.score)
        self._scored[application.posting_title].append(entry)

    def _rank_postings(self) -> Rankings:
        """Stage `rank`, once per posting (decision T10-3)."""
        rankings: Rankings = {}
        for title in POSTING_TITLES:
            with self._services.tracker.stage(STAGE_RANK):
                rankings[title] = rank_entries(self._scored[title])
        return rankings

    def _build_result(
        self, counts: WorkloadCounts, store_bytes: int, rankings: Rankings
    ) -> PipelineResult:
        """Gather the store size, LLM usage and notes into the pipeline result.

        Nothing is deleted, so the storage is the same before and after retention.
        """
        usage = self._llm_usage
        return PipelineResult(
            counts=counts,
            storage_bytes_active=store_bytes,
            storage_bytes_after_retention=store_bytes,
            llm_calls=usage.calls,
            llm_prompt_tokens=usage.prompt_tokens,
            llm_completion_tokens=usage.completion_tokens,
            parse_cache_hits=0,
            embed_cache_hits=0,
            llm_status=LLM_STATUS_AVAILABLE,
            llm_call_seconds=tuple(self._llm.call_seconds),
            rankings=rankings,
            notes=self._pipeline_notes(counts),
        )

    def _pipeline_notes(self, counts: WorkloadCounts) -> dict[str, str]:
        """The green runner's note fields plus `pipeline=baseline`."""
        facts = PipelineNoteFacts(
            embeddings_computed=0,
            records_deleted=0,
            llm_failures=self._llm_usage.failures,
        )
        return {
            **build_pipeline_notes(counts, facts),
            PIPELINE_NOTE_KEY: self.pipeline.value,
        }
