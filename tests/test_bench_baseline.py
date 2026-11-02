"""End-to-end tests of `BaselineRunner` on the tiny synthetic dataset (T-10).

No GPU, no Ollama and no CodeCarbon: a scoring LLM double, a tracker that
charges fixed amounts, a fake GPU sampler and a temp bench directory.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest
from baseline_support import REASON_MARKER, ScoringLlm
from bench_support import (
    CATEGORY_ACCOUNTANT,
    CATEGORY_DISTRACTOR,
    CATEGORY_FINANCE,
    CATEGORY_HR,
    CATEGORY_IT,
    DURATION_PER_CALL_S,
    MODEL_LOAD_ENERGY_WH,
    Rig,
    build_rig,
    parse_notes,
    touch_records,
)
from pipeline_support import (
    CANDIDATE_EMAIL,
    CANDIDATE_NAME,
    CANDIDATE_PHONE,
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
)
from test_bench_runner import (
    ACCOUNTANT,
    APPLICATIONS,
    BACKEND,
    HR_SPECIALIST,
    SEED,
    SPECS,
    make_plan,
    make_runner,
    read_runs,
    stage_rows,
)

from app.config import Settings
from app.green.tracker import StageTotals
from app.llm.ollama import LlmUnavailableError
from app.models import PipelineName
from bench.baseline import BaselineRunner, rank_entries
from bench.baseline_files import RESULTS_FILE_NAME
from bench.estimate import (
    REPEATS,
    ScenarioCount,
    count_n300_applications,
    estimate_n300,
    format_estimate,
    per_application_seconds,
)
from bench.results import MODEL_LOAD_STAGE, RankedEntry, RunPlan
from bench.run_command import format_baseline_report, run_run
from bench.runner import RunOutcome
from bench.scenario import (
    ScenarioApplication,
    ScenarioConfig,
    ScenarioModels,
    build_scenario,
    scenario_json,
)

MALFORMED_TAG = "CHEF1"
APPLICATION_COUNT = len(APPLICATIONS)
RETRY_ATTEMPTS = 1
POSTING_COUNT = 3
EXPECTED_RANKING = {
    BACKEND: [("IT1", 90), ("IT1", 90), ("IT2", 70), ("ACC1", 60), ("CHEF1", None)],
    ACCOUNTANT: [("ACC1", 60), ("ACC2", 40)],
    HR_SPECIALIST: [("HR1", 80)],
}
LEAK_MARKERS = (
    CANDIDATE_NAME,
    CANDIDATE_EMAIL,
    CANDIDATE_PHONE,
    "Candidate profile",
    REASON_MARKER,
)
BASELINE_STAGES = {"model_load", "store_pdf", "extract", "llm_score", "rank"}
FLOAT_TOLERANCE = 1e-9
FAIL_AFTER_ATTEMPTS = 3
PER_APPLICATION_SECONDS = 2.0
FAIRNESS_SAMPLE = 4
RECORDS_PER_BIG_GROUP = 80
RECORDS_PER_FINANCE_GROUP = 40


@dataclass
class BaselineFinished:
    """A completed baseline run and the doubles that served it."""

    outcome: RunOutcome
    rig: Rig
    llm: ScoringLlm
    plan: RunPlan


def make_baseline(rig: Rig, llm: ScoringLlm) -> BaselineRunner:
    """A `BaselineRunner` on the rig."""
    return BaselineRunner(rig.services, llm)


def execute_baseline(
    tmp_path: Path, settings: Settings, llm: ScoringLlm | None = None
) -> BaselineFinished:
    """Run the default plan once with a scoring LLM."""
    rig = build_rig(tmp_path, settings)
    plan = make_plan(tmp_path)
    scoring_llm = llm or ScoringLlm(malformed_tags=[MALFORMED_TAG])
    outcome = make_baseline(rig, scoring_llm).run(plan)
    return BaselineFinished(outcome, rig, scoring_llm, plan)


@pytest.fixture
def finished(tmp_path: Path, test_settings: Settings) -> BaselineFinished:
    """The default plan, run once through the baseline."""
    return execute_baseline(tmp_path, test_settings)


def store_folder(finished_run: BaselineFinished) -> Path:
    """The published folder of the run, named after its `BenchmarkRun` id."""
    return finished_run.rig.paths.baseline_store / str(finished_run.outcome.run_id)


def read_results(finished_run: BaselineFinished) -> list[dict[str, object]]:
    """The lines of the run's results file."""
    path = store_folder(finished_run) / RESULTS_FILE_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_rankings(finished_run: BaselineFinished) -> dict[str, list[dict[str, object]]]:
    """The postings of the run's rankings file."""
    path = finished_run.rig.paths.rankings_directory / f"{finished_run.outcome.run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["postings"]


def test_one_run_row_with_the_baseline_stage_rows(finished: BaselineFinished) -> None:
    runs = read_runs(finished.rig)
    assert len(runs) == 1
    run = runs[0]
    assert run.pipeline is PipelineName.BASELINE
    assert run.id == finished.outcome.run_id
    assert run.scenario_json == scenario_json(finished.plan.config)
    assert set(stage_rows(run)) == BASELINE_STAGES
    assert stage_rows(run)[MODEL_LOAD_STAGE] == (MODEL_LOAD_ENERGY_WH, DURATION_PER_CALL_S, 1)


def test_every_application_runs_every_stage_without_a_cache(
    finished: BaselineFinished,
) -> None:
    rows = stage_rows(read_runs(finished.rig)[0])
    for stage in ("store_pdf", "extract", "llm_score"):
        assert rows[stage][2] == APPLICATION_COUNT
    resubmits = [application for application in APPLICATIONS if application.is_resubmit]
    assert resubmits
    resubmitted = resubmits[0].record_id
    prompts_of_resubmitted = [p for p in finished.llm.prompts if f"Candidate profile {resubmitted} " in p]
    assert len(prompts_of_resubmitted) == 2


def test_the_llm_is_called_with_json_for_every_application_and_one_retry(
    finished: BaselineFinished,
) -> None:
    run = read_runs(finished.rig)[0]
    assert finished.llm.warm_ups == 1
    expected_calls = APPLICATION_COUNT + RETRY_ATTEMPTS
    assert run.llm_calls == len(finished.llm.prompts) == expected_calls
    assert finished.llm.json_requests == expected_calls
    assert run.llm_prompt_tokens == expected_calls * PROMPT_TOKENS
    assert run.llm_completion_tokens == expected_calls * COMPLETION_TOKENS
    assert (run.parse_cache_hits, run.embed_cache_hits) == (0, 0)


def test_rank_runs_once_per_posting(finished: BaselineFinished) -> None:
    assert stage_rows(read_runs(finished.rig)[0])["rank"][2] == POSTING_COUNT


def test_one_pdf_per_application_in_the_run_folder(finished: BaselineFinished) -> None:
    folder = store_folder(finished)
    assert folder.is_dir()
    assert not (finished.rig.paths.baseline_store / "in_progress").exists()
    pdfs = sorted(folder.glob("*.pdf"))
    assert len(pdfs) == len({pdf.name for pdf in pdfs}) == APPLICATION_COUNT
    source_sizes = [
        finished.plan.records[application.record_id].pdf_path.stat().st_size
        for application in APPLICATIONS
    ]
    assert [pdf.stat().st_size for pdf in pdfs] == source_sizes


def test_results_file_is_appended_in_application_order(finished: BaselineFinished) -> None:
    lines = read_results(finished)
    assert len(lines) == APPLICATION_COUNT
    assert [line["ordinal"] for line in lines] == list(range(APPLICATION_COUNT))
    assert [(line["record_id"], line["posting_title"]) for line in lines] == [
        (application.record_id, application.posting_title) for application in APPLICATIONS
    ]
    assert all(CANDIDATE_NAME in str(line["text"]) for line in lines)
    assert [line["score"] for line in lines][-1] == 90
    chef = next(line for line in lines if line["record_id"] == MALFORMED_TAG)
    assert chef["score"] is None


def test_storage_covers_the_whole_folder_and_both_values_match(
    finished: BaselineFinished,
) -> None:
    run = read_runs(finished.rig)[0]
    folder_size = sum(file.stat().st_size for file in store_folder(finished).rglob("*") if file.is_file())
    assert run.storage_bytes_active == run.storage_bytes_after_retention == folder_size
    assert folder_size > 0


def test_rankings_are_score_descending_with_unscored_last(finished: BaselineFinished) -> None:
    postings = read_rankings(finished)
    assert set(postings) == set(EXPECTED_RANKING)
    for title, expected in EXPECTED_RANKING.items():
        entries = postings[title]
        assert [(entry["record_id"], entry["score"]) for entry in entries] == expected
        assert all(set(entry) == {"record_id", "score"} for entry in entries)


def test_resubmits_stay_in_the_ranking_for_the_quality_step_to_dedupe(
    finished: BaselineFinished,
) -> None:
    backend_ids = [entry["record_id"] for entry in read_rankings(finished)[BACKEND]]
    assert backend_ids.count("IT1") == 2


def test_rank_entries_are_stable_and_put_none_last() -> None:
    entries = [
        RankedEntry("a", None),
        RankedEntry("b", 50),
        RankedEntry("c", 80),
        RankedEntry("d", 50),
        RankedEntry("e", None),
    ]
    assert [entry.record_id for entry in rank_entries(entries)] == ["c", "b", "d", "a", "e"]


def test_notes_carry_the_green_fields_plus_the_pipeline(finished: BaselineFinished) -> None:
    notes = parse_notes(finished.outcome.record.measurement_notes)
    assert notes["pipeline"] == "baseline"
    assert notes["llm_status"] == "available"
    assert notes["applications"] == str(APPLICATION_COUNT)
    assert notes["ranked"] == str(APPLICATION_COUNT)
    assert notes["llm_failures"] == "1"
    assert notes["duplicates_blocked"] == "0"
    assert notes["eliminated_by_reason"] == "none"
    assert notes["embeddings_computed"] == notes["records_deleted"] == "0"
    for key in ("cuda_context", "n", "seed", "scenario", "gpu_stream_wh", "unstaged_share",
                "llm_call_median_s", "llm_call_p90_s", "git", "power_plan"):  # fmt: skip
        assert key in notes


def test_no_cv_data_reaches_notes_rankings_summary_or_logs(
    tmp_path: Path, test_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        finished_run = execute_baseline(tmp_path, test_settings)
    run = read_runs(finished_run.rig)[0]
    rankings_path = finished_run.rig.paths.rankings_directory / f"{finished_run.outcome.run_id}.json"
    counts = (ScenarioCount("scenario", 10),)
    report = format_baseline_report(finished_run.outcome, counts)
    for text in (run.measurement_notes, rankings_path.read_text(encoding="utf-8"), report, caplog.text):
        for marker in LEAK_MARKERS:
            assert marker not in text


def test_an_unreadable_pdf_is_counted_and_not_sent_to_the_llm(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    plan = make_plan(tmp_path)
    plan.records["ACC2"].pdf_path.write_bytes(b"this is not a pdf")
    llm = ScoringLlm(malformed_tags=[MALFORMED_TAG])
    outcome = make_baseline(rig, llm).run(plan)
    counts = outcome.record.result.counts
    assert counts.unreadable == 1
    assert counts.ranked == APPLICATION_COUNT - 1
    rows = stage_rows(read_runs(rig)[0])
    assert rows["store_pdf"][2] == rows["extract"][2] == APPLICATION_COUNT
    assert rows["llm_score"][2] == APPLICATION_COUNT - 1


def test_an_unreachable_llm_at_warm_up_aborts_and_stores_nothing(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    llm = ScoringLlm(unavailable=True)
    with pytest.raises(LlmUnavailableError):
        make_baseline(rig, llm).run(make_plan(tmp_path))
    assert not rig.paths.bench_db.exists()
    assert not rig.paths.rankings_directory.exists()
    assert list(rig.paths.baseline_store.iterdir()) == []
    assert rig.cuda.acquired == rig.cuda.released == 1
    assert llm.prompts == []


def test_an_llm_that_goes_away_mid_run_aborts_and_stores_nothing(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    llm = ScoringLlm(fail_after=FAIL_AFTER_ATTEMPTS)
    with pytest.raises(LlmUnavailableError):
        make_baseline(rig, llm).run(make_plan(tmp_path))
    assert len(llm.prompts) == FAIL_AFTER_ATTEMPTS
    assert not rig.paths.bench_db.exists()
    assert not rig.paths.rankings_directory.exists()
    assert list(rig.paths.baseline_store.iterdir()) == []
    assert rig.sampler.started == rig.sampler.stopped == 1
    assert rig.cuda.released == 1


def test_the_cli_turns_an_aborted_run_into_a_clear_message_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    test_settings: Settings,
) -> None:
    from bench.__main__ import build_parser

    def refuse(*_: object) -> int:
        raise LlmUnavailableError("Ollama is unavailable: connection failed.")

    monkeypatch.setattr("bench.run_command.run_baseline", refuse)
    monkeypatch.setattr("bench.run_command.Settings", lambda: test_settings)
    arguments = build_parser().parse_args(["run", "--pipeline", "baseline"])
    assert run_run(arguments) != 0
    error = capsys.readouterr().err
    assert "Ollama is unavailable" in error
    assert "nothing was stored" in error


def test_earlier_run_folders_are_removed_and_the_current_one_is_kept(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    old_folder = rig.paths.baseline_store / "7"
    old_folder.mkdir(parents=True)
    (old_folder / "old.pdf").write_bytes(b"old")
    stray = rig.paths.baseline_store / "stray.txt"
    stray.write_text("stray", encoding="utf-8")
    plan = make_plan(tmp_path)
    runner = make_baseline(rig, ScoringLlm())
    first = runner.run(plan)
    assert not old_folder.exists()
    assert not stray.exists()
    assert (rig.paths.baseline_store / str(first.run_id)).is_dir()
    second = runner.run(plan)
    remaining = sorted(path.name for path in rig.paths.baseline_store.iterdir())
    assert remaining == [str(second.run_id)]
    assert second.run_id != first.run_id


def test_green_and_baseline_process_the_identical_application_list(
    tmp_path: Path, test_settings: Settings
) -> None:
    records = list(make_plan(tmp_path).records.values())
    config = ScenarioConfig(
        n=FAIRNESS_SAMPLE, seed=SEED, top_k=5, models=ScenarioModels("fake-embed", "fake-llm")
    )
    applications = build_scenario(records, config)
    assert any(application.is_resubmit for application in applications)
    assert applications == build_scenario(records, config)
    green_seen = iterate_through(
        tmp_path / "green", test_settings, config, applications, records, green=True
    )
    baseline_seen = iterate_through(
        tmp_path / "baseline", test_settings, config, applications, records, green=False
    )
    expected = [(a.record_id, a.posting_title) for a in applications]
    assert green_seen == baseline_seen == expected


class RecordingApplications:
    """A sequence that remembers every application a runner iterates over."""

    def __init__(self, applications: list[ScenarioApplication]) -> None:
        self._applications = applications
        self.seen: list[tuple[str, str]] = []

    def __len__(self) -> int:
        return len(self._applications)

    def __getitem__(self, index: int) -> ScenarioApplication:
        return self._applications[index]

    def __iter__(self):  # type: ignore[no-untyped-def]
        for application in self._applications:
            self.seen.append((application.record_id, application.posting_title))
            yield application


def iterate_through(
    directory: Path,
    settings: Settings,
    config: ScenarioConfig,
    applications: list[ScenarioApplication],
    records: list,  # type: ignore[type-arg]
    green: bool,
) -> list[tuple[str, str]]:
    """Run one pipeline over a recording copy of the plan; return what it iterated."""
    rig = build_rig(directory, settings)
    recording = RecordingApplications(applications)
    plan = RunPlan(config, recording, {record.id: record for record in records})  # type: ignore[arg-type]
    if green:
        make_runner(rig)[0].run(plan)
    else:
        make_baseline(rig, ScoringLlm()).run(plan)
    return recording.seen


def test_per_application_seconds_sums_only_the_per_application_stages() -> None:
    stages = {
        "model_load": StageTotals(duration_s=100.0, count=1),
        "store_pdf": StageTotals(duration_s=1.0, count=4),
        "extract": StageTotals(duration_s=3.0, count=4),
        "llm_score": StageTotals(duration_s=16.0, count=4),
        "rank": StageTotals(duration_s=50.0, count=3),
    }
    assert per_application_seconds(stages, 4) == pytest.approx(5.0)
    assert per_application_seconds(stages, 0) == 0.0
    assert per_application_seconds({}, 4) == 0.0


def test_the_n300_estimate_multiplies_per_application_seconds_by_the_counts() -> None:
    counts = (ScenarioCount("scenario A", 300), ScenarioCount("default scenario", 667))
    estimate = estimate_n300(PER_APPLICATION_SECONDS, counts)
    assert [scenario.seconds for scenario in estimate.scenarios] == [600.0, 1334.0]
    assert estimate.repeats == REPEATS == 3
    assert estimate.total_seconds == pytest.approx(3 * (600.0 + 1334.0))
    text = format_estimate(estimate)
    assert "300 applications" in text and "667 applications" in text
    assert "3 repeats of both" in text and "5802 s (1 h 36 min)" in text


def test_n300_counts_come_from_build_scenario(tmp_path: Path) -> None:
    records = touch_records(
        tmp_path,
        {
            CATEGORY_IT: RECORDS_PER_BIG_GROUP,
            CATEGORY_ACCOUNTANT: RECORDS_PER_FINANCE_GROUP,
            CATEGORY_FINANCE: RECORDS_PER_FINANCE_GROUP,
            CATEGORY_HR: RECORDS_PER_BIG_GROUP,
            CATEGORY_DISTRACTOR: RECORDS_PER_BIG_GROUP,
        },
    )
    base = ScenarioConfig(
        n=50, seed=SEED, top_k=5, models=ScenarioModels("fake-embed", "fake-llm")
    )
    scenario_a, default = count_n300_applications(records, base)
    from dataclasses import replace

    assert scenario_a.applications == len(
        build_scenario(records, replace(base, n=300, jobs_per_cv=1, resubmit_rate=0.0))
    )
    assert default.applications == len(build_scenario(records, replace(base, n=300)))
    assert scenario_a.applications == 300
    assert default.applications > 2 * 300


def test_the_baseline_report_prints_total_duration_and_the_estimate(
    finished: BaselineFinished,
) -> None:
    counts = (ScenarioCount("scenario A", 300), ScenarioCount("default scenario", 667))
    report = format_baseline_report(finished.outcome, counts)
    assert "Total duration:" in report
    assert "N=300 duration estimate" in report
    assert "scenario A: 300 applications" in report
    assert "3 repeats of both" in report
    assert "Run " in report and "(baseline)" in report


def test_run_options_select_the_baseline() -> None:
    from bench.__main__ import build_parser

    arguments = build_parser().parse_args(
        ["run", "--pipeline", "baseline", "--n", "50", "--jobs-per-cv", "1",
         "--resubmit-rate", "0", "--repeats", "2"]
    )  # fmt: skip
    assert arguments.pipeline == "baseline"
    assert (arguments.n, arguments.jobs_per_cv, arguments.resubmit_rate) == (50, 1, 0.0)
    assert arguments.handler is run_run
