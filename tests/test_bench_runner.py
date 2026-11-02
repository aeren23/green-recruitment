"""End-to-end tests of `GreenRunner` on a tiny synthetic dataset.

No GPU, no Ollama and no CodeCarbon: a fake embedder, a fake LLM, a tracker that
charges fixed amounts, a fake GPU sampler and a temp bench directory.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from bench_support import (
    CATEGORY_ACCOUNTANT,
    CATEGORY_DISTRACTOR,
    CATEGORY_HR,
    CATEGORY_IT,
    DURATION_PER_CALL_S,
    ENERGY_PER_CALL_WH,
    FAKE_ENVIRONMENT,
    FAKE_GPU_READING,
    GPU_SHARE,
    IDLE_POWER_W,
    KG_PER_CALL,
    MODEL_LOAD_ENERGY_WH,
    ONE_YEAR_CV,
    CvRecordSpec,
    FakeCudaHolder,
    LoadableEmbedder,
    Rig,
    WarmFakeLlm,
    build_dataset_tree,
    build_rig,
    parse_notes,
)
from pipeline_support import (
    CANDIDATE_EMAIL,
    CANDIDATE_NAME,
    CANDIDATE_PHONE,
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import Settings
from app.db import create_db_engine, create_session_factory
from app.green.tracker import StageTotals, TaskReading
from app.models import BenchmarkRun, PipelineName
from bench.environment import find_ollama_executable, parse_power_plan_name
from bench.gpu_sampler import UNAVAILABLE_READING
from bench.green import GreenRunner
from bench.idle import IdleDataError, measure_with_cuda_context
from bench.results import (
    MODEL_LOAD_STAGE,
    RunPlan,
    WorkloadCounts,
    compute_run_totals,
    percentile,
    subtract_totals,
)
from bench.run_command import run_run
from bench.run_summary import format_run_summary
from bench.runner import RunOutcome
from bench.scenario import ScenarioApplication, ScenarioConfig, ScenarioModels, scenario_json
from bench.store import sqlite_url

BACKEND = "Backend Developer"
ACCOUNTANT = "Accountant"
HR_SPECIALIST = "HR Specialist"
SCORE_TOLERANCE = 1e-4
FLOAT_TOLERANCE = 1e-9
NOTE_TOLERANCE = 1e-5
SEED = 3
SPECS = (
    CvRecordSpec("IT1", CATEGORY_IT, 90),
    CvRecordSpec("IT2", CATEGORY_IT, 70),
    CvRecordSpec("HR1", CATEGORY_HR, 80),
    CvRecordSpec("ACC1", CATEGORY_ACCOUNTANT, 60),
    CvRecordSpec("ACC2", CATEGORY_ACCOUNTANT, 40, years=ONE_YEAR_CV),
    CvRecordSpec("CHEF1", CATEGORY_DISTRACTOR, 50),
)
APPLICATIONS = (
    ScenarioApplication("IT1", BACKEND),
    ScenarioApplication("IT2", BACKEND),
    ScenarioApplication("HR1", HR_SPECIALIST),
    ScenarioApplication("ACC1", ACCOUNTANT),
    ScenarioApplication("ACC1", BACKEND),
    ScenarioApplication("CHEF1", BACKEND),
    ScenarioApplication("ACC2", ACCOUNTANT),
    ScenarioApplication("IT1", BACKEND, is_resubmit=True),
)
EXPECTED_RANKINGS = {
    BACKEND: [("IT1", 0.9), ("IT2", 0.7), ("ACC1", 0.6), ("CHEF1", 0.5)],
    ACCOUNTANT: [("ACC1", 0.6)],
    HR_SPECIALIST: [("HR1", 0.8)],
}
EXPECTED_RANKED = 6
LEAK_MARKERS = (CANDIDATE_NAME, CANDIDATE_EMAIL, CANDIDATE_PHONE, "Candidate profile")


@dataclass
class Finished:
    """A completed run and the doubles that served it."""

    outcome: RunOutcome
    rig: Rig
    embedder: LoadableEmbedder
    llm: WarmFakeLlm
    plan: RunPlan
    runner: GreenRunner


def make_plan(
    root: Path,
    specs: tuple[CvRecordSpec, ...] = SPECS,
    applications: tuple[ScenarioApplication, ...] = APPLICATIONS,
) -> RunPlan:
    """Build the synthetic dataset on disk and a plan over it."""
    records = build_dataset_tree(root / "dataset", specs)
    config = ScenarioConfig(
        n=len(specs), seed=SEED, top_k=5, models=ScenarioModels("fake-embed", "fake-llm")
    )
    return RunPlan(config, applications, {record.id: record for record in records})


def make_runner(rig: Rig, llm: WarmFakeLlm | None = None) -> tuple[GreenRunner, LoadableEmbedder, WarmFakeLlm]:
    """A `GreenRunner` with fake models."""
    embedder, fake_llm = LoadableEmbedder(), llm or WarmFakeLlm()
    return GreenRunner(rig.services, embedder, fake_llm), embedder, fake_llm


def execute(tmp_path: Path, settings: Settings, llm: WarmFakeLlm | None = None) -> Finished:
    """Run the default plan once."""
    rig = build_rig(tmp_path, settings)
    plan = make_plan(tmp_path)
    runner, embedder, fake_llm = make_runner(rig, llm)
    return Finished(runner.run(plan), rig, embedder, fake_llm, plan, runner)


@pytest.fixture
def finished(tmp_path: Path, test_settings: Settings) -> Finished:
    """The default plan, run once with everything available."""
    return execute(tmp_path, test_settings)


def read_runs(rig: Rig) -> list[BenchmarkRun]:
    """All stored runs with their stage rows."""
    engine = create_db_engine(sqlite_url(rig.paths.bench_db))
    try:
        with create_session_factory(engine)() as session:
            statement = select(BenchmarkRun).options(selectinload(BenchmarkRun.stage_measurements))
            runs = list(session.scalars(statement.order_by(BenchmarkRun.id)))
            session.expunge_all()
            return runs
    finally:
        engine.dispose()


def stage_rows(run: BenchmarkRun) -> dict[str, tuple[float, float, int]]:
    """`stage -> (energy_wh, duration_s, count)` of a stored run."""
    return {
        row.stage: (row.energy_wh, row.duration_s, row.count)
        for row in run.stage_measurements
    }


def measured_calls(run: BenchmarkRun) -> int:
    """Total stage calls of a run, `model_load` excluded."""
    return sum(
        count for stage, (_, _, count) in stage_rows(run).items() if stage != MODEL_LOAD_STAGE
    )


def test_run_and_stage_rows_are_written(finished: Finished) -> None:
    runs = read_runs(finished.rig)
    assert len(runs) == 1
    run = runs[0]
    assert run.id == finished.outcome.run_id
    assert run.pipeline is PipelineName.GREEN
    assert run.scenario_json == scenario_json(finished.plan.config)
    assert set(stage_rows(run)) >= {
        "model_load", "extract", "hash", "parse", "filter", "embed", "score",
        "llm_summary", "retention",
    }  # fmt: skip
    assert run.precision_at_k is None
    assert run.overlap_at_k is None


def test_model_load_is_recorded_but_excluded_from_gross(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    rows = stage_rows(run)
    assert rows[MODEL_LOAD_STAGE] == (MODEL_LOAD_ENERGY_WH, DURATION_PER_CALL_S, 1)
    calls = measured_calls(run)
    assert run.energy_wh_gross == pytest.approx(calls * ENERGY_PER_CALL_WH)
    assert run.energy_wh_gross < MODEL_LOAD_ENERGY_WH
    assert run.duration_s == pytest.approx(calls * DURATION_PER_CALL_S)


def test_net_energy_and_co2_follow_the_formulas(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    expected_net = run.energy_wh_gross - IDLE_POWER_W * run.duration_s / 3600
    assert run.idle_power_w == IDLE_POWER_W
    assert run.energy_wh_net == pytest.approx(expected_net, abs=FLOAT_TOLERANCE)
    assert run.co2_g == pytest.approx(measured_calls(run) * KG_PER_CALL * 1000)


def test_storage_is_measured_before_and_after_retention(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    assert run.storage_bytes_after_retention > 0
    assert run.storage_bytes_after_retention <= run.storage_bytes_active


def test_rankings_file_holds_record_ids_in_score_order(finished: Finished) -> None:
    path = finished.rig.paths.rankings_directory / f"{finished.outcome.run_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["run_id"] == finished.outcome.run_id
    assert document["pipeline"] == "green"
    postings = document["postings"]
    assert set(postings) == set(EXPECTED_RANKINGS)
    for title, expected in EXPECTED_RANKINGS.items():
        entries = postings[title]
        assert [entry["record_id"] for entry in entries] == [rid for rid, _ in expected]
        scores = [entry["score"] for entry in entries]
        assert scores == sorted(scores, reverse=True)
        assert scores == pytest.approx([score for _, score in expected], abs=SCORE_TOLERANCE)
        assert all(set(entry) == {"record_id", "score"} for entry in entries)


def test_duplicates_are_blocked_and_eliminations_counted(finished: Finished) -> None:
    result = finished.outcome.record.result
    counts = result.counts
    assert counts.submitted == len(APPLICATIONS)
    assert counts.blocked == 1
    assert counts.ranked == EXPECTED_RANKED
    assert dict(counts.eliminated_by_reason) == {"MIN_YEARS": 1}
    assert counts.unreadable == 0
    notes = parse_notes(finished.outcome.record.measurement_notes)
    assert notes["duplicates_blocked"] == "1"
    assert notes["eliminated_by_reason"] == "MIN_YEARS:1"
    assert notes["applications"] == str(len(APPLICATIONS))


def test_cache_hits_and_llm_counters_are_filled(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    assert run.parse_cache_hits >= 1
    assert run.embed_cache_hits >= 1
    assert run.llm_calls == len(finished.llm.prompts) == EXPECTED_RANKED
    assert run.llm_prompt_tokens == EXPECTED_RANKED * PROMPT_TOKENS
    assert run.llm_completion_tokens == EXPECTED_RANKED * COMPLETION_TOKENS
    assert stage_rows(run)["llm_summary"][2] == EXPECTED_RANKED


def test_notes_record_the_measurement_conditions(finished: Finished) -> None:
    notes = parse_notes(finished.outcome.record.measurement_notes)
    for key, value in FAKE_ENVIRONMENT.items():
        assert notes[key] == value
    assert notes["tracker"] == "fixed"
    assert notes["llm_status"] == "available"
    assert notes["cuda_context"] == "held"
    assert notes["n"] == str(len(SPECS))
    assert notes["seed"] == str(SEED)
    assert json.loads(notes["scenario"]) == scenario_json(finished.plan.config)
    assert "llm_call_median_s" in notes
    assert "llm_call_p90_s" in notes


def test_gpu_cross_check_is_recorded(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    notes = parse_notes(finished.outcome.record.measurement_notes)
    stage_sum = measured_calls(run) * ENERGY_PER_CALL_WH * GPU_SHARE
    assert float(notes["gpu_stream_wh"]) == pytest.approx(FAKE_GPU_READING.energy_wh)
    assert float(notes["gpu_stage_sum_wh"]) == pytest.approx(stage_sum, abs=NOTE_TOLERANCE)
    expected_ratio = stage_sum / FAKE_GPU_READING.energy_wh
    assert float(notes["gpu_stage_sum_over_stream"]) == pytest.approx(expected_ratio, abs=NOTE_TOLERANCE)
    assert float(notes["gpu_stream_window_s"]) == pytest.approx(FAKE_GPU_READING.window_s)
    assert notes["gpu_stream_samples"] == str(FAKE_GPU_READING.sample_count)


def test_unstaged_wall_time_is_wall_minus_stage_durations(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    notes = parse_notes(finished.outcome.record.measurement_notes)
    wall = float(notes["measured_wall_s"])
    assert wall == pytest.approx(finished.outcome.timing.measured_wall_s, abs=NOTE_TOLERANCE)
    assert float(notes["unstaged_wall_s"]) == pytest.approx(wall - run.duration_s, abs=NOTE_TOLERANCE)
    assert float(notes["unstaged_share"]) == pytest.approx(
        (wall - run.duration_s) / wall, abs=NOTE_TOLERANCE
    )


def test_no_cv_data_reaches_notes_or_rankings(finished: Finished) -> None:
    run = read_runs(finished.rig)[0]
    path = finished.rig.paths.rankings_directory / f"{finished.outcome.run_id}.json"
    for text in (run.measurement_notes, path.read_text(encoding="utf-8")):
        for marker in LEAK_MARKERS:
            assert marker not in text


def test_models_load_once_and_the_run_is_wrapped(finished: Finished) -> None:
    assert finished.embedder.loads == 1
    assert finished.llm.warm_ups == 1
    assert finished.rig.cuda.acquired == finished.rig.cuda.released == 1
    assert finished.rig.sampler.started == finished.rig.sampler.stopped == 1


def test_scratch_database_is_recreated_for_every_run(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    plan = make_plan(tmp_path)
    runner, _, _ = make_runner(rig)
    first, second = runner.run(plan), runner.run(plan)
    runs = read_runs(rig)
    assert [run.id for run in runs] == [first.run_id, second.run_id]
    assert len(runs) == 2
    assert runs[0].scenario_json == runs[1].scenario_json
    assert runs[0].parse_cache_hits == runs[1].parse_cache_hits
    assert second.record.result.counts.blocked == 1
    assert runs[0].energy_wh_gross == pytest.approx(runs[1].energy_wh_gross)
    assert (rig.paths.rankings_directory / f"{second.run_id}.json").is_file()


def test_missing_idle_file_fails_fast(tmp_path: Path, test_settings: Settings) -> None:
    rig = build_rig(tmp_path, test_settings, write_idle=False)
    runner, embedder, _ = make_runner(rig)
    with pytest.raises(IdleDataError, match="bench idle"):
        runner.run(make_plan(tmp_path))
    assert embedder.loads == 0
    assert rig.cuda.acquired == rig.sampler.started == 0
    assert not rig.paths.bench_db.exists()


def test_unreachable_llm_marks_the_run_and_it_completes(
    tmp_path: Path, test_settings: Settings
) -> None:
    finished_run = execute(tmp_path, test_settings, WarmFakeLlm(unavailable=True))
    run = read_runs(finished_run.rig)[0]
    notes = parse_notes(finished_run.outcome.record.measurement_notes)
    assert notes["llm_status"] == "llm_unavailable"
    assert "llm_call_median_s" not in notes
    assert finished_run.llm.attempts == 0
    assert run.llm_calls == 0
    assert "llm_summary" not in stage_rows(run)
    assert finished_run.outcome.record.result.counts.ranked == EXPECTED_RANKED


class FailingRunner(GreenRunner):
    """A green runner whose application loop fails midway."""

    def run_applications(self, plan: RunPlan) -> WorkloadCounts:
        """Fail like a crash in the middle of the measured work."""
        raise RuntimeError("boom")


def test_sampler_and_cuda_context_are_released_on_error(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    runner = FailingRunner(rig.services, LoadableEmbedder(), WarmFakeLlm())
    with pytest.raises(RuntimeError, match="boom"):
        runner.run(make_plan(tmp_path))
    assert rig.sampler.started == 1
    assert rig.sampler.stopped == 1
    assert rig.cuda.released == 1
    assert not rig.paths.bench_db.exists()


def test_unavailable_gpu_stream_is_recorded_and_the_run_continues(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    rig.sampler.reading = UNAVAILABLE_READING
    runner, _, _ = make_runner(rig)
    outcome = runner.run(make_plan(tmp_path))
    notes = parse_notes(outcome.record.measurement_notes)
    assert notes["gpu_stream"] == "unavailable"
    assert "gpu_stage_sum_wh" in notes
    assert "gpu_stream_wh" not in notes


def test_an_unreadable_pdf_is_counted_and_skipped(
    tmp_path: Path, test_settings: Settings
) -> None:
    rig = build_rig(tmp_path, test_settings)
    plan = make_plan(tmp_path)
    plan.records["CHEF1"].pdf_path.write_bytes(b"this is not a pdf")
    runner, _, _ = make_runner(rig)
    counts = runner.run(plan).record.result.counts
    assert counts.unreadable == 1
    assert counts.ranked == EXPECTED_RANKED - 1


def test_summary_prints_the_aggregates(finished: Finished) -> None:
    text = format_run_summary(finished.outcome)
    for expected in ("Gross", "net", "Storage:", "Cache hits:", "GPU:", "Unstaged wall", "retention"):
        assert expected in text
    for marker in LEAK_MARKERS:
        assert marker not in text


def stage(energy: float, duration: float, count: int = 1, **extra: float) -> StageTotals:
    """A stage total with the given energy and duration."""
    return StageTotals(energy_wh=energy, duration_s=duration, count=count, **extra)


def test_run_totals_exclude_model_load_and_subtract_idle() -> None:
    stages = {
        MODEL_LOAD_STAGE: stage(100.0, 50.0, emissions_kg=9.0, gpu_energy_wh=90.0),
        "embed": stage(2.0, 3600.0, emissions_kg=0.001, gpu_energy_wh=1.5),
        "score": stage(1.0, 3600.0, emissions_kg=0.002, gpu_energy_wh=0.0),
    }
    totals = compute_run_totals(stages, idle_power_w=0.5)
    assert totals.energy_wh_gross == pytest.approx(3.0)
    assert totals.duration_s == pytest.approx(7200.0)
    assert totals.energy_wh_net == pytest.approx(3.0 - 0.5 * 2.0)
    assert totals.co2_g == pytest.approx(3.0)
    assert totals.gpu_stage_sum_wh == pytest.approx(1.5)


def test_subtract_totals_reports_only_what_a_later_run_added() -> None:
    before = {"embed": stage(1.0, 2.0, count=2)}
    after = {"embed": stage(4.0, 5.0, count=5), "score": stage(1.0, 1.0, count=1), "idle": stage(9.0, 9.0, count=0)}
    added = subtract_totals(after, before)
    assert added["embed"] == stage(3.0, 3.0, count=3)
    assert added["score"] == stage(1.0, 1.0, count=1)
    assert "idle" not in added


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [([], 0.9, 0.0), ([5.0], 0.9, 5.0), (list(map(float, range(1, 11))), 0.9, 9.0), ([3.0, 1.0, 2.0], 1.0, 3.0)],
)
def test_percentile_is_nearest_rank(values: list[float], fraction: float, expected: float) -> None:
    assert percentile(values, fraction) == expected


def test_baseline_exits_non_zero_until_t10(capsys: pytest.CaptureFixture[str]) -> None:
    from bench.__main__ import main

    assert main(["run", "--pipeline", "baseline", "--n", "4"]) != 0
    assert "T-10" in capsys.readouterr().err


def test_run_options_are_parsed() -> None:
    from bench.__main__ import build_parser

    arguments = build_parser().parse_args(
        ["run", "--pipeline", "green", "--n", "8", "--jobs-per-cv", "1",
         "--resubmit-rate", "0", "--seed", "5", "--repeats", "3"]
    )  # fmt: skip
    assert (arguments.n, arguments.jobs_per_cv, arguments.resubmit_rate) == (8, 1, 0.0)
    assert (arguments.seed, arguments.repeats, arguments.pipeline) == (5, 3, "green")
    assert arguments.handler is run_run



class IdleMeterStub:
    """Minimal idle meter; raises on demand to prove the context is released."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def measure_idle(self, seconds: float) -> TaskReading:
        """Return a fixed reading, or fail."""
        if self.fail:
            raise IdleDataError("measurement failed")
        return TaskReading(energy_wh=0.25, duration_s=seconds, closed=True)

    def measurement_notes(self) -> str:
        """A fixed note."""
        return "cpu_method=test"


def test_idle_is_measured_under_a_cuda_context_that_is_always_released() -> None:
    holder = FakeCudaHolder()
    record = measure_with_cuda_context(IdleMeterStub(), 30.0, holder)
    assert record.idle_power_w == pytest.approx(30.0)
    assert record.measurement_notes.endswith("cuda_context=held")
    assert holder.acquired == holder.released == 1
    with pytest.raises(IdleDataError):
        measure_with_cuda_context(IdleMeterStub(fail=True), 30.0, holder)
    assert holder.acquired == holder.released == 2


def test_power_plan_name_is_read_from_powercfg_output() -> None:
    output = "Power Scheme GUID: a1841308  (Balanced)"
    assert parse_power_plan_name(output) == "Balanced"
    assert parse_power_plan_name("no parentheses") == "unknown"


def test_ollama_is_found_in_the_per_user_install_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setattr("bench.environment.shutil.which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert find_ollama_executable() == str(executable)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing"))
    assert find_ollama_executable() is None
