"""Tests for the stage tracker and the idle measurement.

The CodeCarbon layer is replaced by `FakeEmissionsTracker`, so these tests need
no GPU and never import `codecarbon`.
"""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.green import tracker as tracker_module
from app.green.tracker import (
    STAGE_NAMES,
    CodeCarbonStageTracker,
    NestedStageError,
    NoOpStageTracker,
    TaskReading,
    UnknownStageError,
    create_stage_tracker,
)
from bench.__main__ import build_parser
from bench.idle import (
    IdleDataError,
    IdlePowerRecord,
    compute_idle_power_w,
    load_idle_record,
    run_idle_measurement,
    save_idle_record,
)

FAKE_POWER_W = 3600.0
FAKE_GPU_SHARE = 0.25
FAKE_KG_CO2_PER_KWH = 0.5
SECONDS_PER_HOUR = 3600.0
WH_PER_KWH = 1000.0
SHORT_SLEEP_S = 0.05
LONG_SLEEP_S = 0.15
TIMING_TOLERANCE_S = 0.05
HARDWARE_DESCRIPTIONS = ["RAM()", "GPU() (Fake GPU)", "CPU(Windows Emi)"]
FAKE_HARDWARE = {"cpu_model": "Fake CPU", "gpu_model": "1 x Fake GPU"}
SUBPROCESS_TIMEOUT_S = 60
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
NO_CODECARBON_SCRIPT = """
import sys
from app.green.tracker import CodeCarbonStageTracker, NoOpStageTracker
noop = NoOpStageTracker()
with noop.stage("parse"):
    pass
assert noop.totals() == {}
class Fake:
    _conf = {}
    def get_detected_hardware(self):
        return {}
    def start_task(self, name):
        pass
    def stop_task(self):
        return None
tracker = CodeCarbonStageTracker("TUR", emissions_tracker=Fake())
with tracker.stage("parse"):
    pass
assert "codecarbon" not in sys.modules, "codecarbon was imported"
"""


class FakeEmissionsTracker:
    """Stands in for CodeCarbon's tracker: energy = FAKE_POWER_W times elapsed."""

    def __init__(self) -> None:
        self._conf: dict[str, Any] = {"hardware": HARDWARE_DESCRIPTIONS}
        self.started: list[str] = []
        self.stopped = 0
        self.active_since: float | None = None
        self.stop_returns_none = False

    def get_detected_hardware(self) -> dict[str, str]:
        return FAKE_HARDWARE

    def start_task(self, task_name: str) -> None:
        assert self.active_since is None, "fake: a task is already running"
        self.started.append(task_name)
        self.active_since = time.perf_counter()

    def stop_task(self) -> SimpleNamespace | None:
        assert self.active_since is not None, "fake: no task is running"
        elapsed = time.perf_counter() - self.active_since
        self.active_since = None
        self.stopped += 1
        if self.stop_returns_none:
            return None
        kilowatt_hours = FAKE_POWER_W * elapsed / SECONDS_PER_HOUR / WH_PER_KWH
        return SimpleNamespace(
            energy_consumed=kilowatt_hours,
            gpu_energy=kilowatt_hours * FAKE_GPU_SHARE,
            emissions=kilowatt_hours * FAKE_KG_CO2_PER_KWH,
        )


@pytest.fixture
def fake() -> FakeEmissionsTracker:
    return FakeEmissionsTracker()


def build_tracker(
    fake: FakeEmissionsTracker, embed_device: str | None = "cuda"
) -> CodeCarbonStageTracker:
    """Build a tracker and forget the construction-time warm-up task."""
    built = CodeCarbonStageTracker("TUR", embed_device, fake)
    fake.started.clear()
    fake.stopped = 0
    return built


@pytest.fixture
def tracker(fake: FakeEmissionsTracker) -> CodeCarbonStageTracker:
    return build_tracker(fake)


def expected_energy_wh(seconds: float) -> float:
    return FAKE_POWER_W * seconds / SECONDS_PER_HOUR


def test_stage_names_follow_spec_section_7_4() -> None:
    assert STAGE_NAMES == {
        "store_pdf", "extract", "llm_score", "rank", "hash", "parse",
        "filter", "embed", "score", "llm_summary", "retention", "model_load",
    }  # fmt: skip


def test_construction_pays_the_warm_up_without_recording_it(
    fake: FakeEmissionsTracker,
) -> None:
    built = CodeCarbonStageTracker("TUR", None, fake)
    assert fake.started == ["warm_up"]
    assert fake.active_since is None
    assert built.totals() == {}


def test_two_stages_are_reported_separately(tracker: CodeCarbonStageTracker) -> None:
    with tracker.stage("parse"):
        time.sleep(SHORT_SLEEP_S)
    with tracker.stage("embed"):
        time.sleep(LONG_SLEEP_S)
    totals = tracker.totals()
    assert set(totals) == {"parse", "embed"}
    short_run, long_run = totals["parse"], totals["embed"]
    assert short_run.duration_s == pytest.approx(SHORT_SLEEP_S, abs=TIMING_TOLERANCE_S)
    assert long_run.duration_s == pytest.approx(LONG_SLEEP_S, abs=TIMING_TOLERANCE_S)
    assert totals["embed"].energy_wh > totals["parse"].energy_wh > 0
    assert totals["embed"].energy_wh == pytest.approx(
        expected_energy_wh(totals["embed"].duration_s), rel=0.2
    )


def test_emissions_and_gpu_energy_accumulate_per_stage(
    tracker: CodeCarbonStageTracker,
) -> None:
    for _ in range(2):
        with tracker.stage("embed"):
            time.sleep(SHORT_SLEEP_S / 5)
    totals = tracker.totals()["embed"]
    assert totals.count == 2
    assert totals.gpu_energy_wh == pytest.approx(totals.energy_wh * FAKE_GPU_SHARE)
    assert totals.emissions_kg == pytest.approx(
        totals.energy_wh / WH_PER_KWH * FAKE_KG_CO2_PER_KWH
    )
    assert totals.emissions_kg > 0


def test_count_and_totals_accumulate(tracker: CodeCarbonStageTracker) -> None:
    for _ in range(3):
        with tracker.stage("hash"):
            time.sleep(SHORT_SLEEP_S / 5)
    totals = tracker.totals()["hash"]
    assert totals.count == 3
    assert totals.duration_s >= 3 * SHORT_SLEEP_S / 5


def test_totals_returns_a_copy(tracker: CodeCarbonStageTracker) -> None:
    with tracker.stage("hash"):
        pass
    tracker.totals().clear()
    assert "hash" in tracker.totals()


@pytest.mark.parametrize("implementation", [NoOpStageTracker, None])
def test_unknown_stage_raises_immediately(
    implementation: type | None, fake: FakeEmissionsTracker
) -> None:
    stage_tracker = implementation() if implementation else build_tracker(fake)
    with pytest.raises(UnknownStageError):
        stage_tracker.stage("not_a_stage")  # raised before the block is entered
    assert fake.started == []


def test_exception_inside_stage_still_closes_and_records(
    tracker: CodeCarbonStageTracker, fake: FakeEmissionsTracker
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with tracker.stage("score"):
            time.sleep(SHORT_SLEEP_S / 5)
            raise RuntimeError("boom")
    assert fake.active_since is None
    assert fake.stopped == 1
    assert tracker.totals()["score"].count == 1
    assert tracker.totals()["score"].energy_wh > 0
    with tracker.stage("score"):  # the tracker is usable again
        pass
    assert tracker.totals()["score"].count == 2


def test_nested_stage_raises_and_outer_stage_survives(
    tracker: CodeCarbonStageTracker, fake: FakeEmissionsTracker
) -> None:
    with tracker.stage("parse"):
        with pytest.raises(NestedStageError, match="parse"):
            with tracker.stage("embed"):
                pass
    assert fake.started == ["parse"]
    assert fake.stopped == 1
    assert set(tracker.totals()) == {"parse"}


def test_missing_codecarbon_result_counts_zero_energy(
    tracker: CodeCarbonStageTracker, fake: FakeEmissionsTracker
) -> None:
    fake.stop_returns_none = True
    with tracker.stage("filter"):
        pass
    assert tracker.totals()["filter"].energy_wh == 0.0
    assert tracker.totals()["filter"].count == 1


def test_noop_tracker_measures_nothing() -> None:
    noop = NoOpStageTracker()
    with noop.stage("parse"):
        time.sleep(SHORT_SLEEP_S / 5)
    assert noop.totals() == {}
    assert "GREEN_TRACKING=false" in noop.measurement_notes()


def test_noop_and_injected_tracker_never_import_codecarbon() -> None:
    result = subprocess.run(
        [sys.executable, "-c", NO_CODECARBON_SCRIPT],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("green_tracking", "expected_type"),
    [(False, NoOpStageTracker), (True, CodeCarbonStageTracker)],
)
def test_factory_follows_green_tracking_setting(
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    green_tracking: bool,
    expected_type: type,
) -> None:
    monkeypatch.setattr(
        tracker_module, "build_offline_tracker", lambda country: FakeEmissionsTracker()
    )
    settings = test_settings.model_copy(update={"green_tracking": green_tracking})
    assert type(create_stage_tracker(settings)) is expected_type


def test_measurement_notes_name_cpu_method_gpu_version_and_device(
    tracker: CodeCarbonStageTracker,
) -> None:
    notes = tracker.measurement_notes()
    assert "cpu_method=CPU(Windows Emi)" in notes
    assert "gpu=1 x Fake GPU" in notes
    assert "codecarbon=3." in notes
    assert "embed_device=cuda" in notes


def test_idle_power_arithmetic() -> None:
    assert compute_idle_power_w(energy_wh=0.5, duration_s=30.0) == pytest.approx(60.0)
    assert compute_idle_power_w(energy_wh=0.0, duration_s=10.0) == 0.0
    with pytest.raises(IdleDataError):
        compute_idle_power_w(energy_wh=1.0, duration_s=0.0)


class FakeIdleMeter:
    def measure_idle(self, seconds: float) -> TaskReading:
        return TaskReading(energy_wh=0.25, duration_s=seconds, closed=True)

    def measurement_notes(self) -> str:
        return "cpu_method=CPU(Windows Emi)"


def test_idle_measurement_builds_a_record() -> None:
    measured_at = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    record = run_idle_measurement(FakeIdleMeter(), 30.0, measured_at)
    assert record.idle_power_w == pytest.approx(30.0)
    assert record.measured_at == "2026-11-02T12:00:00+00:00"
    assert "CPU(Windows Emi)" in record.measurement_notes
    with pytest.raises(IdleDataError):
        run_idle_measurement(FakeIdleMeter(), 0.0, measured_at)


def test_idle_record_round_trips_through_json(tmp_path: Path) -> None:
    record = IdlePowerRecord(12.5, 0.1, 28.8, "2026-11-02T12:00:00+00:00", "notes")
    path = tmp_path / "bench" / "idle_power.json"
    save_idle_record(record, path)
    assert json.loads(path.read_text(encoding="utf-8"))["idle_power_w"] == 12.5
    assert load_idle_record(path) == record


def test_missing_or_broken_idle_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IdleDataError, match="bench idle"):
        load_idle_record(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(IdleDataError, match="not a valid idle record"):
        load_idle_record(broken)


def test_idle_command_is_registered() -> None:
    arguments = build_parser().parse_args(["idle", "--seconds", "5"])
    assert arguments.seconds == 5.0
    assert arguments.output == Path("data/bench/idle_power.json")
