"""Test doubles and builders for the benchmark runner tests.

Nothing here needs a GPU, Ollama, CodeCarbon or the real dataset:

* `build_dataset_tree` writes a tiny Kaggle-shaped tree of synthetic PDFs;
* `FixedCostTracker` charges a known energy, duration, GPU share and CO2 per
  stage call, so the run arithmetic can be checked exactly;
* `FakeGpuSampler`, `FakeCudaHolder` and `FakeEnvironment` replace the
  machine-facing parts of a run.
"""

import csv
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pipeline_support import ControlledEmbedder, FakeLlm, build_cv_pdf

from app.config import Settings
from app.green.tracker import StageTotals, require_known_stage
from app.llm.ollama import LlmUnavailableError
from bench.dataset import (
    COLUMN_CATEGORY,
    COLUMN_ID,
    COLUMN_RESUME_HTML,
    COLUMN_RESUME_TEXT,
    PDF_DIRECTORY_RELATIVE_PATH,
    RESUME_CSV_RELATIVE_PATH,
    ResumeRecord,
    iter_resume_records,
)
from bench.gpu_sampler import GpuStreamReading
from bench.idle import IdlePowerRecord, save_idle_record
from bench.results import MODEL_LOAD_STAGE
from bench.runner import RunnerServices
from bench.store import BenchPaths

CATEGORY_IT = "INFORMATION-TECHNOLOGY"
CATEGORY_ACCOUNTANT = "ACCOUNTANT"
CATEGORY_FINANCE = "FINANCE"
CATEGORY_HR = "HR"
CATEGORY_DISTRACTOR = "CHEF"
DEFAULT_CV_YEARS = (2010, 2018)
ONE_YEAR_CV = (2017, 2018)
PLACEHOLDER_TEXT = "x"
CSV_ENCODING = "utf-8"

IDLE_POWER_W = 36.0
FIXED_NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ENERGY_PER_CALL_WH = 0.01
MODEL_LOAD_ENERGY_WH = 5.0
DURATION_PER_CALL_S = 0.002
GPU_SHARE = 0.5
KG_PER_CALL = 1e-6
FAKE_GPU_READING = GpuStreamReading(
    available=True, energy_wh=0.2, window_s=30.0, sample_count=61, malformed_lines=1
)
FAKE_ENVIRONMENT = {
    "git": "abc1234",
    "git_dirty": "true",
    "power_plan": "Test plan",
    "onedrive_running": "false",
}
FAKE_OLLAMA_PS = "NAME ID SIZE PROCESSOR\nfake 0 1 GB 100% GPU"
FAKE_TRACKER_NOTES = "tracker=fixed"
NOTES_SEPARATOR = "; "


@dataclass(frozen=True)
class CvRecordSpec:
    """One synthetic dataset record.

    Attributes:
        record_id: Dataset id (also the PDF file name).
        category: Dataset category folder.
        quality: Similarity percentage the fake embedder gives the CV.
        years: Start and end year of the single experience entry.
    """

    record_id: str
    category: str
    quality: int
    years: tuple[int, int] = DEFAULT_CV_YEARS


def build_dataset_tree(root: Path, specs: Sequence[CvRecordSpec]) -> list[ResumeRecord]:
    """Write `Resume/Resume.csv` and one synthetic PDF per spec; return the records."""
    csv_path = root / RESUME_CSV_RELATIVE_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [COLUMN_ID, COLUMN_RESUME_TEXT, COLUMN_RESUME_HTML, COLUMN_CATEGORY]
    with csv_path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for spec in specs:
            writer.writerow([spec.record_id, PLACEHOLDER_TEXT, PLACEHOLDER_TEXT, spec.category])
            write_cv_pdf(root, spec)
    return list(iter_resume_records(root))


def write_cv_pdf(root: Path, spec: CvRecordSpec) -> None:
    """Write the synthetic CV of a spec under `data/data/<category>/`."""
    directory = root / PDF_DIRECTORY_RELATIVE_PATH / spec.category
    directory.mkdir(parents=True, exist_ok=True)
    pdf = build_cv_pdf(spec.record_id, spec.quality, spec.years)
    (directory / f"{spec.record_id}.pdf").write_bytes(pdf)


def touch_records(root: Path, counts: dict[str, int]) -> list[ResumeRecord]:
    """Records with empty PDF files, enough for sampling tests (`count` per category)."""
    records: list[ResumeRecord] = []
    for category, count in counts.items():
        directory = root / category
        directory.mkdir(parents=True, exist_ok=True)
        for number in range(count):
            record_id = f"{category[:3]}{number:03d}"
            path = directory / f"{record_id}.pdf"
            path.write_bytes(b"")
            records.append(ResumeRecord(record_id, category, path, PLACEHOLDER_TEXT))
    return records


class LoadableEmbedder(ControlledEmbedder):
    """`ControlledEmbedder` with the explicit `load()` the runner calls once."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load(self) -> None:
        """Count the load."""
        self.loads += 1


class WarmFakeLlm(FakeLlm):
    """`FakeLlm` that can also be warmed up, or fail to."""

    def __init__(self, unavailable: bool = False) -> None:
        super().__init__(unavailable=unavailable)
        self.warm_ups = 0

    def warm_up(self) -> None:
        """Count the warm-up; raise when Ollama is switched off."""
        self.warm_ups += 1
        if self.unavailable:
            raise LlmUnavailableError("Ollama is unavailable: test.")


class FixedCostTracker:
    """A `StageTracker` that charges fixed amounts per stage call.

    `model_load` costs far more than any other stage, so a mistaken inclusion
    in the gross total cannot go unnoticed.
    """

    def __init__(self) -> None:
        self._totals: dict[str, StageTotals] = {}

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Validate the name; the cost is added when the stage ends."""
        require_known_stage(name)
        return self._charged(name)

    @contextmanager
    def _charged(self, name: str) -> Iterator[None]:
        """Run the stage, then add one call's fixed cost."""
        try:
            yield
        finally:
            self._add(name)

    def _add(self, name: str) -> None:
        """Add one call's cost to the stage totals."""
        energy = MODEL_LOAD_ENERGY_WH if name == MODEL_LOAD_STAGE else ENERGY_PER_CALL_WH
        previous = self._totals.get(name, StageTotals())
        self._totals[name] = StageTotals(
            energy_wh=previous.energy_wh + energy,
            duration_s=previous.duration_s + DURATION_PER_CALL_S,
            count=previous.count + 1,
            emissions_kg=previous.emissions_kg + KG_PER_CALL,
            gpu_energy_wh=previous.gpu_energy_wh + energy * GPU_SHARE,
        )

    def totals(self) -> dict[str, StageTotals]:
        """A copy of the accumulated totals."""
        return dict(self._totals)

    def measurement_notes(self) -> str:
        """A fixed tracker description."""
        return FAKE_TRACKER_NOTES


class FakeGpuSampler:
    """A `GpuPowerSampler` that records its lifecycle and returns a fixed reading."""

    def __init__(self, reading: GpuStreamReading = FAKE_GPU_READING) -> None:
        self.reading = reading
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        """Count the start."""
        self.started += 1

    def stop(self) -> GpuStreamReading:
        """Count the stop and return the fixed reading."""
        self.stopped += 1
        return self.reading


class FakeCudaHolder:
    """A `CudaContextHolder` that records acquire and release."""

    def __init__(self) -> None:
        self.acquired = 0
        self.released = 0

    def acquire(self) -> bool:
        """Count the acquire; report a held context."""
        self.acquired += 1
        return True

    def release(self) -> None:
        """Count the release."""
        self.released += 1


class FakeEnvironment:
    """An `EnvironmentProbe` with fixed answers."""

    def describe(self) -> dict[str, str]:
        """Fixed git, power plan and OneDrive facts."""
        return dict(FAKE_ENVIRONMENT)

    def ollama_ps(self) -> str:
        """A fixed `ollama ps` table."""
        return FAKE_OLLAMA_PS


@dataclass
class Rig:
    """Everything a runner test needs, built on a temp directory."""

    services: RunnerServices
    tracker: FixedCostTracker
    sampler: FakeGpuSampler
    cuda: FakeCudaHolder
    paths: BenchPaths


def build_rig(
    directory: Path, settings: Settings, write_idle: bool = True
) -> Rig:
    """Runner services on a temp bench directory, with an idle record unless disabled."""
    paths = BenchPaths(directory / "bench")
    if write_idle:
        record = IdlePowerRecord(IDLE_POWER_W, 0.1, 10.0, FIXED_NOW.isoformat(), "idle")
        save_idle_record(record, paths.idle_power)
    tracker, sampler, cuda = FixedCostTracker(), FakeGpuSampler(), FakeCudaHolder()
    services = RunnerServices(
        settings=settings,
        tracker=tracker,
        cuda_holder=cuda,
        gpu_sampler=sampler,
        environment=FakeEnvironment(),
        paths=paths,
        clock=lambda: FIXED_NOW,
    )
    return Rig(services, tracker, sampler, cuda, paths)


def parse_notes(notes: str) -> dict[str, str]:
    """Split `key=value; key=value` notes into a dictionary (first `=` splits)."""
    pairs = (item.split("=", 1) for item in notes.split(NOTES_SEPARATOR) if "=" in item)
    return {key: value for key, value in pairs}
