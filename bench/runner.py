"""`BenchmarkRunner`: the template method shared by the green and baseline runs.

A run has a fixed shape, owned by `run()`:

1. fail fast if the idle power file is missing (`python -m bench idle`);
2. hold a CUDA context for the whole run (decision D2);
3. `prepare()`: load models (stage `model_load`, excluded from gross energy and
   duration) and warm the LLM up. Nothing here is part of the measured phase;
4. start the GPU sampler, then `run_applications()` and `finalize()`;
5. always stop the sampler and release the context, even on error;
6. compute gross, net and CO2 from the stage totals, assemble the notes and
   store the rows and the rankings.

Subclasses implement the three pipeline-specific steps only.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from app.config import Settings
from app.green.tracker import StageTotals, StageTracker
from app.models import PipelineName
from bench.cuda_context import CudaContextHolder
from bench.environment import EnvironmentProbe
from bench.gpu_sampler import GpuPowerSampler, GpuStreamReading
from bench.idle import load_idle_record
from bench.notes import NotesInputs, TimingFacts, build_measurement_notes
from bench.results import (
    PipelineResult,
    RunPlan,
    RunRecord,
    RunTotals,
    WorkloadCounts,
    compute_run_totals,
    subtract_totals,
)
from bench.scenario import scenario_json
from bench.store import BenchmarkStore, BenchPaths


@dataclass(frozen=True)
class RunnerServices:
    """Everything a runner depends on, injected by the caller.

    Attributes:
        settings: Application settings (`.env`).
        tracker: Stage tracker shared by all runs of one process.
        cuda_holder: Holds the CUDA context (a no-op in tests).
        gpu_sampler: Whole-run GPU power stream (a fake in tests).
        environment: Git, power plan and OneDrive facts.
        paths: Locations under `data/bench`.
        clock: Returns the current UTC time for the pipeline's `now`.
    """

    settings: Settings
    tracker: StageTracker
    cuda_holder: CudaContextHolder
    gpu_sampler: GpuPowerSampler
    environment: EnvironmentProbe
    paths: BenchPaths
    clock: Callable[[], datetime]


@dataclass(frozen=True)
class MeasuredPhase:
    """What the measured phase produced."""

    result: PipelineResult
    wall_s: float
    gpu: GpuStreamReading


@dataclass(frozen=True)
class Measurements:
    """The measured phase together with the stage totals derived from it."""

    phase: MeasuredPhase
    stages: Mapping[str, StageTotals]
    totals: RunTotals
    idle_power_w: float
    cuda_held: bool


@dataclass(frozen=True)
class RunOutcome:
    """What `BenchmarkRunner.run` returns.

    Attributes:
        run_id: Id of the stored `BenchmarkRun`.
        record: Everything that was stored.
        gpu: The whole-run GPU stream reading.
        timing: Wall-clock facts of the measured phase.
        cuda_context_held: Whether a CUDA context was held.
    """

    run_id: int
    record: RunRecord
    gpu: GpuStreamReading
    timing: TimingFacts
    cuda_context_held: bool


def read_cv_bytes(plan: RunPlan, record_id: str) -> bytes:
    """Read one CV's PDF bytes from the dataset (harness work, outside any stage)."""
    return plan.records[record_id].pdf_path.read_bytes()


class BenchmarkRunner(ABC):
    """Template method for one benchmark run.

    Args:
        services: Injected dependencies.
        store: Where results are written; defaults to the paths in `services`.
    """

    def __init__(
        self, services: RunnerServices, store: BenchmarkStore | None = None
    ) -> None:
        self._services = services
        self._store = store or BenchmarkStore(services.paths)

    @property
    @abstractmethod
    def pipeline(self) -> PipelineName:
        """Which pipeline this runner measures."""

    @abstractmethod
    def prepare(self, plan: RunPlan) -> None:
        """Set up storage, load the models (stage `model_load`) and warm the LLM up."""

    @abstractmethod
    def run_applications(self, plan: RunPlan) -> WorkloadCounts:
        """Process every application of the plan; this is the measured work."""

    @abstractmethod
    def finalize(self, plan: RunPlan, counts: WorkloadCounts) -> PipelineResult:
        """Capture rankings, run retention and report the pipeline's numbers."""

    def close(self) -> None:
        """Release resources opened by `prepare`; called even after an error."""

    def run(self, plan: RunPlan) -> RunOutcome:
        """Run the plan once and store the result.

        Raises:
            IdleDataError: If the idle power file is missing or broken.
        """
        idle = load_idle_record(self._services.paths.idle_power)
        before = self._services.tracker.totals()
        cuda_held = self._services.cuda_holder.acquire()
        try:
            phase = self._measured_phase(plan)
        finally:
            self._services.cuda_holder.release()
            self.close()
        stages = subtract_totals(self._services.tracker.totals(), before)
        totals = compute_run_totals(stages, idle.idle_power_w)
        measurements = Measurements(phase, stages, totals, idle.idle_power_w, cuda_held)
        record = self._build_record(plan, measurements)
        run_id = self._store.save(record)
        timing = TimingFacts(phase.wall_s, totals.duration_s)
        return RunOutcome(run_id, record, phase.gpu, timing, cuda_held)

    def _measured_phase(self, plan: RunPlan) -> MeasuredPhase:
        """Prepare, then run and finalize under the GPU sampler, which always stops."""
        self.prepare(plan)
        sampler = self._services.gpu_sampler
        started = time.perf_counter()
        try:
            sampler.start()
            counts = self.run_applications(plan)
            result = self.finalize(plan, counts)
        finally:
            wall_s = time.perf_counter() - started
            gpu = sampler.stop()
        return MeasuredPhase(result, wall_s, gpu)

    def _build_record(self, plan: RunPlan, measured: "Measurements") -> RunRecord:
        """Assemble the stored record, including the notes."""
        phase, totals = measured.phase, measured.totals
        notes = build_measurement_notes(
            NotesInputs(
                tracker_notes=self._services.tracker.measurement_notes(),
                environment=self._services.environment.describe(),
                cuda_context_held=measured.cuda_held,
                config=plan.config,
                gpu=phase.gpu,
                gpu_stage_sum_wh=totals.gpu_stage_sum_wh,
                timing=TimingFacts(phase.wall_s, totals.duration_s),
                result=phase.result,
            )
        )
        return RunRecord(
            pipeline=self.pipeline,
            scenario=scenario_json(plan.config),
            totals=totals,
            idle_power_w=measured.idle_power_w,
            stages=measured.stages,
            result=phase.result,
            measurement_notes=notes,
        )
