"""The data a benchmark report is built from, as plain dataclasses.

`report.json` is `dataclasses.asdict` of `Report`, and `report.md` is rendered
from the very same objects, so the two files always carry the same numbers.
Everything here is an aggregate, a run id, a scenario parameter or a
measurement condition; no CV text and no per-candidate data.
"""

from dataclasses import dataclass
from typing import Any

from bench.quality_update import QualityUpdate


@dataclass(frozen=True)
class Summary:
    """Mean and sample standard deviation of a metric over repeated runs.

    Attributes:
        mean: Mean of the available values, or `None` without values.
        sd: Sample standard deviation, or `None` for fewer than two values.
        n: How many values the mean covers.
    """

    mean: float | None
    sd: float | None
    n: int


@dataclass(frozen=True)
class MetricRow:
    """One row of the main table (SPEC 7.8)."""

    key: str
    baseline: Summary
    green: Summary
    difference_pct: float | None


@dataclass(frozen=True)
class StageRow:
    """Mean figures of one stage over the runs of one pipeline.

    Attributes:
        stage: Stage name (SPEC 7.4).
        energy_wh: Mean energy per run.
        duration_s: Mean duration per run.
        count: Mean number of calls per run.
        ms_per_call: Mean of the per-run milliseconds per call.
        approximate: True when the mean per-call time is below 100 ms (D1).
        counted_in_totals: False for `model_load`, which the totals exclude.
        runs: How many runs contain the stage.
    """

    stage: str
    energy_wh: float
    duration_s: float
    count: float
    ms_per_call: float | None
    approximate: bool
    counted_in_totals: bool
    runs: int


@dataclass(frozen=True)
class StageTables:
    """Per-stage breakdown of both pipelines of a group."""

    baseline: tuple[StageRow, ...]
    green: tuple[StageRow, ...]


@dataclass(frozen=True)
class CriterionResult:
    """One pre-registered K1 criterion with the exact numbers used.

    Attributes:
        key: Criterion identifier.
        rule: The condition in words.
        baseline_value: Group mean of the baseline.
        green_value: Group mean of the green pipeline.
        threshold: Value the green number is compared with, when it is not
            the baseline value itself.
        passed: True or False, or `None` when a number is missing.
    """

    key: str
    rule: str
    baseline_value: float | None
    green_value: float | None
    threshold: float | None
    passed: bool | None


@dataclass(frozen=True)
class K1Evaluation:
    """The K1 verdict of a group: every criterion and whether all passed."""

    criteria: tuple[CriterionResult, ...]
    all_passed: bool | None


@dataclass(frozen=True)
class RunCondition:
    """Measurement conditions of one run, read from its notes.

    Attributes:
        run_id: The run.
        pipeline: `green` or `baseline`.
        power_plan: Active Windows power plan at run time.
        onedrive_running: `true`, `false` or `unknown`.
        git_revision: Short git hash at run time.
        git_dirty: `true`, `false` or `unknown`.
        idle_power_w: Idle power used for the net energy.
        gpu_stage_sum_over_stream: D4 ratio, or `None` when unavailable.
        unstaged_share: Share of wall time no stage measured.
        extract_ms_per_call: Mean `extract` milliseconds per call.
        warnings: Reasons this run is not final-condition.
    """

    run_id: int
    pipeline: str
    power_plan: str
    onedrive_running: str
    git_revision: str
    git_dirty: str
    idle_power_w: float
    gpu_stage_sum_over_stream: float | None
    unstaged_share: float | None
    extract_ms_per_call: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExtractComparison:
    """The per-call `extract` cost of both pipelines (decision D6).

    Attributes:
        baseline_ms: Mean milliseconds per call, baseline.
        green_ms: Mean milliseconds per call, green.
        difference_ms: green minus baseline.
        penalizes_green: True when green pays more per call, so the
            comparison is conservative against green; `None` without data.
    """

    baseline_ms: float | None
    green_ms: float | None
    difference_ms: float | None
    penalizes_green: bool | None


@dataclass(frozen=True)
class GroupConditions:
    """Conditions and caveats of one scenario group."""

    runs: tuple[RunCondition, ...]
    valid_for_k1: bool
    cpu_methods: tuple[str, ...]
    extract: ExtractComparison


@dataclass(frozen=True)
class GroupReport:
    """Everything reported for one scenario group that has both pipelines."""

    scenario: dict[str, Any]
    green_run_ids: tuple[int, ...]
    baseline_run_ids: tuple[int, ...]
    metrics: tuple[MetricRow, ...]
    stages: StageTables
    k1: K1Evaluation
    conditions: GroupConditions


@dataclass(frozen=True)
class ExcludedRun:
    """A run left out of the comparison and why."""

    run_id: int
    pipeline: str
    reason: str


@dataclass(frozen=True)
class SkippedGroup:
    """A scenario group that cannot be compared (one pipeline is missing)."""

    scenario: dict[str, Any]
    green_run_ids: tuple[int, ...]
    baseline_run_ids: tuple[int, ...]


@dataclass(frozen=True)
class Report:
    """The whole report.

    Attributes:
        k: The quality cut-off.
        groups: One entry per comparable scenario group.
        excluded_runs: Runs excluded from every group.
        skipped_groups: Groups with only one pipeline.
        quality: What the quality update did, when it ran just before.
    """

    k: int
    groups: tuple[GroupReport, ...]
    excluded_runs: tuple[ExcludedRun, ...]
    skipped_groups: tuple[SkippedGroup, ...]
    quality: QualityUpdate | None
