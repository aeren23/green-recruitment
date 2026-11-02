"""Data carried through a benchmark run, and the arithmetic on stage totals.

`compute_run_totals` is where the SPEC 5 rule lives: gross energy, duration,
GPU share and CO2 are sums over the stages *excluding* `model_load`, and the
net energy subtracts idle power over that same duration (SPEC 7.5).
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from app.green.tracker import StageTotals
from app.models import PipelineName
from bench.dataset import ResumeRecord
from bench.scenario import ScenarioApplication, ScenarioConfig

MODEL_LOAD_STAGE = "model_load"
SECONDS_PER_HOUR = 3600.0
GRAMS_PER_KILOGRAM = 1000.0
PERCENTILE_P90 = 0.9
LLM_STATUS_AVAILABLE = "available"
LLM_STATUS_UNAVAILABLE = "llm_unavailable"


@dataclass(frozen=True)
class RunPlan:
    """What a run processes.

    Attributes:
        config: Scenario parameters (stored as `scenario_json`).
        applications: The ordered application list.
        records: Dataset records by id, for every id in `applications`.
    """

    config: ScenarioConfig
    applications: Sequence[ScenarioApplication]
    records: Mapping[str, ResumeRecord]


@dataclass(frozen=True)
class RankedEntry:
    """One ranked application: a dataset record id and its raw score.

    The score is `None` only for a baseline application whose LLM answer stayed
    malformed; such entries are ranked last.
    """

    record_id: str
    score: float | None


Rankings = dict[str, list[RankedEntry]]


@dataclass
class WorkloadCounts:
    """What happened to the scenario's applications (aggregates only).

    Attributes:
        submitted: Applications taken from the scenario.
        ranked: Applications that ended up ranked.
        blocked: Uploads refused as duplicates.
        unreadable: Uploads whose PDF could not be read.
        eliminated_by_reason: Hard-filter eliminations per reason code.
    """

    submitted: int = 0
    ranked: int = 0
    blocked: int = 0
    unreadable: int = 0
    eliminated_by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def eliminated(self) -> int:
        """Total hard-filter eliminations."""
        return sum(self.eliminated_by_reason.values())

    @property
    def confirmed(self) -> int:
        """Applications that went through confirmation (ranked or eliminated)."""
        return self.ranked + self.eliminated


@dataclass(frozen=True)
class PipelineResult:
    """What a pipeline reports after its run, before energy is attached.

    Attributes:
        counts: Outcome counts of the applications.
        storage_bytes_active: Storage before retention.
        storage_bytes_after_retention: Storage after retention and `VACUUM`.
        llm_calls: LLM generations.
        llm_prompt_tokens: Prompt tokens.
        llm_completion_tokens: Completion tokens.
        parse_cache_hits: Parse cache hits.
        embed_cache_hits: Embedding cache hits.
        llm_status: `available` or `llm_unavailable`.
        llm_call_seconds: Wall time of every LLM call, in call order.
        rankings: Final ranking per posting title, before retention.
        notes: Pipeline-specific notes for `measurement_notes`.
    """

    counts: WorkloadCounts
    storage_bytes_active: int
    storage_bytes_after_retention: int
    llm_calls: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    parse_cache_hits: int
    embed_cache_hits: int
    llm_status: str
    llm_call_seconds: Sequence[float] = ()
    rankings: Rankings = field(default_factory=dict)
    notes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunTotals:
    """Run-level sums over the measured stages (`model_load` excluded)."""

    energy_wh_gross: float
    energy_wh_net: float
    co2_g: float
    duration_s: float
    gpu_stage_sum_wh: float


@dataclass(frozen=True)
class RunRecord:
    """Everything stored for one run.

    Attributes:
        pipeline: Which pipeline ran.
        scenario: Canonical `scenario_json`.
        totals: Run-level sums.
        idle_power_w: Idle power used for the net energy.
        stages: Per-stage totals of this run, `model_load` included.
        result: The pipeline's own report.
        measurement_notes: The assembled notes string.
    """

    pipeline: PipelineName
    scenario: dict[str, Any]
    totals: RunTotals
    idle_power_w: float
    stages: Mapping[str, StageTotals]
    result: PipelineResult
    measurement_notes: str


def subtract_totals(
    after: Mapping[str, StageTotals], before: Mapping[str, StageTotals]
) -> dict[str, StageTotals]:
    """Per-stage difference of two tracker snapshots (one run among several).

    The tracker accumulates over its lifetime; a repeated run reports only what
    it added. Stages that did not run since `before` are left out.
    """
    difference: dict[str, StageTotals] = {}
    for name, current in after.items():
        previous = before.get(name, StageTotals())
        added = replace(
            current,
            energy_wh=current.energy_wh - previous.energy_wh,
            duration_s=current.duration_s - previous.duration_s,
            count=current.count - previous.count,
            emissions_kg=current.emissions_kg - previous.emissions_kg,
            gpu_energy_wh=current.gpu_energy_wh - previous.gpu_energy_wh,
        )
        if added.count > 0:
            difference[name] = added
    return difference


def compute_run_totals(
    stages: Mapping[str, StageTotals], idle_power_w: float
) -> RunTotals:
    """Sum the measured stages (without `model_load`) and derive the net energy.

    `energy_wh_net = gross - idle_power_w * duration_s / 3600`.
    """
    measured = [totals for name, totals in stages.items() if name != MODEL_LOAD_STAGE]
    gross = sum(totals.energy_wh for totals in measured)
    duration = sum(totals.duration_s for totals in measured)
    return RunTotals(
        energy_wh_gross=gross,
        energy_wh_net=gross - idle_power_w * duration / SECONDS_PER_HOUR,
        co2_g=sum(totals.emissions_kg for totals in measured) * GRAMS_PER_KILOGRAM,
        duration_s=duration,
        gpu_stage_sum_wh=sum(totals.gpu_energy_wh for totals in measured),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; `fraction` in (0, 1]. Empty input gives 0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[rank - 1]
