"""The per-stage energy breakdown of one pipeline (SPEC 7.8, decision D1).

Stages that average under 100 ms per call are flagged approximate: the energy
counters update in coarse steps, so a single short stage is not reliable
(measurement gate D1). `model_load` is listed but marked as outside the totals.
"""

import statistics
from collections.abc import Sequence

from bench.report_metrics import RunView
from bench.report_model import StageRow
from bench.results import MODEL_LOAD_STAGE
from bench.store import StoredStage

APPROXIMATE_BELOW_MS = 100.0
MILLISECONDS_PER_SECOND = 1000.0


def milliseconds_per_call(stage: StoredStage) -> float | None:
    """Milliseconds per call of one stored stage, `None` for a zero count."""
    if stage.count <= 0:
        return None
    return stage.duration_s / stage.count * MILLISECONDS_PER_SECOND


def collect_stages(views: Sequence[RunView]) -> dict[str, list[StoredStage]]:
    """The stored stage rows of all runs, by stage name."""
    collected: dict[str, list[StoredStage]] = {}
    for view in views:
        for stage in view.run.stages:
            collected.setdefault(stage.stage, []).append(stage)
    return collected


def mean_ms_per_call(stages: Sequence[StoredStage]) -> float | None:
    """Mean over runs of the per-run milliseconds per call."""
    values = [ms for ms in map(milliseconds_per_call, stages) if ms is not None]
    return statistics.fmean(values) if values else None


def build_stage_row(name: str, stages: Sequence[StoredStage]) -> StageRow:
    """Mean figures of one stage over the runs that contain it."""
    ms_per_call = mean_ms_per_call(stages)
    return StageRow(
        stage=name,
        energy_wh=statistics.fmean(stage.energy_wh for stage in stages),
        duration_s=statistics.fmean(stage.duration_s for stage in stages),
        count=statistics.fmean(stage.count for stage in stages),
        ms_per_call=ms_per_call,
        approximate=ms_per_call is not None and ms_per_call < APPROXIMATE_BELOW_MS,
        counted_in_totals=name != MODEL_LOAD_STAGE,
        runs=len(stages),
    )


def build_stage_rows(views: Sequence[RunView]) -> tuple[StageRow, ...]:
    """The stage table of one pipeline, stages in name order."""
    collected = collect_stages(views)
    return tuple(build_stage_row(name, collected[name]) for name in sorted(collected))
