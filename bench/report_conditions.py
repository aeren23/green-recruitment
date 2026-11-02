"""Measurement conditions and caveats of a scenario group (SPEC 7.5 and 11).

A group is a valid K1 measurement only when every run was taken under the
final conditions: power plan "Dengeli", OneDrive not running and a clean git
tree. Anything else (including an unknown value, which cannot be confirmed) is
reported as a warning per run.
"""

from collections.abc import Sequence

from bench.environment import (
    BOOLEAN_FALSE,
    GIT_DIRTY,
    GIT_REVISION,
    ONEDRIVE_RUNNING,
    POWER_PLAN,
    UNKNOWN,
)
from bench.report_metrics import RunView
from bench.report_model import ExtractComparison, GroupConditions, RunCondition, StageRow
from bench.report_stages import milliseconds_per_call

FINAL_POWER_PLAN = "Dengeli"
NOTE_GPU_RATIO = "gpu_stage_sum_over_stream"
NOTE_UNSTAGED_SHARE = "unstaged_share"
NOTE_CPU_METHOD = "cpu_method"
EXTRACT_STAGE = "extract"
WARNING_POWER_PLAN = "power plan is '{plan}', the final measurement needs '" + FINAL_POWER_PLAN + "'"
WARNING_ONEDRIVE = "OneDrive state is '{state}', it must be '" + BOOLEAN_FALSE + "' (not running)"
WARNING_GIT_DIRTY = "git tree state is '{state}', it must be '" + BOOLEAN_FALSE + "' (clean)"


def parse_float(text: str | None) -> float | None:
    """A float from a note value, `None` for a missing or non-numeric value."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def run_warnings(view: RunView) -> tuple[str, ...]:
    """Why a run is not a final-condition measurement (empty when it is)."""
    plan = view.notes.get(POWER_PLAN, UNKNOWN)
    onedrive = view.notes.get(ONEDRIVE_RUNNING, UNKNOWN)
    dirty = view.notes.get(GIT_DIRTY, UNKNOWN)
    warnings: list[str] = []
    if plan != FINAL_POWER_PLAN:
        warnings.append(WARNING_POWER_PLAN.format(plan=plan))
    if onedrive != BOOLEAN_FALSE:
        warnings.append(WARNING_ONEDRIVE.format(state=onedrive))
    if dirty != BOOLEAN_FALSE:
        warnings.append(WARNING_GIT_DIRTY.format(state=dirty))
    return tuple(warnings)


def extract_ms_per_call(view: RunView) -> float | None:
    """Milliseconds per call of the run's `extract` stage."""
    for stage in view.run.stages:
        if stage.stage == EXTRACT_STAGE:
            return milliseconds_per_call(stage)
    return None


def build_run_condition(view: RunView) -> RunCondition:
    """The conditions of one run, from its notes and stored columns."""
    notes = view.notes
    return RunCondition(
        run_id=view.run_id,
        pipeline=view.pipeline.value,
        power_plan=notes.get(POWER_PLAN, UNKNOWN),
        onedrive_running=notes.get(ONEDRIVE_RUNNING, UNKNOWN),
        git_revision=notes.get(GIT_REVISION, UNKNOWN),
        git_dirty=notes.get(GIT_DIRTY, UNKNOWN),
        idle_power_w=view.run.idle_power_w,
        gpu_stage_sum_over_stream=parse_float(notes.get(NOTE_GPU_RATIO)),
        unstaged_share=parse_float(notes.get(NOTE_UNSTAGED_SHARE)),
        extract_ms_per_call=extract_ms_per_call(view),
        warnings=run_warnings(view),
    )


def stage_ms_per_call(rows: Sequence[StageRow], stage: str) -> float | None:
    """Mean milliseconds per call of a named stage in a stage table."""
    for row in rows:
        if row.stage == stage:
            return row.ms_per_call
    return None


def compare_extract(
    baseline_rows: Sequence[StageRow], green_rows: Sequence[StageRow]
) -> ExtractComparison:
    """The `extract` per-call cost of both pipelines (D6).

    Green paying more per call than the baseline means the tracker overhead
    hurts green, so the comparison is conservative against green.
    """
    baseline = stage_ms_per_call(baseline_rows, EXTRACT_STAGE)
    green = stage_ms_per_call(green_rows, EXTRACT_STAGE)
    if baseline is None or green is None:
        return ExtractComparison(baseline, green, None, None)
    return ExtractComparison(baseline, green, green - baseline, green > baseline)


def distinct_cpu_methods(views: Sequence[RunView]) -> tuple[str, ...]:
    """The CPU measurement methods recorded in the runs' notes, in first-seen order."""
    methods = [view.notes[NOTE_CPU_METHOD] for view in views if NOTE_CPU_METHOD in view.notes]
    return tuple(dict.fromkeys(methods))


def build_group_conditions(
    views: Sequence[RunView],
    baseline_rows: Sequence[StageRow],
    green_rows: Sequence[StageRow],
) -> GroupConditions:
    """Conditions of all runs of a group and whether it is a valid K1 measurement."""
    runs = tuple(build_run_condition(view) for view in views)
    return GroupConditions(
        runs=runs,
        valid_for_k1=not any(run.warnings for run in runs),
        cpu_methods=distinct_cpu_methods(views),
        extract=compare_extract(baseline_rows, green_rows),
    )

