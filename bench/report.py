"""`bench compare`: build the benchmark report and write `report.md` / `report.json`.

Runs are grouped by canonical `scenario_json` (SC-11). Runs whose LLM was
unavailable are excluded and listed. For every group that has both pipelines
the report holds the main table (mean ± sample standard deviation, SPEC 7.8),
the per-stage breakdown, the K1 evaluation and the measurement conditions.
`report.json` is `asdict` of the same `Report` the Markdown is rendered from.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.quality_update import QualityUpdate
from bench.report_conditions import build_group_conditions
from bench.report_k1 import evaluate_k1
from bench.report_metrics import (
    RunView,
    ScenarioGroup,
    build_metric_rows,
    build_run_view,
    group_by_scenario,
    skipped_group,
    split_excluded,
)
from bench.report_model import GroupReport, Report, StageTables
from bench.report_render import render_markdown
from bench.report_stages import build_stage_rows
from bench.store import JSON_ENCODING, JSON_INDENT, StoredRun

REPORT_MARKDOWN_NAME = "report.md"
REPORT_JSON_NAME = "report.json"


@dataclass(frozen=True)
class ReportPaths:
    """Where the two report files were written."""

    markdown: Path
    json: Path


def build_group_report(group: ScenarioGroup) -> GroupReport:
    """Everything reported for one comparable scenario group."""
    rows = build_metric_rows(group.baseline, group.green)
    stages = StageTables(
        baseline=build_stage_rows(group.baseline), green=build_stage_rows(group.green)
    )
    views: list[RunView] = sorted([*group.green, *group.baseline], key=lambda v: v.run_id)
    return GroupReport(
        scenario=group.scenario,
        green_run_ids=tuple(view.run_id for view in group.green),
        baseline_run_ids=tuple(view.run_id for view in group.baseline),
        metrics=rows,
        stages=stages,
        k1=evaluate_k1({row.key: row for row in rows}),
        conditions=build_group_conditions(views, stages.baseline, stages.green),
    )


def build_report(
    runs: Sequence[StoredRun], k: int, quality: QualityUpdate | None = None
) -> Report:
    """Build the report from stored runs (their quality columns already filled)."""
    included, excluded = split_excluded([build_run_view(run) for run in runs])
    groups = group_by_scenario(included)
    return Report(
        k=k,
        groups=tuple(build_group_report(group) for group in groups if group.comparable),
        excluded_runs=tuple(excluded),
        skipped_groups=tuple(skipped_group(group) for group in groups if not group.comparable),
        quality=quality,
    )


def report_to_json(report: Report) -> str:
    """The machine-readable report: the same numbers as the Markdown."""
    return json.dumps(asdict(report), indent=JSON_INDENT, ensure_ascii=False)


def write_report(report: Report, directory: Path) -> ReportPaths:
    """Write `report.md` and `report.json` into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(directory / REPORT_MARKDOWN_NAME, directory / REPORT_JSON_NAME)
    paths.markdown.write_text(render_markdown(report), encoding=JSON_ENCODING)
    paths.json.write_text(report_to_json(report), encoding=JSON_ENCODING)
    return paths
