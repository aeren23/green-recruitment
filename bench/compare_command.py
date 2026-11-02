"""`python -m bench compare`: fill the quality metrics, then write the report.

The dataset is read only for the record id to category lookup that
`precision@k` needs. The report files go next to `bench.db`.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from bench.dataset import DEFAULT_DATASET_ROOT, DatasetStructureError, iter_resume_records
from bench.quality_update import build_quality_context, update_quality_metrics
from bench.report import ReportPaths, build_report, write_report
from bench.report_model import GroupReport, Report
from bench.report_render import format_difference, format_number, verdict_label
from bench.store import DEFAULT_BENCH_DIRECTORY, BenchmarkStore, BenchPaths

COMPARE_COMMAND = "compare"
COMPARE_HELP = "Fill precision@k and overlap@k, then write data/bench/report.md and report.json."
HANDLER_DESTINATION = "handler"
EXIT_OK = 0
EXIT_ERROR = 1
ERROR_PREFIX = "Compare error: "
NO_RUNS_MESSAGE = "no benchmark runs are stored; run `bench run` first"
NET_ENERGY_KEY = "energy_wh_net_per_application"
NET_ENERGY_DECIMALS = 5
WARNING_LINE = "  WARNING: not a valid K1 measurement (conditions are not final)"


def register_compare(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the `compare` subcommand to the parser."""
    parser = subparsers.add_parser(COMPARE_COMMAND, help=COMPARE_HELP)
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIRECTORY)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.set_defaults(**{HANDLER_DESTINATION: run_compare})


def format_group_summary(index: int, group: GroupReport) -> list[str]:
    """The console lines of one group: net energy per application and the K1 verdicts."""
    energy = next(row for row in group.metrics if row.key == NET_ENERGY_KEY)
    lines = [
        f"Group {index}: green runs {list(group.green_run_ids)}, "
        f"baseline runs {list(group.baseline_run_ids)}",
        f"  {NET_ENERGY_KEY} (Wh): baseline "
        f"{format_number(energy.baseline.mean, NET_ENERGY_DECIMALS)}, green "
        f"{format_number(energy.green.mean, NET_ENERGY_DECIMALS)}, difference "
        f"{format_difference(energy.difference_pct)}",
    ]
    lines.extend(
        f"  K1 {criterion.key}: {verdict_label(criterion.passed)}"
        for criterion in group.k1.criteria
    )
    lines.append(f"  K1 overall: {verdict_label(group.k1.all_passed)}")
    if not group.conditions.valid_for_k1:
        lines.append(WARNING_LINE)
    return lines


def format_console_summary(report: Report, paths: ReportPaths) -> str:
    """A short console summary of the report."""
    lines = [
        f"Report written: {paths.markdown} and {paths.json}",
        f"Groups compared: {len(report.groups)}; excluded runs: {len(report.excluded_runs)}; "
        f"groups without both pipelines: {len(report.skipped_groups)}",
    ]
    for index, group in enumerate(report.groups, start=1):
        lines.extend(format_group_summary(index, group))
    return "\n".join(lines)


@dataclass(frozen=True)
class CompareRequest:
    """What one `compare` run needs: where the files are and the cut-off."""

    paths: BenchPaths
    dataset_root: Path
    quality_k: int


def execute_compare(request: CompareRequest) -> int:
    """Update the quality metrics, build and write the report, print the summary."""
    store = BenchmarkStore(request.paths)
    try:
        if not store.load_runs():
            raise ValueError(NO_RUNS_MESSAGE)
        records = iter_resume_records(request.dataset_root)
        context = build_quality_context(records, request.quality_k)
        quality = update_quality_metrics(store, context)
        report = build_report(store.load_runs(), request.quality_k, quality)
        report_paths = write_report(report, request.paths.directory)
    except (DatasetStructureError, ValueError) as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
    print(format_console_summary(report, report_paths))
    return EXIT_OK


def run_compare(arguments: argparse.Namespace) -> int:
    """Handle `compare` with the quality cut-off from the settings."""
    request = CompareRequest(
        paths=BenchPaths(arguments.bench_dir),
        dataset_root=arguments.root,
        quality_k=Settings().quality_k,
    )
    return execute_compare(request)
