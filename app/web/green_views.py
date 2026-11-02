"""View data of the green panel: counters, the benchmark summary and page weight.

The numbers come from the existing modules (`app.green.counters`, the `bench`
report code, `app.green.pageweight`); this module only labels and formats them.
"""

from dataclasses import dataclass

from app.green.counters import CounterName, CounterSnapshot
from app.green.pageweight import PageWeightRow
from app.web import messages
from bench.report_metrics import METRICS_BY_KEY
from bench.report_model import GroupReport
from bench.report_render import format_difference, format_summary

COMPARISON_METRIC_KEYS = (
    "energy_wh_net_per_application",
    "sci_g_per_application",
    "duration_s",
    "llm_calls",
    "storage_bytes_after_retention",
    "precision_at_k",
    "overlap_at_k",
)
BYTES_PER_KILOBYTE = 1024
DECIMAL_POINT = "."
DECIMAL_COMMA = ","
NO_BYTES_TEXT = "0"


@dataclass(frozen=True)
class CounterLine:
    """One live counter: a Turkish label and the value as text."""

    label: str
    value: str


@dataclass(frozen=True)
class ComparisonLine:
    """One row of the compact comparison table."""

    label: str
    baseline: str
    green: str
    difference: str


@dataclass(frozen=True)
class ComparisonView:
    """The latest benchmark group, ready to render.

    Attributes:
        lines: The compact metric table.
        k1_overall: Turkish verdict of the whole K1 evaluation.
        k1_criteria: `(criterion key, Turkish verdict)` pairs.
        warnings: Turkish warnings; empty when the runs are final-condition.
        run_ids: `(baseline ids, green ids)` of the runs behind the table.
    """

    lines: tuple[ComparisonLine, ...]
    k1_overall: str
    k1_criteria: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    run_ids: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class PageWeightLine:
    """One path template with its count and average transferred size."""

    path: str
    response_count: int
    average: str


def format_counter_value(name: CounterName, value: int) -> str:
    """A counter as text: a size for the PDF bytes, a plain number otherwise."""
    if name is CounterName.PDF_BYTES_NOT_STORED:
        return messages.format_size(value) if value else NO_BYTES_TEXT
    return str(value)


def build_counter_lines(
    snapshot: CounterSnapshot, llm_calls_avoided: int
) -> tuple[CounterLine, ...]:
    """The live counters with Turkish labels; `llm_calls_avoided` is derived (SC-14)."""
    values = {name: getattr(snapshot, name.value) for name in CounterName}
    values[CounterName.LLM_CALLS_AVOIDED] = llm_calls_avoided
    return tuple(
        CounterLine(
            messages.GREEN_COUNTER_LABELS[name.value], format_counter_value(name, value)
        )
        for name, value in values.items()
    )


def build_comparison_view(group: GroupReport) -> ComparisonView:
    """The compact comparison of a group: metric, baseline, green, difference %."""
    rows = {row.key: row for row in group.metrics}
    lines = tuple(
        ComparisonLine(
            label=messages.GREEN_METRIC_LABELS[key],
            baseline=format_summary(rows[key].baseline, METRICS_BY_KEY[key].decimals),
            green=format_summary(rows[key].green, METRICS_BY_KEY[key].decimals),
            difference=format_difference(rows[key].difference_pct),
        )
        for key in COMPARISON_METRIC_KEYS
    )
    warnings = () if group.conditions.valid_for_k1 else (messages.K1_NOT_FINAL_WARNING,)
    return ComparisonView(
        lines=lines,
        k1_overall=messages.K1_VERDICT_LABELS[group.k1.all_passed],
        k1_criteria=tuple(
            (criterion.key, messages.K1_VERDICT_LABELS[criterion.passed])
            for criterion in group.k1.criteria
        ),
        warnings=warnings,
        run_ids=(group.baseline_run_ids, group.green_run_ids),
    )


def format_average_kilobytes(average_bytes: float) -> str:
    """An average size in kilobytes with one decimal and a Turkish decimal comma."""
    kilobytes = f"{average_bytes / BYTES_PER_KILOBYTE:.1f}"
    return messages.PAGE_WEIGHT_AVERAGE_KB.format(
        value=kilobytes.replace(DECIMAL_POINT, DECIMAL_COMMA)
    )


def build_page_weight_lines(rows: list[PageWeightRow]) -> tuple[PageWeightLine, ...]:
    """Average transferred size per path template, in kilobytes."""
    return tuple(
        PageWeightLine(row.path, row.response_count, format_average_kilobytes(row.average_bytes))
        for row in rows
    )
