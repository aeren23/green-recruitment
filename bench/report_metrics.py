"""Run views, the metric registry and the statistics of the main table (SPEC 7.8).

A `RunView` is a stored run plus its parsed `measurement_notes`. The registry
`METRICS` lists every row of the main table once, with the function that reads
it from a run; per-application values divide by the `applications` note and
the SCI value is `co2_g / applications` (M, the embodied carbon, is excluded).
"""

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.models import PipelineName
from bench.notes import parse_measurement_notes
from bench.report_model import ExcludedRun, MetricRow, SkippedGroup, Summary
from bench.results import LLM_STATUS_UNAVAILABLE
from bench.scenario import scenario_key
from bench.store import StoredRun

NOTE_APPLICATIONS = "applications"
NOTE_LLM_STATUS = "llm_status"
PERCENT = 100.0
SAMPLE_SD_MIN_VALUES = 2
EXCLUSION_REASON = "llm_status=llm_unavailable (T9-4)"


@dataclass(frozen=True)
class RunView:
    """A stored run with its notes parsed into `key -> value`."""

    run: StoredRun
    notes: Mapping[str, str]

    @property
    def run_id(self) -> int:
        """The run id."""
        return self.run.run_id

    @property
    def pipeline(self) -> PipelineName:
        """Which pipeline ran."""
        return self.run.pipeline

    @property
    def applications(self) -> int | None:
        """Applications submitted in the run (the `applications` note), if valid."""
        text = self.notes.get(NOTE_APPLICATIONS, "")
        return int(text) if text.isdigit() and int(text) > 0 else None


def build_run_view(run: StoredRun) -> RunView:
    """Parse the notes of a stored run."""
    return RunView(run, parse_measurement_notes(run.measurement_notes))


def per_application(value: float, view: RunView) -> float | None:
    """`value` divided by the run's application count, `None` without a count."""
    applications = view.applications
    return value / applications if applications else None


@dataclass(frozen=True)
class MetricSpec:
    """One row of the main table.

    Attributes:
        key: Metric identifier (English).
        unit: Unit shown next to the key.
        decimals: Decimal places in the Markdown table.
        read: Reads the metric from a run; `None` when it is not available.
    """

    key: str
    unit: str
    decimals: int
    read: Callable[[RunView], float | None]


def stored_attribute(name: str) -> Callable[[RunView], float | None]:
    """A reader for a column of the stored run."""

    def read(view: RunView) -> float | None:
        value = getattr(view.run, name)
        return None if value is None else float(value)

    return read


def applications_reader(view: RunView) -> float | None:
    """Applications submitted in the run."""
    return view.applications


def per_application_reader(name: str) -> Callable[[RunView], float | None]:
    """A reader for a stored column divided by the application count."""

    def read(view: RunView) -> float | None:
        return per_application(float(getattr(view.run, name)), view)

    return read


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("applications", "count", 1, applications_reader),
    MetricSpec("energy_wh_gross", "Wh", 4, stored_attribute("energy_wh_gross")),
    MetricSpec("energy_wh_net", "Wh", 4, stored_attribute("energy_wh_net")),
    MetricSpec("co2_g", "g", 4, stored_attribute("co2_g")),
    MetricSpec("duration_s", "s", 2, stored_attribute("duration_s")),
    MetricSpec(
        "energy_wh_gross_per_application", "Wh", 5, per_application_reader("energy_wh_gross")
    ),
    MetricSpec(
        "energy_wh_net_per_application", "Wh", 5, per_application_reader("energy_wh_net")
    ),
    MetricSpec("sci_g_per_application", "g, M excluded", 5, per_application_reader("co2_g")),
    MetricSpec("storage_bytes_active", "B", 0, stored_attribute("storage_bytes_active")),
    MetricSpec(
        "storage_bytes_after_retention", "B", 0, stored_attribute("storage_bytes_after_retention")
    ),
    MetricSpec("llm_calls", "count", 1, stored_attribute("llm_calls")),
    MetricSpec("llm_prompt_tokens", "count", 1, stored_attribute("llm_prompt_tokens")),
    MetricSpec("llm_completion_tokens", "count", 1, stored_attribute("llm_completion_tokens")),
    MetricSpec("parse_cache_hits", "count", 1, stored_attribute("parse_cache_hits")),
    MetricSpec("embed_cache_hits", "count", 1, stored_attribute("embed_cache_hits")),
    MetricSpec("precision_at_k", "share", 3, stored_attribute("precision_at_k")),
    MetricSpec("overlap_at_k", "share", 3, stored_attribute("overlap_at_k")),
)
METRICS_BY_KEY: Mapping[str, MetricSpec] = {spec.key: spec for spec in METRICS}


def summarize(values: Sequence[float | None]) -> Summary:
    """Mean and sample standard deviation of the available values.

    The deviation is `None` for fewer than two values (shown as "± n/a").
    """
    present = [value for value in values if value is not None]
    if not present:
        return Summary(mean=None, sd=None, n=0)
    spread = statistics.stdev(present) if len(present) >= SAMPLE_SD_MIN_VALUES else None
    return Summary(mean=statistics.fmean(present), sd=spread, n=len(present))


def difference_pct(baseline: float | None, green: float | None) -> float | None:
    """`(green - baseline) / baseline * 100`, `None` without numbers or a zero baseline."""
    if baseline is None or green is None or baseline == 0:
        return None
    return (green - baseline) / baseline * PERCENT


def build_metric_row(
    spec: MetricSpec, baseline: Sequence[RunView], green: Sequence[RunView]
) -> MetricRow:
    """Summaries of both pipelines for one metric, and their difference."""
    baseline_summary = summarize([spec.read(view) for view in baseline])
    green_summary = summarize([spec.read(view) for view in green])
    return MetricRow(
        key=spec.key,
        baseline=baseline_summary,
        green=green_summary,
        difference_pct=difference_pct(baseline_summary.mean, green_summary.mean),
    )


def build_metric_rows(
    baseline: Sequence[RunView], green: Sequence[RunView]
) -> tuple[MetricRow, ...]:
    """The whole main table of one group."""
    return tuple(build_metric_row(spec, baseline, green) for spec in METRICS)


@dataclass(frozen=True)
class ScenarioGroup:
    """Runs of one canonical scenario, split by pipeline (run id order)."""

    scenario: dict[str, Any]
    green: tuple[RunView, ...]
    baseline: tuple[RunView, ...]

    @property
    def comparable(self) -> bool:
        """True when both pipelines have at least one run."""
        return bool(self.green) and bool(self.baseline)


def split_excluded(
    views: Sequence[RunView],
) -> tuple[list[RunView], list[ExcludedRun]]:
    """Separate the runs whose LLM was unavailable (T9-4) from the usable ones."""
    included: list[RunView] = []
    excluded: list[ExcludedRun] = []
    for view in views:
        if view.notes.get(NOTE_LLM_STATUS) == LLM_STATUS_UNAVAILABLE:
            excluded.append(ExcludedRun(view.run_id, view.pipeline.value, EXCLUSION_REASON))
        else:
            included.append(view)
    return included, excluded


def group_by_scenario(views: Sequence[RunView]) -> list[ScenarioGroup]:
    """Groups of identical canonical `scenario_json`, ordered by their first run."""
    members: dict[str, list[RunView]] = {}
    for view in sorted(views, key=lambda item: item.run_id):
        members.setdefault(scenario_key(view.run.scenario), []).append(view)
    return [build_group(items) for items in members.values()]


def build_group(items: Sequence[RunView]) -> ScenarioGroup:
    """A scenario group from its runs."""
    return ScenarioGroup(
        scenario=dict(items[0].run.scenario),
        green=tuple(v for v in items if v.pipeline is PipelineName.GREEN),
        baseline=tuple(v for v in items if v.pipeline is PipelineName.BASELINE),
    )


def skipped_group(group: ScenarioGroup) -> SkippedGroup:
    """The report entry of a group that lacks one pipeline."""
    return SkippedGroup(
        scenario=group.scenario,
        green_run_ids=tuple(view.run_id for view in group.green),
        baseline_run_ids=tuple(view.run_id for view in group.baseline),
    )
