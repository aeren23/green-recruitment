"""N=300 duration estimate printed after a baseline run (decision T10-4).

Estimate = mean per-application stage seconds of the run just finished times
the number of applications of the N=300 scenario, separately for scenario A
(`jobs_per_cv=1, resubmit_rate=0`) and the default scenario, and the total of
`REPEATS` repeats of both. The application counts come from `build_scenario`
itself, never from hand arithmetic.

The per-application stages are `store_pdf`, `extract` and `llm_score`. `rank`
runs once per posting rather than per application, so it is left out of the
per-application mean. Aggregates only, no CV content.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from app.green.tracker import StageTotals
from bench.dataset import ResumeRecord
from bench.scenario import (
    DEFAULT_JOBS_PER_CV,
    DEFAULT_RESUBMIT_RATE,
    ScenarioConfig,
    build_scenario,
)

ESTIMATE_N = 300
REPEATS = 3
SCENARIO_A_JOBS_PER_CV = 1
SCENARIO_A_RESUBMIT_RATE = 0.0
SCENARIO_A_LABEL = "scenario A (jobs_per_cv=1, resubmit_rate=0)"
SCENARIO_DEFAULT_LABEL = "default scenario"
PER_APPLICATION_STAGES: tuple[str, ...] = ("store_pdf", "extract", "llm_score")
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
MINUTES_PER_HOUR = 60


@dataclass(frozen=True)
class ScenarioCount:
    """The application count of one N=300 scenario."""

    label: str
    applications: int


@dataclass(frozen=True)
class ScenarioEstimate:
    """The estimated duration of one N=300 scenario run."""

    label: str
    applications: int
    seconds: float


@dataclass(frozen=True)
class DurationEstimate:
    """Estimates for the N=300 scenarios and the total for all repeats.

    Attributes:
        per_application_s: Mean per-application stage seconds of the run.
        scenarios: One estimate per scenario.
        repeats: How many repeats of every scenario the total covers.
        total_seconds: `repeats` times the sum over the scenarios.
    """

    per_application_s: float
    scenarios: tuple[ScenarioEstimate, ...]
    repeats: int
    total_seconds: float


def per_application_seconds(
    stages: Mapping[str, StageTotals], applications: int
) -> float:
    """Mean seconds the per-application stages took for one application."""
    if applications <= 0:
        return 0.0
    staged = sum(
        stages[name].duration_s for name in PER_APPLICATION_STAGES if name in stages
    )
    return staged / applications


def n300_configs(base: ScenarioConfig) -> tuple[tuple[str, ScenarioConfig], ...]:
    """The two N=300 scenarios, sharing the seed and models of `base`."""
    scenario_a = replace(
        base,
        n=ESTIMATE_N,
        jobs_per_cv=SCENARIO_A_JOBS_PER_CV,
        resubmit_rate=SCENARIO_A_RESUBMIT_RATE,
    )
    default = replace(
        base,
        n=ESTIMATE_N,
        jobs_per_cv=DEFAULT_JOBS_PER_CV,
        resubmit_rate=DEFAULT_RESUBMIT_RATE,
    )
    return ((SCENARIO_A_LABEL, scenario_a), (SCENARIO_DEFAULT_LABEL, default))


def count_n300_applications(
    records: Iterable[ResumeRecord], base: ScenarioConfig
) -> tuple[ScenarioCount, ...]:
    """Application counts of the N=300 scenarios, from `build_scenario`.

    Raises:
        ScenarioConfigError: On an invalid scenario.
        ValueError: If a dataset stratum is too small for 300 CVs.
    """
    records = list(records)
    return tuple(
        ScenarioCount(label, len(build_scenario(records, config)))
        for label, config in n300_configs(base)
    )


def estimate_n300(
    per_application_s: float, counts: Iterable[ScenarioCount]
) -> DurationEstimate:
    """Multiply the per-application seconds by every scenario's count."""
    scenarios = tuple(
        ScenarioEstimate(count.label, count.applications, per_application_s * count.applications)
        for count in counts
    )
    total = REPEATS * sum(scenario.seconds for scenario in scenarios)
    return DurationEstimate(per_application_s, scenarios, REPEATS, total)


def format_duration(seconds: float) -> str:
    """`123.4 s (2 min 3 s)` or `... (1 h 05 min)` for long durations."""
    whole_minutes = int(seconds // SECONDS_PER_MINUTE)
    if seconds >= SECONDS_PER_HOUR:
        hours, minutes = divmod(whole_minutes, MINUTES_PER_HOUR)
        return f"{seconds:.0f} s ({hours} h {minutes:02d} min)"
    remainder = seconds - whole_minutes * SECONDS_PER_MINUTE
    return f"{seconds:.1f} s ({whole_minutes} min {remainder:.0f} s)"


def format_estimate(estimate: DurationEstimate) -> str:
    """The estimate as plain text for the console."""
    lines = [
        f"N={ESTIMATE_N} duration estimate "
        f"(mean {estimate.per_application_s:.3f} s per application "
        f"over {', '.join(PER_APPLICATION_STAGES)}):"
    ]
    for scenario in estimate.scenarios:
        lines.append(
            f"  {scenario.label}: {scenario.applications} applications, "
            f"{format_duration(scenario.seconds)}"
        )
    lines.append(
        f"  {estimate.repeats} repeats of both: {format_duration(estimate.total_seconds)}"
    )
    return "\n".join(lines)
