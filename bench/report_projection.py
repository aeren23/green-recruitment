"""The yearly projection: per-application means scaled to a fixed application count.

The projection is a linear extrapolation of the group means of one scenario,
so it is an estimate for this hardware and this scenario, not a measurement.
Energy is converted from Wh to kWh and the SCI carbon from g to kg CO2e
(M, the embodied carbon, is excluded as everywhere else in the report).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from bench.report_metrics import difference_pct
from bench.report_model import MetricRow

PROJECTED_APPLICATIONS_PER_YEAR = 1_000_000
UNITS_PER_KILO = 1000.0
KEY_NET_ENERGY = "energy_wh_net_per_application"
KEY_GROSS_ENERGY = "energy_wh_gross_per_application"
KEY_SCI = "sci_g_per_application"


@dataclass(frozen=True)
class ProjectionSpec:
    """One projected quantity: the metric it comes from and its label."""

    key: str
    label: str


@dataclass(frozen=True)
class ProjectionRow:
    """A projected yearly total of both pipelines and their difference.

    Attributes:
        label: Row label with the unit.
        baseline: Yearly total of the baseline, `None` without a mean.
        green: Yearly total of the green pipeline, `None` without a mean.
        difference_pct: `(green - baseline) / baseline * 100`, or `None`.
    """

    label: str
    baseline: float | None
    green: float | None
    difference_pct: float | None


PROJECTION_SPECS: tuple[ProjectionSpec, ...] = (
    ProjectionSpec(KEY_NET_ENERGY, "net energy (kWh)"),
    ProjectionSpec(KEY_GROSS_ENERGY, "gross energy (kWh)"),
    ProjectionSpec(KEY_SCI, "sci (kg CO₂e, M excluded)"),
)


def scale_to_year(per_application_mean: float | None) -> float | None:
    """A per-application mean (Wh or g) as a yearly total in kWh or kg."""
    if per_application_mean is None:
        return None
    return per_application_mean * PROJECTED_APPLICATIONS_PER_YEAR / UNITS_PER_KILO


def project_row(spec: ProjectionSpec, row: MetricRow) -> ProjectionRow:
    """The projection of one metric row."""
    baseline = scale_to_year(row.baseline.mean)
    green = scale_to_year(row.green.mean)
    return ProjectionRow(spec.label, baseline, green, difference_pct(baseline, green))


def build_projection(rows: Mapping[str, MetricRow]) -> tuple[ProjectionRow, ...]:
    """The yearly projection rows of one group, from its metric rows by key."""
    return tuple(project_row(spec, rows[spec.key]) for spec in PROJECTION_SPECS)
