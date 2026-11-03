"""The yearly projection: scaling arithmetic and the rendered section."""

import pytest
from report_support import make_run

from app.models import PipelineName
from bench.report import build_report
from bench.report_model import MetricRow, Summary
from bench.report_projection import (
    KEY_GROSS_ENERGY,
    KEY_NET_ENERGY,
    KEY_SCI,
    PROJECTED_APPLICATIONS_PER_YEAR,
    build_projection,
    scale_to_year,
)
from bench.report_render import PROJECTION_HEADING, PROJECTION_NOTE, render_markdown

QUALITY_K = 10


def metric_row(key: str, baseline: float | None, green: float | None) -> MetricRow:
    """A metric row with the given means."""
    return MetricRow(key, Summary(baseline, None, 1), Summary(green, None, 1), None)


def test_projected_application_count_is_one_million() -> None:
    assert PROJECTED_APPLICATIONS_PER_YEAR == 1_000_000


def test_scaling_converts_wh_to_kwh_and_g_to_kg() -> None:
    assert scale_to_year(0.02) == pytest.approx(20.0)
    assert scale_to_year(None) is None


def test_projection_rows_have_totals_and_difference() -> None:
    rows = {
        KEY_NET_ENERGY: metric_row(KEY_NET_ENERGY, 0.02, 0.01),
        KEY_GROSS_ENERGY: metric_row(KEY_GROSS_ENERGY, 0.08, 0.04),
        KEY_SCI: metric_row(KEY_SCI, 0.01, None),
    }
    net, gross, sci = build_projection(rows)
    assert (net.baseline, net.green) == (pytest.approx(20.0), pytest.approx(10.0))
    assert net.difference_pct == pytest.approx(-50.0)
    assert gross.green == pytest.approx(40.0)
    assert sci.green is None and sci.difference_pct is None


def test_report_markdown_contains_the_projection_section() -> None:
    runs = [
        make_run(1, PipelineName.GREEN),
        make_run(2, PipelineName.BASELINE),
    ]
    text = render_markdown(build_report(runs, QUALITY_K))
    assert PROJECTION_HEADING in text
    assert PROJECTION_NOTE in text
    assert "net energy (kWh)" in text
