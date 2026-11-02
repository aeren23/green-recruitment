"""Grouping, statistics, K1 criteria, conditions and the files of `bench compare`."""

import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest
from bench_support import CvRecordSpec, build_dataset_tree
from report_support import (
    CATEGORY_IT,
    POSTING_IT,
    SECOND_SCENARIO,
    build_notes,
    insert_runs,
    make_run,
    write_rankings_file,
)

from app.models import PipelineName
import bench.report_model as report_model
from bench.compare_command import CompareRequest, execute_compare
from bench.quality_update import CountMismatch, OverlapPair, QualityUpdate
from bench.report import build_report, report_to_json, write_report
from bench.report_k1 import evaluate_k1
from bench.report_metrics import difference_pct, summarize
from bench.report_model import GroupReport, MetricRow, Report, Summary
from bench.report_render import format_summary, render_markdown
from bench.store import BenchPaths, StoredRun, StoredStage

GREEN = PipelineName.GREEN
BASELINE = PipelineName.BASELINE
QUALITY_K = 10
WARNING_TOKEN = "WARNING"
SCENARIO_KEYS = {"jobs_per_cv", "models", "embedding", "llm", "n", "resubmit_rate", "seed", "top_k"}
FORBIDDEN_KEYS = {"reason", "text", "name", "email", "phone", "record_id", "score"}


def only_group(report: Report) -> GroupReport:
    """The single group of a report."""
    assert len(report.groups) == 1
    return report.groups[0]


def metric(group: GroupReport, key: str) -> MetricRow:
    """A main-table row by key."""
    return next(row for row in group.metrics if row.key == key)


def pair_of_runs(**green_fields: Any) -> list[StoredRun]:
    """One baseline and one green run in final conditions."""
    return [
        make_run(1, GREEN, **green_fields),
        make_run(2, BASELINE),
    ]


def row(key: str, baseline: float | None, green: float | None) -> MetricRow:
    """A metric row with single-run summaries."""
    return MetricRow(
        key=key,
        baseline=Summary(baseline, None, 0 if baseline is None else 1),
        green=Summary(green, None, 0 if green is None else 1),
        difference_pct=None,
    )


def k1_rows(**values: tuple[float, float]) -> dict[str, MetricRow]:
    """Rows for the four K1 metrics; green passes by default."""
    defaults = {
        "energy_wh_net_per_application": (2.0, 1.0),
        "llm_calls": (100.0, 40.0),
        "storage_bytes_after_retention": (1000.0, 100.0),
        "precision_at_k": (0.5, 0.5),
    }
    merged = {**defaults, **values}
    return {key: row(key, pair[0], pair[1]) for key, pair in merged.items()}


# --- statistics -----------------------------------------------------------------


def test_summarize_gives_mean_and_sample_standard_deviation() -> None:
    summary = summarize([1.0, 2.0, 3.0])

    assert summary.mean == pytest.approx(2.0)
    assert summary.sd == pytest.approx(1.0)  # sample sd (n - 1), not 0.816
    assert summary.n == 3


def test_summarize_single_value_has_no_deviation() -> None:
    summary = summarize([4.0])

    assert (summary.mean, summary.sd, summary.n) == (4.0, None, 1)
    assert format_summary(summary, 2) == "4.00 ± n/a"


def test_summarize_ignores_missing_values_and_handles_none_at_all() -> None:
    assert summarize([None, 2.0, 4.0]).n == 2
    assert summarize([None]) == Summary(None, None, 0)
    assert format_summary(summarize([]), 2) == "n/a"


def test_difference_percent_sign_and_zero_baseline() -> None:
    assert difference_pct(10.0, 8.0) == pytest.approx(-20.0)
    assert difference_pct(10.0, 12.5) == pytest.approx(25.0)
    assert difference_pct(0.0, 3.0) is None
    assert difference_pct(None, 3.0) is None


# --- grouping and exclusion -----------------------------------------------------


def test_runs_are_grouped_by_canonical_scenario() -> None:
    runs = [
        make_run(1, GREEN),
        make_run(2, BASELINE),
        make_run(3, GREEN, scenario=dict(SECOND_SCENARIO)),
        make_run(4, BASELINE, scenario=dict(reversed(list(SECOND_SCENARIO.items())))),
    ]

    report = build_report(runs, QUALITY_K)

    assert [g.green_run_ids + g.baseline_run_ids for g in report.groups] == [(1, 2), (3, 4)]
    assert report.skipped_groups == ()


def test_a_group_with_one_pipeline_is_listed_as_skipped() -> None:
    runs = [make_run(1, GREEN), make_run(2, BASELINE), make_run(3, GREEN, scenario=SECOND_SCENARIO)]

    report = build_report(runs, QUALITY_K)

    assert len(report.groups) == 1
    assert [g.green_run_ids for g in report.skipped_groups] == [(3,)]


def test_llm_unavailable_runs_are_excluded_and_listed() -> None:
    runs = [
        make_run(1, GREEN),
        make_run(2, GREEN, notes=build_notes(llm_status="llm_unavailable")),
        make_run(3, BASELINE),
    ]

    report = build_report(runs, QUALITY_K)

    group = only_group(report)
    assert group.green_run_ids == (1,)
    assert [(run.run_id, run.pipeline) for run in report.excluded_runs] == [(2, "green")]
    assert "llm_unavailable" in report.excluded_runs[0].reason
    assert "run 2 (green): llm_status=llm_unavailable" in render_markdown(report)


# --- main table -----------------------------------------------------------------


def test_main_table_mean_sd_and_difference() -> None:
    runs = [
        make_run(1, GREEN, energy_wh_gross=8.0),
        make_run(2, GREEN, energy_wh_gross=12.0),
        make_run(3, BASELINE, energy_wh_gross=20.0),
    ]

    gross = metric(only_group(build_report(runs, QUALITY_K)), "energy_wh_gross")

    assert gross.green.mean == pytest.approx(10.0)
    assert gross.green.sd == pytest.approx(math.sqrt(8.0))
    assert gross.baseline.sd is None
    assert gross.difference_pct == pytest.approx(-50.0)


def test_per_application_values_and_sci_use_the_applications_note() -> None:
    runs = [
        make_run(1, GREEN, energy_wh_gross=10.0, energy_wh_net=4.0, co2_g=5.0,
                 notes=build_notes(applications="50")),
        make_run(2, BASELINE, energy_wh_gross=10.0, energy_wh_net=4.0, co2_g=5.0,
                 notes=build_notes(applications="100")),
    ]  # fmt: skip

    group = only_group(build_report(runs, QUALITY_K))

    assert metric(group, "energy_wh_gross_per_application").green.mean == pytest.approx(0.2)
    assert metric(group, "energy_wh_net_per_application").baseline.mean == pytest.approx(0.04)
    assert metric(group, "sci_g_per_application").green.mean == pytest.approx(0.1)
    assert metric(group, "sci_g_per_application").baseline.mean == pytest.approx(0.05)


def test_per_application_is_unavailable_without_an_applications_note() -> None:
    runs = [make_run(1, GREEN, notes="llm_status=available"), make_run(2, BASELINE)]

    group = only_group(build_report(runs, QUALITY_K))

    assert metric(group, "sci_g_per_application").green.mean is None


def test_every_requested_metric_is_in_the_main_table() -> None:
    group = only_group(build_report(pair_of_runs(), QUALITY_K))

    keys = {row_.key for row_ in group.metrics}

    assert {
        "energy_wh_gross", "energy_wh_net", "co2_g", "duration_s",
        "energy_wh_gross_per_application", "energy_wh_net_per_application",
        "sci_g_per_application", "storage_bytes_active", "storage_bytes_after_retention",
        "llm_calls", "llm_prompt_tokens", "llm_completion_tokens",
        "parse_cache_hits", "embed_cache_hits", "precision_at_k", "overlap_at_k",
    } <= keys  # fmt: skip


# --- stage table ----------------------------------------------------------------


def stage_run(run_id: int, pipeline: PipelineName, ms_per_call: float) -> StoredRun:
    """A run whose `extract` stage takes the given milliseconds per call."""
    stages = (
        StoredStage("extract", 1.0, ms_per_call, 1000),
        StoredStage("model_load", 5.0, 9.0, 1),
    )
    return make_run(run_id, pipeline, stages=stages)


@pytest.mark.parametrize(
    ("seconds_for_1000_calls", "approximate"),
    [(50.0, True), (99.0, True), (100.0, False), (150.0, False)],
)
def test_stages_under_100_ms_per_call_are_flagged_approximate(
    seconds_for_1000_calls: float, approximate: bool
) -> None:
    runs = [stage_run(1, GREEN, seconds_for_1000_calls), stage_run(2, BASELINE, 500.0)]

    stages = {s.stage: s for s in only_group(build_report(runs, QUALITY_K)).stages.green}

    assert stages["extract"].ms_per_call == pytest.approx(seconds_for_1000_calls)
    assert stages["extract"].approximate is approximate


def test_model_load_is_listed_but_not_counted_in_totals() -> None:
    runs = [stage_run(1, GREEN, 150.0), stage_run(2, BASELINE, 150.0)]

    stages = {s.stage: s for s in only_group(build_report(runs, QUALITY_K)).stages.green}

    assert stages["model_load"].counted_in_totals is False
    assert stages["extract"].counted_in_totals is True


# --- K1 -------------------------------------------------------------------------


def results(evaluation: Any) -> dict[str, bool | None]:
    """Pass flag by criterion key."""
    return {criterion.key: criterion.passed for criterion in evaluation.criteria}


def test_k1_all_criteria_pass() -> None:
    evaluation = evaluate_k1(k1_rows())

    assert all(results(evaluation).values())
    assert evaluation.all_passed is True


@pytest.mark.parametrize(
    "failing_key",
    ["energy_wh_net_per_application", "llm_calls", "storage_bytes_after_retention"],
)
def test_k1_strict_criteria_fail_when_green_is_not_lower(failing_key: str) -> None:
    equal = evaluate_k1(k1_rows(**{failing_key: (5.0, 5.0)}))
    higher = evaluate_k1(k1_rows(**{failing_key: (5.0, 6.0)}))

    assert results(equal)[failing_key] is False
    assert results(higher)[failing_key] is False
    assert equal.all_passed is False


@pytest.mark.parametrize(
    ("baseline", "green", "passed"),
    [
        (0.70, 0.60, True),  # exactly baseline - 0.10: allowed (D-07)
        (0.50, 0.40, True),
        (0.90, 0.80, True),
        (0.30, 0.20, True),
        (0.70, 0.5999, False),
        (0.50, 0.39, False),
        (0.50, 0.90, True),
    ],
)
def test_k1_quality_boundary(baseline: float, green: float, passed: bool) -> None:
    evaluation = evaluate_k1(k1_rows(precision_at_k=(baseline, green)))

    assert results(evaluation)["precision_at_k"] is passed
    quality = evaluation.criteria[-1]
    assert quality.threshold == pytest.approx(baseline - 0.10)


def test_k1_with_a_missing_number_is_not_evaluable() -> None:
    rows = k1_rows()
    rows["precision_at_k"] = row("precision_at_k", None, 0.5)

    evaluation = evaluate_k1(rows)

    assert results(evaluation)["precision_at_k"] is None
    assert evaluation.all_passed is None


def test_k1_section_shows_pass_fail_and_exact_numbers() -> None:
    runs = [make_run(1, GREEN, precision_at_k=0.61, llm_calls=60), make_run(2, BASELINE, precision_at_k=0.7)]

    text = render_markdown(build_report(runs, QUALITY_K))

    assert "K1 değerlendirmesi" in text
    assert "| `precision_at_k` | quality (D-07): green precision@k >= baseline precision@k - 0.10 | 0.700 | 0.610 | 0.600 | PASS |" in text
    assert "| `llm_calls` | LLM calls: green < baseline | 40.0 | 60.0 | n/a | FAIL |" in text


# --- conditions -----------------------------------------------------------------


def test_final_conditions_produce_no_warning() -> None:
    report = build_report(pair_of_runs(), QUALITY_K)

    assert only_group(report).conditions.valid_for_k1 is True
    assert WARNING_TOKEN not in render_markdown(report)


@pytest.mark.parametrize(
    ("override", "expected_text"),
    [
        ({"power_plan": "Güç tasarrufu"}, "power plan is 'Güç tasarrufu'"),
        ({"onedrive_running": "true"}, "OneDrive state is 'true'"),
        ({"git_dirty": "true"}, "git tree state is 'true'"),
    ],
)
def test_each_non_final_condition_triggers_the_warning(
    override: dict[str, str], expected_text: str
) -> None:
    runs = [make_run(1, GREEN, notes=build_notes(**override)), make_run(2, BASELINE)]

    report = build_report(runs, QUALITY_K)
    text = render_markdown(report)

    assert only_group(report).conditions.valid_for_k1 is False
    assert f"{WARNING_TOKEN}: bu grup geçerli bir K1 ölçümü DEĞİLDİR" in text
    assert f"run 1 (green): {expected_text}" in text
    assert "run 2 (baseline):" not in text


def test_conditions_list_the_per_run_facts() -> None:
    runs = [
        make_run(1, GREEN, idle_power_w=46.03,
                 notes=build_notes(git="c01c638", gpu_stage_sum_over_stream="0.948119",
                                   unstaged_share="0.027639")),
        make_run(2, BASELINE),
    ]  # fmt: skip

    condition = only_group(build_report(runs, QUALITY_K)).conditions.runs[0]

    assert (condition.git_revision, condition.git_dirty) == ("c01c638", "false")
    assert condition.idle_power_w == pytest.approx(46.03)
    assert condition.gpu_stage_sum_over_stream == pytest.approx(0.948119)
    assert condition.unstaged_share == pytest.approx(0.027639)


def test_extract_difference_that_penalizes_green_is_stated() -> None:
    runs = [stage_run(1, GREEN, 492.0), stage_run(2, BASELINE, 378.0)]

    report = build_report(runs, QUALITY_K)
    extract = only_group(report).conditions.extract

    assert extract.difference_ms == pytest.approx(114.0)
    assert extract.penalizes_green is True
    assert "green aleyhine temkinlidir" in render_markdown(report)


def test_fixed_method_sentences_are_in_the_report() -> None:
    text = render_markdown(build_report(pair_of_runs(), QUALITY_K))

    for fragment in (
        "Windows Energy Meter Interface",
        "sabit güç modelidir",
        "Gömülü karbon (M) ölçüm dışıdır",
        "dil ve konum filtreleri kapalıdır",
        "3B modelle puanlanır",
    ):
        assert fragment in text


# --- files ----------------------------------------------------------------------


def collect_keys(node: Any) -> set[str]:
    """All dictionary keys of a parsed JSON document."""
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in collect_keys(value)}
    if isinstance(node, list):
        return {key for item in node for key in collect_keys(item)}
    return set()


def allowed_keys() -> set[str]:
    """Every dataclass field name of the report plus the scenario parameter names."""
    classes = [
        member
        for _, member in inspect.getmembers(report_model, inspect.isclass)
        if dataclasses.is_dataclass(member)
    ]
    classes.extend([QualityUpdate, OverlapPair, CountMismatch])
    names = {f.name for cls in classes for f in dataclasses.fields(cls)}
    return names | SCENARIO_KEYS


def test_report_files_hold_only_aggregates_and_ids(tmp_path: Path) -> None:
    secret_notes = build_notes() + "; reason=SECRET-CV-TEXT; candidate_name=SECRET-NAME"
    runs = [make_run(1, GREEN, notes=secret_notes), make_run(2, BASELINE, notes=secret_notes)]
    quality = QualityUpdate({1: (POSTING_IT,)}, (), (OverlapPair(1, 2, 0.5),), (), ())
    report = build_report(runs, QUALITY_K, quality)

    paths = write_report(report, tmp_path)

    markdown = paths.markdown.read_text(encoding="utf-8")
    document = json.loads(paths.json.read_text(encoding="utf-8"))
    assert "SECRET" not in markdown + paths.json.read_text(encoding="utf-8")
    unexpected = {key for key in collect_keys(document) if not key.isdigit()} - allowed_keys()
    assert unexpected == set()
    assert not collect_keys(document) & FORBIDDEN_KEYS


def test_json_and_markdown_carry_the_same_numbers(tmp_path: Path) -> None:
    runs = [make_run(1, GREEN, energy_wh_gross=7.5), make_run(2, BASELINE, energy_wh_gross=10.0)]
    report = build_report(runs, QUALITY_K)

    document = json.loads(report_to_json(report))

    gross = next(m for m in document["groups"][0]["metrics"] if m["key"] == "energy_wh_gross")
    assert gross["green"]["mean"] == pytest.approx(7.5)
    assert gross["difference_pct"] == pytest.approx(-25.0)
    assert "-25.0%" in render_markdown(report)


def test_compare_command_updates_metrics_and_writes_both_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = BenchPaths(tmp_path / "bench")
    dataset_root = tmp_path / "dataset"
    build_dataset_tree(
        dataset_root,
        [CvRecordSpec("it1", CATEGORY_IT, 90), CvRecordSpec("it2", "CHEF", 80)],
    )
    insert_runs(paths, [make_run(1, GREEN, precision_at_k=None), make_run(2, BASELINE)])
    write_rankings_file(paths, 1, {POSTING_IT: ["it1", "it2"]})
    write_rankings_file(paths, 2, {POSTING_IT: ["it1", "it2"]})

    code = execute_compare(CompareRequest(paths, dataset_root, QUALITY_K))

    output = capsys.readouterr().out
    assert code == 0
    assert (paths.directory / "report.md").is_file()
    assert (paths.directory / "report.json").is_file()
    assert "K1 overall" in output
    document = json.loads((paths.directory / "report.json").read_text(encoding="utf-8"))
    precision = next(m for m in document["groups"][0]["metrics"] if m["key"] == "precision_at_k")
    assert precision["green"]["mean"] == pytest.approx(0.5)


def test_compare_command_fails_cleanly_without_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = execute_compare(CompareRequest(BenchPaths(tmp_path), tmp_path, QUALITY_K))

    assert code == 1
    assert "no benchmark runs" in capsys.readouterr().err
