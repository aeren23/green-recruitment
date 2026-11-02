"""The K1 evaluation with the criteria pre-registered in plan 8.2.

Every criterion compares group means of the two pipelines:

* net energy per application: green < baseline;
* LLM calls: green < baseline;
* storage after retention: green < baseline;
* quality (D-07): green `precision@10` >= baseline `precision@10` - 0.10.
"""

from collections.abc import Callable, Mapping

from bench.report_model import CriterionResult, K1Evaluation, MetricRow

KEY_NET_ENERGY = "energy_wh_net_per_application"
KEY_LLM_CALLS = "llm_calls"
KEY_STORAGE = "storage_bytes_after_retention"
KEY_PRECISION = "precision_at_k"

RULE_NET_ENERGY = "net energy per application: green < baseline"
RULE_LLM_CALLS = "LLM calls: green < baseline"
RULE_STORAGE = "storage after retention: green < baseline"
RULE_QUALITY = "quality (D-07): green precision@k >= baseline precision@k - 0.10"

QUALITY_TOLERANCE = 0.10
COMPARISON_DECIMALS = 9


def is_strictly_lower(green: float, baseline: float) -> bool:
    """The green number is below the baseline number."""
    return green < baseline


def is_within_quality_tolerance(green: float, baseline: float) -> bool:
    """`green >= baseline - 0.10`, compared after rounding away float noise.

    Rounding makes the exact -0.10 boundary a pass, as D-07 states.
    """
    floor = round(baseline - QUALITY_TOLERANCE, COMPARISON_DECIMALS)
    return round(green, COMPARISON_DECIMALS) >= floor


def evaluate_criterion(
    rule: str,
    row: MetricRow,
    compare: Callable[[float, float], bool],
) -> CriterionResult:
    """One criterion from a metric row; `passed` is `None` when a mean is missing."""
    baseline, green = row.baseline.mean, row.green.mean
    passed = None if baseline is None or green is None else compare(green, baseline)
    return CriterionResult(
        key=row.key,
        rule=rule,
        baseline_value=baseline,
        green_value=green,
        threshold=None,
        passed=passed,
    )


def with_quality_threshold(result: CriterionResult) -> CriterionResult:
    """Attach the baseline-minus-tolerance threshold to the quality criterion."""
    if result.baseline_value is None:
        return result
    return CriterionResult(
        key=result.key,
        rule=result.rule,
        baseline_value=result.baseline_value,
        green_value=result.green_value,
        threshold=result.baseline_value - QUALITY_TOLERANCE,
        passed=result.passed,
    )


def overall_verdict(criteria: tuple[CriterionResult, ...]) -> bool | None:
    """True when all pass, False when one fails, `None` when one is not evaluable."""
    results = [criterion.passed for criterion in criteria]
    if any(result is False for result in results):
        return False
    return None if any(result is None for result in results) else True


def evaluate_k1(rows: Mapping[str, MetricRow]) -> K1Evaluation:
    """Evaluate the four K1 criteria from the metric rows of one group."""
    criteria = (
        evaluate_criterion(RULE_NET_ENERGY, rows[KEY_NET_ENERGY], is_strictly_lower),
        evaluate_criterion(RULE_LLM_CALLS, rows[KEY_LLM_CALLS], is_strictly_lower),
        evaluate_criterion(RULE_STORAGE, rows[KEY_STORAGE], is_strictly_lower),
        with_quality_threshold(
            evaluate_criterion(RULE_QUALITY, rows[KEY_PRECISION], is_within_quality_tolerance)
        ),
    )
    return K1Evaluation(criteria=criteria, all_passed=overall_verdict(criteria))
