"""Hand-computed checks of `precision_at_k` and `overlap_at_k` (SPEC 7.7)."""

import pytest
from report_support import (
    CATEGORY_CHEF,
    CATEGORY_HR,
    CATEGORY_IT,
    POSTING_ACCOUNTANT,
    POSTING_HR,
    POSTING_IT,
    entries,
    quality_context,
)

from bench.quality import QualityInputError, overlap_at_k, precision_at_k, unique_record_ids

CATEGORIES = {
    "it1": CATEGORY_IT,
    "it2": CATEGORY_IT,
    "it3": CATEGORY_IT,
    "chef1": CATEGORY_CHEF,
    "chef2": CATEGORY_CHEF,
    "hr1": CATEGORY_HR,
    "hr2": CATEGORY_HR,
}


def test_unique_record_ids_keeps_the_first_occurrence() -> None:
    assert unique_record_ids(entries("a", "b", "a", "c", "b")) == ["a", "b", "c"]


def test_precision_dedupe_keeps_the_best_occurrence() -> None:
    """Without dedupe the top 2 would be it1, it1 (1.0); with it, it1 and chef1 (0.5)."""
    context = quality_context(CATEGORIES, k=2)
    rankings = {POSTING_IT: entries("it1", "it1", "chef1", "it2")}

    result = precision_at_k(rankings, context)

    assert result.value == pytest.approx(0.5)


def test_precision_hand_computed_value() -> None:
    """Unique top 4: it1, chef1, it2, chef2 -> 2 of 4 relevant."""
    context = quality_context(CATEGORIES, k=4)
    rankings = {POSTING_IT: entries("it1", "chef1", "it1", "it2", "chef2", "it3")}

    assert precision_at_k(rankings, context).value == pytest.approx(0.5)


def test_precision_with_fewer_candidates_than_k_uses_the_candidate_count() -> None:
    context = quality_context(CATEGORIES, k=10)
    rankings = {POSTING_IT: entries("it1", "chef1", "it1", "it2")}

    assert precision_at_k(rankings, context).value == pytest.approx(2 / 3)


def test_precision_is_averaged_over_postings() -> None:
    context = quality_context(CATEGORIES, k=2)
    rankings = {
        POSTING_IT: entries("it1", "it2"),
        POSTING_HR: entries("hr1", "chef1"),
    }

    result = precision_at_k(rankings, context)

    assert result.per_posting == {POSTING_IT: 1.0, POSTING_HR: 0.5}
    assert result.value == pytest.approx(0.75)


def test_precision_excludes_an_empty_posting_and_reports_it() -> None:
    context = quality_context(CATEGORIES, k=2)
    rankings = {POSTING_IT: [], POSTING_HR: entries("hr1", "hr2")}

    result = precision_at_k(rankings, context)

    assert result.value == pytest.approx(1.0)
    assert result.excluded_postings == (POSTING_IT,)


def test_precision_without_any_candidate_has_no_value() -> None:
    result = precision_at_k({POSTING_IT: []}, quality_context(CATEGORIES))

    assert result.value is None
    assert result.excluded_postings == (POSTING_IT,)


def test_precision_accepts_a_posting_with_several_relevant_categories() -> None:
    categories = {"acc1": "ACCOUNTANT", "fin1": "FINANCE", "chef1": CATEGORY_CHEF}
    context = quality_context(categories, k=3)

    result = precision_at_k({POSTING_ACCOUNTANT: entries("acc1", "fin1", "chef1")}, context)

    assert result.value == pytest.approx(2 / 3)


def test_precision_rejects_an_unknown_posting() -> None:
    with pytest.raises(QualityInputError):
        precision_at_k({"Unknown posting": entries("it1")}, quality_context(CATEGORIES))


def test_precision_rejects_a_record_without_category() -> None:
    with pytest.raises(QualityInputError):
        precision_at_k({POSTING_IT: entries("ghost")}, quality_context(CATEGORIES))


def test_overlap_of_identical_rankings_is_one() -> None:
    rankings = {POSTING_IT: entries("a", "b", "c")}

    assert overlap_at_k(rankings, rankings, k=3).value == pytest.approx(1.0)


def test_overlap_of_disjoint_rankings_is_zero() -> None:
    green = {POSTING_IT: entries("a", "b")}
    baseline = {POSTING_IT: entries("c", "d")}

    assert overlap_at_k(green, baseline, k=2).value == pytest.approx(0.0)


def test_overlap_partial_case_is_hand_computed() -> None:
    """Shared c and d out of 4: 2/4."""
    green = {POSTING_IT: entries("a", "b", "c", "d")}
    baseline = {POSTING_IT: entries("c", "d", "e", "f")}

    assert overlap_at_k(green, baseline, k=4).value == pytest.approx(0.5)


def test_overlap_denominator_is_the_smallest_unique_list() -> None:
    """Green has only 2 unique candidates (filtered), so the denominator is 2, not k=4."""
    green = {POSTING_IT: entries("a", "a", "b")}
    baseline = {POSTING_IT: entries("a", "x", "y", "z", "w")}

    assert overlap_at_k(green, baseline, k=4).value == pytest.approx(1 / 2)


def test_overlap_dedupes_both_rankings() -> None:
    green = {POSTING_IT: entries("a", "a", "b")}
    baseline = {POSTING_IT: entries("a", "b", "b")}

    assert overlap_at_k(green, baseline, k=2).value == pytest.approx(1.0)


def test_overlap_compares_only_the_top_k() -> None:
    green = {POSTING_IT: entries("a", "b", "c")}
    baseline = {POSTING_IT: entries("c", "b", "a")}

    assert overlap_at_k(green, baseline, k=2).value == pytest.approx(0.5)


def test_overlap_excludes_postings_missing_on_a_side_or_empty() -> None:
    green = {POSTING_IT: entries("a"), POSTING_HR: entries("h1")}
    baseline = {POSTING_IT: entries("a"), POSTING_ACCOUNTANT: entries("x")}

    result = overlap_at_k(green, baseline, k=3)

    assert result.value == pytest.approx(1.0)
    assert set(result.excluded_postings) == {POSTING_HR, POSTING_ACCOUNTANT}


def test_overlap_averages_postings() -> None:
    green = {POSTING_IT: entries("a", "b"), POSTING_HR: entries("h1", "h2")}
    baseline = {POSTING_IT: entries("a", "b"), POSTING_HR: entries("h1", "x")}

    assert overlap_at_k(green, baseline, k=2).value == pytest.approx(0.75)
