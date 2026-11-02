"""Pairing by repeat order and the write-back of quality metrics through the store."""

from pathlib import Path

import pytest
from report_support import (
    CATEGORY_CHEF,
    CATEGORY_IT,
    POSTING_HR,
    POSTING_IT,
    SECOND_SCENARIO,
    insert_runs,
    make_run,
    quality_context,
    write_rankings_file,
)

from app.models import PipelineName
from bench.quality_update import (
    OverlapPair,
    overlap_by_run,
    pair_by_repeat_order,
    update_quality_metrics,
)
from bench.store import BenchmarkStore, BenchPaths

GREEN = PipelineName.GREEN
BASELINE = PipelineName.BASELINE
CATEGORIES = {"it1": CATEGORY_IT, "it2": CATEGORY_IT, "chef1": CATEGORY_CHEF, "hr1": "HR"}


def run_ids(pairs: tuple[tuple[object, object], ...]) -> list[tuple[int, int]]:
    """(green id, baseline id) of every pair."""
    return [(green.run_id, baseline.run_id) for green, baseline in pairs]  # type: ignore[attr-defined]


def test_pairing_matches_the_nth_green_with_the_nth_baseline() -> None:
    runs = [make_run(1, GREEN), make_run(2, BASELINE), make_run(3, GREEN), make_run(4, BASELINE)]

    pairing = pair_by_repeat_order(runs)

    assert run_ids(pairing.pairs) == [(1, 2), (3, 4)]
    assert pairing.unpaired_run_ids == ()
    assert pairing.mismatches == ()


def test_pairing_with_unequal_counts_uses_the_available_pairs_and_reports() -> None:
    runs = [make_run(1, GREEN), make_run(2, BASELINE), make_run(3, GREEN), make_run(5, GREEN)]

    pairing = pair_by_repeat_order(runs)

    assert run_ids(pairing.pairs) == [(1, 2)]
    assert pairing.unpaired_run_ids == (3, 5)
    assert [(m.green_runs, m.baseline_runs) for m in pairing.mismatches] == [(3, 1)]


def test_pairing_never_mixes_scenarios() -> None:
    runs = [
        make_run(1, GREEN),
        make_run(2, BASELINE, scenario=dict(SECOND_SCENARIO)),
        make_run(3, BASELINE),
        make_run(4, GREEN, scenario=dict(SECOND_SCENARIO)),
    ]

    pairing = pair_by_repeat_order(runs)

    assert sorted(run_ids(pairing.pairs)) == [(1, 3), (4, 2)]


def test_baseline_takes_the_mean_overlap_of_its_pairs_and_unpaired_are_cleared() -> None:
    pairs = [OverlapPair(1, 9, 0.5), OverlapPair(2, 9, 1.0)]

    values = overlap_by_run(pairs, unpaired_run_ids=(7,))

    assert values == {1: 0.5, 2: 1.0, 9: 0.75, 7: None}


def test_update_fills_precision_and_overlap_through_the_store(tmp_path: Path) -> None:
    paths = BenchPaths(tmp_path)
    insert_runs(
        paths,
        [
            make_run(1, GREEN, precision_at_k=None),
            make_run(2, BASELINE, precision_at_k=None),
            make_run(3, GREEN, precision_at_k=None),
        ],
    )
    write_rankings_file(paths, 1, {POSTING_IT: ["it1", "it2"], POSTING_HR: ["hr1"]})
    write_rankings_file(paths, 2, {POSTING_IT: ["it1", "it1", "chef1"], POSTING_HR: ["hr1"]})
    store = BenchmarkStore(paths)

    update = update_quality_metrics(store, quality_context(CATEGORIES, k=2))

    stored = {run.run_id: run for run in store.load_runs()}
    assert stored[1].precision_at_k == pytest.approx(1.0)
    assert stored[2].precision_at_k == pytest.approx((0.5 + 1.0) / 2)
    assert stored[3].precision_at_k is None
    assert update.runs_without_rankings == (3,)
    assert stored[1].overlap_at_k == pytest.approx((0.5 + 1.0) / 2)
    assert stored[2].overlap_at_k == stored[1].overlap_at_k
    assert stored[3].overlap_at_k is None
    assert [(m.green_runs, m.baseline_runs) for m in update.mismatches] == []


def test_update_reports_unequal_counts_and_is_repeatable(tmp_path: Path) -> None:
    paths = BenchPaths(tmp_path)
    insert_runs(paths, [make_run(1, GREEN), make_run(2, GREEN), make_run(3, BASELINE)])
    for run_id in (1, 2, 3):
        write_rankings_file(paths, run_id, {POSTING_IT: ["it1", "it2"]})
    store = BenchmarkStore(paths)
    context = quality_context(CATEGORIES, k=2)

    first = update_quality_metrics(store, context)
    second = update_quality_metrics(store, context)

    assert first == second
    assert first.unpaired_run_ids == (2,)
    assert [(m.green_runs, m.baseline_runs) for m in first.mismatches] == [(2, 1)]
    assert {run.run_id: run.overlap_at_k for run in store.load_runs()} == {
        1: 1.0,
        2: None,
        3: 1.0,
    }


def test_rankings_reader_ignores_every_field_but_ids_and_scores(tmp_path: Path) -> None:
    paths = BenchPaths(tmp_path)
    write_rankings_file(paths, 1, {POSTING_IT: ["it1"]})

    rankings = BenchmarkStore(paths).load_rankings(1)

    assert rankings is not None
    assert [(e.record_id, e.score) for e in rankings[POSTING_IT]] == [("it1", 0.5)]
