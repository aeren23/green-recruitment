"""Fill `precision_at_k` and `overlap_at_k` of the stored runs (SPEC 7.7).

The metric arithmetic is in `bench.quality`; this module only gathers the
inputs (rankings files, the dataset category lookup, the seeded postings),
pairs the runs and hands the numbers to the store. Rows are updated only
through `BenchmarkStore`.

Pairing (decision for T-11): within one scenario group the n-th green run is
paired with the n-th baseline run, both ordered by run id (repeat order). If
the counts differ, only the available pairs are used and the difference is
reported. A baseline run takes the mean overlap of the pairs it belongs to.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.models import PipelineName
from app.seed import SEED_POSTINGS
from bench.dataset import ResumeRecord
from bench.quality import QualityContext, QualityResult, overlap_at_k, precision_at_k
from bench.results import Rankings
from bench.scenario import scenario_key
from bench.store import BenchmarkStore, StoredRun


@dataclass(frozen=True)
class OverlapPair:
    """One green run paired with one baseline run of the same scenario group.

    Attributes:
        green_run_id: Id of the green run.
        baseline_run_id: Id of the baseline run.
        value: Their `overlap@k`, or `None` when no posting could be counted.
    """

    green_run_id: int
    baseline_run_id: int
    value: float | None


@dataclass(frozen=True)
class CountMismatch:
    """A scenario group whose green and baseline run counts differ."""

    scenario: str
    green_runs: int
    baseline_runs: int


@dataclass(frozen=True)
class RunPairing:
    """The result of pairing runs by repeat order.

    Attributes:
        pairs: (green run, baseline run) in repeat order.
        unpaired_run_ids: Runs with no partner.
        mismatches: Groups whose counts differ.
    """

    pairs: tuple[tuple[StoredRun, StoredRun], ...]
    unpaired_run_ids: tuple[int, ...]
    mismatches: tuple[CountMismatch, ...]


@dataclass(frozen=True)
class QualityUpdate:
    """What `update_quality_metrics` did, for the report's quality notes.

    Attributes:
        precision_excluded: Postings left out of `precision@k`, by run id.
        runs_without_rankings: Runs with no rankings file (metrics untouched).
        pairs: The overlap pairs with their values.
        unpaired_run_ids: Runs with rankings but no partner (overlap cleared).
        mismatches: Scenario groups with unequal green and baseline counts.
    """

    precision_excluded: Mapping[int, tuple[str, ...]]
    runs_without_rankings: tuple[int, ...]
    pairs: tuple[OverlapPair, ...]
    unpaired_run_ids: tuple[int, ...]
    mismatches: tuple[CountMismatch, ...]


def relevant_categories_by_title() -> dict[str, frozenset[str]]:
    """Relevant dataset categories of every seeded posting, by title."""
    return {seed.title: frozenset(seed.relevant_categories) for seed in SEED_POSTINGS}


def build_quality_context(records: Iterable[ResumeRecord], k: int) -> QualityContext:
    """The quality lookups from dataset records (record id to category)."""
    return QualityContext(
        categories={record.id: record.category for record in records},
        relevant_categories=relevant_categories_by_title(),
        k=k,
    )


def group_runs_by_scenario(runs: Sequence[StoredRun]) -> dict[str, list[StoredRun]]:
    """Runs by canonical scenario key, each list ordered by run id."""
    groups: dict[str, list[StoredRun]] = defaultdict(list)
    for run in sorted(runs, key=lambda item: item.run_id):
        groups[scenario_key(run.scenario)].append(run)
    return dict(groups)


def pair_by_repeat_order(runs: Sequence[StoredRun]) -> RunPairing:
    """Pair the n-th green with the n-th baseline run of each scenario group."""
    pairs: list[tuple[StoredRun, StoredRun]] = []
    unpaired: list[int] = []
    mismatches: list[CountMismatch] = []
    for key, members in group_runs_by_scenario(runs).items():
        greens = [run for run in members if run.pipeline is PipelineName.GREEN]
        baselines = [run for run in members if run.pipeline is PipelineName.BASELINE]
        pairs.extend(zip(greens, baselines, strict=False))
        paired = min(len(greens), len(baselines))
        unpaired.extend(run.run_id for run in greens[paired:] + baselines[paired:])
        if len(greens) != len(baselines):
            mismatches.append(CountMismatch(key, len(greens), len(baselines)))
    return RunPairing(tuple(pairs), tuple(unpaired), tuple(mismatches))


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Mean of the present values, or `None` when there are none."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def compute_overlap_pairs(
    pairing: RunPairing, rankings: Mapping[int, Rankings], k: int
) -> tuple[OverlapPair, ...]:
    """`overlap@k` of every pair."""
    return tuple(
        OverlapPair(
            green.run_id,
            baseline.run_id,
            overlap_at_k(rankings[green.run_id], rankings[baseline.run_id], k).value,
        )
        for green, baseline in pairing.pairs
    )


def overlap_by_run(
    pairs: Sequence[OverlapPair], unpaired_run_ids: Sequence[int]
) -> dict[int, float | None]:
    """The value to store per run: green takes its pair, baseline the mean of its pairs."""
    by_baseline: dict[int, list[float | None]] = defaultdict(list)
    values: dict[int, float | None] = {run_id: None for run_id in unpaired_run_ids}
    for pair in pairs:
        values[pair.green_run_id] = pair.value
        by_baseline[pair.baseline_run_id].append(pair.value)
    values.update({run_id: mean_or_none(items) for run_id, items in by_baseline.items()})
    return values


def update_quality_metrics(
    store: BenchmarkStore, context: QualityContext
) -> QualityUpdate:
    """Compute and store `precision_at_k` and `overlap_at_k` of every run.

    Runs without a rankings file keep their stored values. Re-running replaces
    earlier values, so the result always follows the current rankings files.
    """
    runs = store.load_runs()
    loaded = {run.run_id: store.load_rankings(run.run_id) for run in runs}
    rankings = {run_id: found for run_id, found in loaded.items() if found is not None}
    precision = {run_id: precision_at_k(found, context) for run_id, found in rankings.items()}
    store.save_precision({run_id: result.value for run_id, result in precision.items()})
    pairing = pair_by_repeat_order([run for run in runs if run.run_id in rankings])
    pairs = compute_overlap_pairs(pairing, rankings, context.k)
    store.save_overlap(overlap_by_run(pairs, pairing.unpaired_run_ids))
    return QualityUpdate(
        precision_excluded=excluded_postings_by_run(precision),
        runs_without_rankings=tuple(r.run_id for r in runs if r.run_id not in rankings),
        pairs=pairs,
        unpaired_run_ids=pairing.unpaired_run_ids,
        mismatches=pairing.mismatches,
    )


def excluded_postings_by_run(
    precision: Mapping[int, QualityResult],
) -> dict[int, tuple[str, ...]]:
    """Postings left out of `precision@k`, for the runs where there were any."""
    return {
        run_id: result.excluded_postings
        for run_id, result in precision.items()
        if result.excluded_postings
    }
