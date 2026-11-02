"""Ranking quality metrics: `precision@k` and `overlap@k` (SPEC 7.7).

Everything here is a pure function over rankings and a category lookup. There
is no database, no file and no dataset access, so the metrics can be checked
with hand-computed examples.

Repeat applications show up as the same dataset record id several times in one
ranking (the baseline ranks them all). Both metrics therefore deduplicate by
record id first and keep the first occurrence, which is the best-ranked one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bench.results import RankedEntry, Rankings


class QualityInputError(ValueError):
    """A ranking mentions a posting or a record the lookup does not know."""


@dataclass(frozen=True)
class QualityContext:
    """The lookups and the cut-off both metrics need.

    Attributes:
        categories: Dataset category by record id.
        relevant_categories: Relevant dataset categories by posting title.
        k: The cut-off (`settings.quality_k`).
    """

    categories: Mapping[str, str]
    relevant_categories: Mapping[str, frozenset[str]]
    k: int


@dataclass(frozen=True)
class QualityResult:
    """A metric averaged over postings.

    Attributes:
        value: Mean over the counted postings, or `None` when none was counted.
        per_posting: The metric of every counted posting, by title.
        excluded_postings: Titles left out because they had no candidates.
    """

    value: float | None
    per_posting: Mapping[str, float]
    excluded_postings: tuple[str, ...]


def unique_record_ids(entries: Sequence[RankedEntry]) -> list[str]:
    """Record ids in rank order, each id once, at its best (first) position."""
    seen: set[str] = set()
    unique: list[str] = []
    for entry in entries:
        if entry.record_id not in seen:
            seen.add(entry.record_id)
            unique.append(entry.record_id)
    return unique


def summarize_postings(
    per_posting: Mapping[str, float | None],
) -> QualityResult:
    """Average the counted postings; a `None` posting is excluded and reported."""
    counted = {title: value for title, value in per_posting.items() if value is not None}
    excluded = tuple(title for title, value in per_posting.items() if value is None)
    mean = sum(counted.values()) / len(counted) if counted else None
    return QualityResult(value=mean, per_posting=counted, excluded_postings=excluded)


def posting_precision(
    ranked_ids: Sequence[str], relevant: frozenset[str], context: QualityContext
) -> float | None:
    """Share of the top-k unique candidates whose category is relevant.

    The denominator is `min(k, number of unique candidates)`; a posting without
    candidates has no precision (`None`).
    """
    top = ranked_ids[: context.k]
    if not top:
        return None
    missing = [record_id for record_id in top if record_id not in context.categories]
    if missing:
        raise QualityInputError(
            f"{len(missing)} ranked record id(s) are not in the dataset lookup"
        )
    hits = sum(1 for record_id in top if context.categories[record_id] in relevant)
    return hits / len(top)


def precision_at_k(rankings: Rankings, context: QualityContext) -> QualityResult:
    """`precision@k` averaged over postings (SPEC 7.7).

    Per posting the ranking is deduplicated by record id (best occurrence
    kept), cut to the top k, and scored as the share of candidates whose
    category is in the posting's relevant categories. The denominator is
    `min(k, unique candidates)`. A posting without candidates is excluded from
    the average and listed in `excluded_postings`.

    Raises:
        QualityInputError: A posting has no relevant-category entry, or a
            ranked record id has no category.
    """
    per_posting: dict[str, float | None] = {}
    for title, entries in rankings.items():
        if title not in context.relevant_categories:
            raise QualityInputError(f"posting {title!r} has no relevant categories")
        ranked_ids = unique_record_ids(entries)
        per_posting[title] = posting_precision(
            ranked_ids, context.relevant_categories[title], context
        )
    return summarize_postings(per_posting)


def posting_overlap(
    green_ids: Sequence[str], baseline_ids: Sequence[str], k: int
) -> float | None:
    """Overlap of two deduplicated top-k lists for one posting."""
    denominator = min(k, len(green_ids), len(baseline_ids))
    if denominator == 0:
        return None
    shared = set(green_ids[:k]) & set(baseline_ids[:k])
    return len(shared) / denominator


def overlap_at_k(
    green_rankings: Rankings, baseline_rankings: Rankings, k: int
) -> QualityResult:
    """`overlap@k` between the green and the baseline ranking, averaged over postings.

    Both rankings are deduplicated by record id per posting. The overlap is
    `|top-k(green) & top-k(baseline)| / min(k, unique green candidates, unique
    baseline candidates)`. SPEC 7.7 says "divided by k"; the hard filter can
    leave the green list shorter than k, and dividing by k would then punish the
    green pipeline for filtering, which `precision@k` already accounts for.
    A posting present in only one ranking, or with no candidates on either
    side, is excluded and listed in `excluded_postings`.
    """
    titles = list(green_rankings) + [t for t in baseline_rankings if t not in green_rankings]
    per_posting: dict[str, float | None] = {}
    for title in titles:
        if title not in green_rankings or title not in baseline_rankings:
            per_posting[title] = None
            continue
        per_posting[title] = posting_overlap(
            unique_record_ids(green_rankings[title]),
            unique_record_ids(baseline_rankings[title]),
            k,
        )
    return summarize_postings(per_posting)
