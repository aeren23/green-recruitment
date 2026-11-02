"""The pipeline-specific `measurement_notes` fields both runners write.

Sharing one builder keeps the key set identical, so a green run and a baseline
run can be compared field by field. Aggregate counts only, never CV content.
"""

from dataclasses import dataclass

from bench.results import WorkloadCounts

NO_ELIMINATIONS = "none"
REASON_SEPARATOR = ","
REASON_COUNT_FORMAT = "{reason}:{count}"


@dataclass(frozen=True)
class PipelineNoteFacts:
    """Counters that do not live in `WorkloadCounts`.

    Attributes:
        embeddings_computed: Embeddings computed (0 for the baseline).
        records_deleted: Records removed by retention (0 for the baseline).
        llm_failures: LLM tasks that ended without a usable result.
    """

    embeddings_computed: int
    records_deleted: int
    llm_failures: int


def format_eliminations(counts: WorkloadCounts) -> str:
    """`MIN_YEARS:3,...` for the notes, or `none`."""
    if not counts.eliminated_by_reason:
        return NO_ELIMINATIONS
    return REASON_SEPARATOR.join(
        REASON_COUNT_FORMAT.format(reason=reason, count=count)
        for reason, count in sorted(counts.eliminated_by_reason.items())
    )


def build_pipeline_notes(
    counts: WorkloadCounts, facts: PipelineNoteFacts
) -> dict[str, str]:
    """Aggregate counts for `measurement_notes`, in the order both runners use."""
    return {
        "applications": str(counts.submitted),
        "ranked": str(counts.ranked),
        "duplicates_blocked": str(counts.blocked),
        "unreadable": str(counts.unreadable),
        "eliminated_by_reason": format_eliminations(counts),
        "embeddings_computed": str(facts.embeddings_computed),
        "records_deleted": str(facts.records_deleted),
        "llm_failures": str(facts.llm_failures),
    }
