"""`python -m bench parse-report`: aggregate statistics of the CV parser.

Runs the real ingest steps (`extract_text`, contact removal, `parse_cv`) over a
seeded sample of dataset PDFs and prints only aggregates. No CV text, name,
email or phone is ever printed, logged or stored (KVKK).
"""

import argparse
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.config import DEFAULT_BENCH_CV_REFERENCE_DATE
from app.ingest.contact import extract_contact, remove_contact
from app.ingest.pdf import PdfIngestError, extract_text
from app.ingest.sections import parse_cv
from bench.dataset import (
    DEFAULT_DATASET_ROOT,
    DatasetStructureError,
    ResumeRecord,
    iter_resume_records,
    sample_stratified,
)

PARSE_REPORT_COMMAND = "parse-report"
PARSE_REPORT_HELP = "Print aggregate parser statistics for a seeded sample of CVs."
HANDLER_DESTINATION = "handler"
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42
MIN_MULTI_TYPE_COUNT = 2
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90
PERCENT = 100.0
TYPE_COLUMN_WIDTH = 16
ERROR_PREFIX = "Parse report error: "
EXIT_OK = 0
EXIT_ERROR = 1


@dataclass(frozen=True)
class CvOutcome:
    """The aggregate-only facts kept about one parsed CV."""

    section_types: tuple[str, ...]
    no_headings_found: bool
    years_experience: float | None
    language_count: int


@dataclass(frozen=True)
class SampleSpec:
    """How the sample was drawn and which date `Current` resolves to."""

    size: int
    seed: int
    reference_date: date


@dataclass(frozen=True)
class ParseReport:
    """Aggregates over all parsed CVs of a sample."""

    parsed_count: int
    failed_count: int
    sections_by_type: Counter
    cvs_by_type: Counter
    no_headings_share: float
    multi_type_share: float
    years_found_share: float
    languages_found_share: float
    sections_per_cv: tuple[float, float, float]
    median_years: float | None


def summarize_cv(pdf_bytes: bytes, reference_date: date) -> CvOutcome:
    """Run extract, contact removal and parsing; keep aggregates only."""
    text = extract_text(pdf_bytes)
    parsed = parse_cv(remove_contact(text, extract_contact(text)), reference_date)
    return CvOutcome(
        section_types=tuple(section.type.value for section in parsed.sections),
        no_headings_found=parsed.no_headings_found,
        years_experience=parsed.years_experience,
        language_count=len(parsed.languages),
    )


def collect_outcomes(
    records: Sequence[ResumeRecord], reference_date: date
) -> tuple[list[CvOutcome], int]:
    """Summarize each record; count (never describe) extraction failures."""
    outcomes: list[CvOutcome] = []
    failed_count = 0
    for record in records:
        try:
            outcomes.append(summarize_cv(record.pdf_path.read_bytes(), reference_date))
        except PdfIngestError:
            failed_count += 1
    return outcomes, failed_count


def share(count: int, total: int) -> float:
    """Return count / total as a percentage; 0 for an empty total."""
    return PERCENT * count / total if total else 0.0


def percentile(sorted_values: list[float], percent: int) -> float:
    """Linear-interpolated percentile of a non-empty ascending list."""
    position = (len(sorted_values) - 1) * percent / PERCENT
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    gap = sorted_values[upper] - sorted_values[lower]
    return sorted_values[lower] + gap * (position - lower)


def section_count_spread(outcomes: list[CvOutcome]) -> tuple[float, float, float]:
    """Return (median, p10, p90) of sections per CV."""
    counts = sorted(float(len(outcome.section_types)) for outcome in outcomes)
    if not counts:
        return 0.0, 0.0, 0.0
    return (
        statistics.median(counts),
        percentile(counts, LOW_PERCENTILE),
        percentile(counts, HIGH_PERCENTILE),
    )


def build_report(outcomes: list[CvOutcome], failed_count: int) -> ParseReport:
    """Aggregate per-CV outcomes into the report."""
    total = len(outcomes)
    years = [o.years_experience for o in outcomes if o.years_experience is not None]
    multi_type = sum(len(set(o.section_types)) >= MIN_MULTI_TYPE_COUNT for o in outcomes)
    return ParseReport(
        parsed_count=total,
        failed_count=failed_count,
        sections_by_type=Counter(t for o in outcomes for t in o.section_types),
        cvs_by_type=Counter(t for o in outcomes for t in set(o.section_types)),
        no_headings_share=share(sum(o.no_headings_found for o in outcomes), total),
        multi_type_share=share(multi_type, total),
        years_found_share=share(len(years), total),
        languages_found_share=share(sum(o.language_count > 0 for o in outcomes), total),
        sections_per_cv=section_count_spread(outcomes),
        median_years=statistics.median(years) if years else None,
    )


def format_report(report: ParseReport, spec: SampleSpec) -> str:
    """Render the report as plain text."""
    median, low, high = report.sections_per_cv
    lines = [
        f"Parse report: n={spec.size} seed={spec.seed} reference_date={spec.reference_date}",
        f"CVs parsed: {report.parsed_count} (extraction failures: {report.failed_count})",
        "Section type distribution (sections total / CVs with the type):",
    ]
    for section_type in sorted(report.sections_by_type):
        lines.append(
            f"  {section_type:<{TYPE_COLUMN_WIDTH}}"
            f"{report.sections_by_type[section_type]:>6} / {report.cvs_by_type[section_type]}"
        )
    median_years = "n/a" if report.median_years is None else f"{report.median_years:.1f}"
    lines += [
        f"CVs without headings: {report.no_headings_share:.1f}%",
        f"CVs with 2+ section types: {report.multi_type_share:.1f}%",
        f"CVs with years_experience: {report.years_found_share:.1f}%",
        f"CVs with languages: {report.languages_found_share:.1f}%",
        f"Sections per CV: median {median:.1f}, p10 {low:.1f}, p90 {high:.1f}",
        f"Median years_experience (where found): {median_years}",
    ]
    return "\n".join(lines)


def run_parse_report(arguments: argparse.Namespace) -> int:
    """Handle `parse-report`: print aggregates or a clean error."""
    try:
        records = sample_stratified(
            iter_resume_records(arguments.root), arguments.n, arguments.seed
        )
    except (DatasetStructureError, ValueError) as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
    outcomes, failed_count = collect_outcomes(records, arguments.reference_date)
    report = build_report(outcomes, failed_count)
    spec = SampleSpec(arguments.n, arguments.seed, arguments.reference_date)
    print(format_report(report, spec))
    return EXIT_OK


def register_parse_report(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the `parse-report` subcommand to the parser."""
    parser = subparsers.add_parser(PARSE_REPORT_COMMAND, help=PARSE_REPORT_HELP)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    parser.add_argument(
        "--reference-date",
        type=date.fromisoformat,
        default=DEFAULT_BENCH_CV_REFERENCE_DATE,
        help="Date that Current ends are resolved to (default: %(default)s).",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.set_defaults(**{HANDLER_DESTINATION: run_parse_report})
