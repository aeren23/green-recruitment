"""Tests for the parse-report aggregates and CLI, on synthetic PDFs only."""

from datetime import date
from pathlib import Path

import pytest

from bench.__main__ import EXIT_ERROR, EXIT_OK, main
from bench.dataset import ResumeRecord
from bench.parse_report import (
    CvOutcome,
    SampleSpec,
    build_report,
    collect_outcomes,
    format_report,
    percentile,
    summarize_cv,
)
from pdf_fixtures import build_pdf
from test_dataset import ALL_COLUMNS, SyntheticResume, write_dataset

REFERENCE = date(2018, 1, 1)
MARKER_NAME = "Jane Roe"
MARKER_EMAIL = "jane.roe@example.org"
SYNTHETIC_CV_LINES = (
    MARKER_NAME,
    MARKER_EMAIL,
    "SUMMARY",
    "Careful accountant who closes the books on time and reconciles every account.",
    "EXPERIENCE",
    "Clerk",
    "01/2012",
    "to",
    "01/2014",
    "Posted journal entries and prepared monthly statements for the finance team.",
    "SKILLS",
    "Excel, ledgers, reconciliation, payroll",
)


def outcome(types: tuple[str, ...], years: float | None, headings: bool = True) -> CvOutcome:
    """Build an outcome with the given section types."""
    return CvOutcome(
        section_types=types,
        no_headings_found=not headings,
        years_experience=years,
        language_count=0,
    )


def test_percentile_interpolates_linearly() -> None:
    """p10 and p90 of 1..11 are 2 and 10."""
    values = [float(value) for value in range(1, 12)]
    assert percentile(values, 10) == 2.0
    assert percentile(values, 90) == 10.0
    assert percentile([5.0], 90) == 5.0


def test_build_report_aggregates_shares_and_distribution() -> None:
    """Shares, per-type counts and section spread come out as expected."""
    report = build_report(
        [
            outcome(("summary", "skills", "experience", "experience"), 4.0),
            outcome(("summary",), None, headings=False),
        ],
        failed_count=1,
    )
    assert report.parsed_count == 2 and report.failed_count == 1
    assert report.sections_by_type["experience"] == 2
    assert report.cvs_by_type["experience"] == 1
    assert report.no_headings_share == 50.0
    assert report.multi_type_share == 50.0
    assert report.years_found_share == 50.0
    assert report.median_years == 4.0
    assert report.sections_per_cv[0] == 2.5


def test_build_report_handles_an_empty_sample() -> None:
    """No parsed CV gives zeros, not a crash."""
    report = build_report([], failed_count=0)
    assert report.multi_type_share == 0.0
    assert report.median_years is None


def test_summarize_cv_runs_the_real_chain_and_keeps_aggregates_only() -> None:
    """A synthetic PDF goes through extract, scrub and parse."""
    result = summarize_cv(build_pdf(SYNTHETIC_CV_LINES), REFERENCE)
    assert set(result.section_types) == {"summary", "experience", "skills"}
    assert result.years_experience == 2.0
    assert not result.no_headings_found
    assert MARKER_NAME not in repr(result)


def test_collect_outcomes_counts_unreadable_pdfs(tmp_path: Path) -> None:
    """A PDF that cannot be read is counted, not described."""
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    good.write_bytes(build_pdf(SYNTHETIC_CV_LINES))
    bad.write_bytes(b"not a pdf")
    records = [
        ResumeRecord(id="1", category="HR", pdf_path=good, resume_str=""),
        ResumeRecord(id="2", category="HR", pdf_path=bad, resume_str=""),
    ]
    outcomes, failed_count = collect_outcomes(records, REFERENCE)
    assert len(outcomes) == 1 and failed_count == 1


def test_format_report_prints_no_cv_text() -> None:
    """The printed report holds only labels and numbers."""
    result = summarize_cv(build_pdf(SYNTHETIC_CV_LINES), REFERENCE)
    text = format_report(build_report([result], 0), SampleSpec(1, 42, REFERENCE))
    assert "n=1 seed=42 reference_date=2018-01-01" in text
    assert "experience" in text and "CVs with 2+ section types: 100.0%" in text
    for marker in (MARKER_NAME, MARKER_EMAIL, "journal"):
        assert marker not in text


def test_cli_prints_aggregates_for_a_synthetic_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`parse-report` runs end to end and exits cleanly."""
    categories = ("INFORMATION-TECHNOLOGY", "ACCOUNTANT", "HR", "ARTS")
    resumes = [SyntheticResume(f"{c[:2]}{i}", c) for c in categories for i in range(2)]
    write_dataset(tmp_path, resumes, ALL_COLUMNS)
    for resume in resumes:
        pdf_path = tmp_path / "data" / "data" / resume.category / f"{resume.resume_id}.pdf"
        pdf_path.write_bytes(build_pdf(SYNTHETIC_CV_LINES))
    exit_code = main(["parse-report", "--root", str(tmp_path), "--n", "4", "--seed", "1"])
    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert "CVs parsed: 4 (extraction failures: 0)" in output
    assert "reference_date=2018-01-01" in output
    assert MARKER_NAME not in output


def test_cli_reports_a_missing_dataset_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wrong root is a one-line error and a non-zero exit code."""
    exit_code = main(["parse-report", "--root", str(tmp_path / "missing")])
    assert exit_code == EXIT_ERROR
    assert capsys.readouterr().err.startswith("Parse report error: ")
