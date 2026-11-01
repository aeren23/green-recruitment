"""Tests for the dataset reader, statistics, stratified sampling and CLI.

Every test builds a small synthetic dataset in `tmp_path`; the real Kaggle
dataset is not needed.
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from bench.__main__ import EXIT_ERROR, EXIT_OK, main
from bench.dataset import (
    CATEGORY_ACCOUNTANT,
    CATEGORY_FINANCE,
    CATEGORY_HR,
    CATEGORY_INFORMATION_TECHNOLOGY,
    COLUMN_CATEGORY,
    COLUMN_ID,
    COLUMN_RESUME_HTML,
    COLUMN_RESUME_TEXT,
    CSV_FIELD_SIZE_LIMIT,
    DatasetStructureError,
    ResumeRecord,
    compute_dataset_stats,
    iter_resume_records,
    sample_stratified,
)

CSV_DEFAULT_FIELD_LIMIT = 131072
OVERSIZED_HTML_LENGTH = CSV_DEFAULT_FIELD_LIMIT + 1000
ALL_COLUMNS = [COLUMN_ID, COLUMN_RESUME_TEXT, COLUMN_RESUME_HTML, COLUMN_CATEGORY]
DISTRACTOR_CATEGORIES = ("AGRICULTURE", "ARTS")
FOCUSED_CATEGORIES = (
    CATEGORY_INFORMATION_TECHNOLOGY,
    CATEGORY_ACCOUNTANT,
    CATEGORY_FINANCE,
    CATEGORY_HR,
)
ACCOUNTING_CATEGORIES = {CATEGORY_ACCOUNTANT, CATEGORY_FINANCE}
PER_CATEGORY_COUNT = 12
SAMPLE_SIZE_DIVISIBLE = 16
SAMPLE_SIZE_WITH_REMAINDER = 18
SEED = 42
OTHER_SEED = 43
SHUFFLE_SEED = 7
MISSING_PDF_SUFFIX = "-missing"


@dataclass(frozen=True)
class SyntheticResume:
    """A row of the synthetic CSV and whether its PDF exists."""

    resume_id: str
    category: str
    has_pdf: bool = True
    html: str = "<p>x</p>"


def write_dataset(root: Path, resumes: list[SyntheticResume], columns: list[str]) -> None:
    """Write `Resume/Resume.csv` and placeholder PDFs under `root`."""
    (root / "Resume").mkdir(parents=True)
    (root / "data" / "data").mkdir(parents=True)
    with (root / "Resume" / "Resume.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for resume in resumes:
            writer.writerow(
                {
                    COLUMN_ID: resume.resume_id,
                    COLUMN_RESUME_TEXT: f"text of {resume.resume_id}",
                    COLUMN_RESUME_HTML: resume.html,
                    COLUMN_CATEGORY: resume.category,
                }
            )
            if resume.has_pdf:
                category_directory = root / "data" / "data" / resume.category
                category_directory.mkdir(exist_ok=True)
                (category_directory / f"{resume.resume_id}.pdf").write_bytes(b"%PDF-placeholder")


def build_full_resume_list() -> list[SyntheticResume]:
    """Resumes of every category: enough for any sample used in these tests."""
    categories = FOCUSED_CATEGORIES + DISTRACTOR_CATEGORIES
    return [
        SyntheticResume(f"{category[:3]}{index:03d}", category)
        for category in categories
        for index in range(PER_CATEGORY_COUNT)
    ]


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A complete synthetic dataset."""
    root = tmp_path / "raw"
    write_dataset(root, build_full_resume_list(), ALL_COLUMNS)
    return root


@pytest.fixture
def all_records(dataset_root: Path) -> list[ResumeRecord]:
    """Every record of the complete synthetic dataset."""
    return list(iter_resume_records(dataset_root))


def category_counts(sample: list[ResumeRecord]) -> Counter[str]:
    """Count sampled records per category."""
    return Counter(record.category for record in sample)


def test_records_map_csv_fields_and_expected_pdf_path(tmp_path: Path) -> None:
    """Each CSV row becomes a record with the documented PDF path."""
    write_dataset(tmp_path, [SyntheticResume("123", CATEGORY_HR)], ALL_COLUMNS)
    records = list(iter_resume_records(tmp_path))
    assert records == [
        ResumeRecord(
            id="123",
            category=CATEGORY_HR,
            pdf_path=tmp_path / "data" / "data" / CATEGORY_HR / "123.pdf",
            resume_str="text of 123",
        )
    ]
    assert records[0].pdf_path.is_file()


def test_pdf_path_is_reported_even_when_file_is_missing(tmp_path: Path) -> None:
    """A record whose PDF is absent still carries the expected path."""
    write_dataset(tmp_path, [SyntheticResume("9", CATEGORY_HR, has_pdf=False)], ALL_COLUMNS)
    record = next(iter_resume_records(tmp_path))
    assert record.pdf_path == tmp_path / "data" / "data" / CATEGORY_HR / "9.pdf"
    assert not record.pdf_path.exists()


def test_record_does_not_retain_html(tmp_path: Path) -> None:
    """The heavy Resume_html column is not part of the record."""
    write_dataset(tmp_path, [SyntheticResume("1", CATEGORY_HR, html="<b>secret</b>")], ALL_COLUMNS)
    record = next(iter_resume_records(tmp_path))
    assert "secret" not in repr(record)
    assert not hasattr(record, "resume_html")


def test_html_longer_than_default_csv_limit_is_readable(tmp_path: Path) -> None:
    """A Resume_html cell above 131072 characters reads without error."""
    oversized = SyntheticResume("1", CATEGORY_HR, html="x" * OVERSIZED_HTML_LENGTH)
    write_dataset(tmp_path, [oversized], ALL_COLUMNS)
    assert [record.id for record in iter_resume_records(tmp_path)] == ["1"]


def test_field_size_limit_constant_fits_a_32_bit_long() -> None:
    """The limit must not overflow the C long that csv uses on Windows."""
    assert CSV_FIELD_SIZE_LIMIT <= 2**31 - 1


def test_missing_csv_raises_structure_error(tmp_path: Path) -> None:
    """A dataset root without the CSV is reported with expected and found."""
    with pytest.raises(DatasetStructureError, match="Resume.csv"):
        iter_resume_records(tmp_path)


def test_missing_pdf_directory_raises_structure_error(tmp_path: Path) -> None:
    """The CSV alone is not enough: the PDF directory must exist."""
    csv_directory = tmp_path / "Resume"
    csv_directory.mkdir()
    (csv_directory / "Resume.csv").write_text(",".join(ALL_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(DatasetStructureError, match="PDF directory"):
        iter_resume_records(tmp_path)


def test_missing_column_raises_structure_error(tmp_path: Path) -> None:
    """A CSV without a required column names the missing and found columns."""
    write_dataset(tmp_path, [SyntheticResume("1", CATEGORY_HR)], [COLUMN_ID, COLUMN_CATEGORY])
    with pytest.raises(DatasetStructureError) as error_info:
        iter_resume_records(tmp_path)
    message = str(error_info.value)
    assert COLUMN_RESUME_TEXT in message
    assert COLUMN_RESUME_HTML in message


def test_stats_count_categories_pdfs_and_missing_pdfs(tmp_path: Path) -> None:
    """Stats report the distribution, PDF count and missing PDF count."""
    resumes = [
        SyntheticResume("1", CATEGORY_HR),
        SyntheticResume("2", CATEGORY_HR, has_pdf=False),
        SyntheticResume("3", CATEGORY_FINANCE),
    ]
    write_dataset(tmp_path, resumes, ALL_COLUMNS)
    stats = compute_dataset_stats(iter_resume_records(tmp_path))
    assert stats.category_distribution == {CATEGORY_FINANCE: 1, CATEGORY_HR: 2}
    assert stats.total_records == 3
    assert stats.pdf_count == 2
    assert stats.missing_pdf_count == 1


def test_same_seed_gives_same_sample(all_records: list[ResumeRecord]) -> None:
    """Sampling is reproducible for a fixed seed."""
    first = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, SEED)
    second = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, SEED)
    assert first == second


def test_different_seed_gives_different_sample(all_records: list[ResumeRecord]) -> None:
    """Another seed picks another sample from a pool larger than the quota."""
    first = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, SEED)
    second = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, OTHER_SEED)
    assert {record.id for record in first} != {record.id for record in second}


def test_input_order_does_not_change_sample(all_records: list[ResumeRecord]) -> None:
    """Shuffling the input leaves the sample unchanged."""
    shuffled = list(all_records)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    assert shuffled != all_records
    expected = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, SEED)
    assert sample_stratified(shuffled, SAMPLE_SIZE_DIVISIBLE, SEED) == expected


def test_strata_counts_when_size_is_divisible_by_four(all_records: list[ResumeRecord]) -> None:
    """N=16 gives 4 IT, 4 accountant/finance, 4 HR and 4 distractors."""
    sample = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, SEED)
    counts = category_counts(sample)
    quarter = SAMPLE_SIZE_DIVISIBLE // 4
    assert len(sample) == SAMPLE_SIZE_DIVISIBLE
    assert counts[CATEGORY_INFORMATION_TECHNOLOGY] == quarter
    assert counts[CATEGORY_HR] == quarter
    assert sum(counts[name] for name in ACCOUNTING_CATEGORIES) == quarter
    assert sum(counts[name] for name in DISTRACTOR_CATEGORIES) == quarter


def test_remainder_goes_to_distractors(all_records: list[ResumeRecord]) -> None:
    """N=18 gives 4 per focused stratum and 6 distractors."""
    sample = sample_stratified(all_records, SAMPLE_SIZE_WITH_REMAINDER, SEED)
    counts = category_counts(sample)
    quarter = SAMPLE_SIZE_WITH_REMAINDER // 4
    assert len(sample) == SAMPLE_SIZE_WITH_REMAINDER
    assert counts[CATEGORY_INFORMATION_TECHNOLOGY] == quarter
    assert counts[CATEGORY_HR] == quarter
    assert sum(counts[name] for name in ACCOUNTING_CATEGORIES) == quarter
    distractors = sum(counts[name] for name in DISTRACTOR_CATEGORIES)
    assert distractors == SAMPLE_SIZE_WITH_REMAINDER - 3 * quarter


def test_accounting_pool_can_yield_both_categories(all_records: list[ResumeRecord]) -> None:
    """ACCOUNTANT and FINANCE share one pool, so both appear across seeds."""
    seen: set[str] = set()
    for seed in range(PER_CATEGORY_COUNT):
        sample = sample_stratified(all_records, SAMPLE_SIZE_DIVISIBLE, seed)
        seen |= {r.category for r in sample if r.category in ACCOUNTING_CATEGORIES}
    assert seen == ACCOUNTING_CATEGORIES


def test_records_without_pdf_are_never_sampled(tmp_path: Path) -> None:
    """Records whose PDF is missing are not eligible."""
    resumes = build_full_resume_list()
    resumes += [
        SyntheticResume(f"{category[:3]}{MISSING_PDF_SUFFIX}{index}", category, has_pdf=False)
        for category in FOCUSED_CATEGORIES + DISTRACTOR_CATEGORIES
        for index in range(PER_CATEGORY_COUNT)
    ]
    write_dataset(tmp_path, resumes, ALL_COLUMNS)
    records = list(iter_resume_records(tmp_path))
    for seed in range(PER_CATEGORY_COUNT):
        sample = sample_stratified(records, SAMPLE_SIZE_WITH_REMAINDER, seed)
        assert all(record.has_pdf() for record in sample)
        assert all(MISSING_PDF_SUFFIX not in record.id for record in sample)


def test_too_small_stratum_raises_value_error(all_records: list[ResumeRecord]) -> None:
    """Asking for more than a stratum holds fails with a clear message."""
    oversized = PER_CATEGORY_COUNT * 4 + 4
    with pytest.raises(ValueError, match="information technology"):
        sample_stratified(all_records, oversized, SEED)


@pytest.mark.parametrize("invalid_size", [0, -4])
def test_non_positive_sample_size_raises_value_error(
    all_records: list[ResumeRecord], invalid_size: int
) -> None:
    """A sample size of zero or less is rejected."""
    with pytest.raises(ValueError, match="positive"):
        sample_stratified(all_records, invalid_size, SEED)


def test_cli_prints_distribution_and_pdf_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`dataset-stats --root` prints the report and exits with 0."""
    resumes = [SyntheticResume("1", CATEGORY_HR), SyntheticResume("2", CATEGORY_HR, has_pdf=False)]
    write_dataset(tmp_path, resumes, ALL_COLUMNS)
    exit_code = main(["dataset-stats", "--root", str(tmp_path)])
    output = capsys.readouterr().out
    assert exit_code == EXIT_OK
    assert CATEGORY_HR in output
    assert "PDF count: 1" in output
    assert "Missing PDF count: 1" in output


def test_cli_reports_structure_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A wrong layout gives a non-zero exit code and a message on stderr."""
    exit_code = main(["dataset-stats", "--root", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == EXIT_ERROR
    assert "Resume.csv" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
