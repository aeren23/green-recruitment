"""Reader, statistics and stratified sampling for the Kaggle Resume Dataset.

Expected layout under the dataset root (SPEC section 7.1):

    Resume/Resume.csv                 columns ID, Resume_str, Resume_html, Category
    data/data/<CATEGORY>/<ID>.pdf     one PDF per resume

Only the standard library is used on purpose: the benchmark must not pull in
heavy dependencies just to read a CSV file.
"""

import csv
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPOSITORY_ROOT / "data" / "raw"

RESUME_CSV_RELATIVE_PATH = Path("Resume") / "Resume.csv"
PDF_DIRECTORY_RELATIVE_PATH = Path("data") / "data"
PDF_FILE_SUFFIX = ".pdf"
CSV_ENCODING = "utf-8"

COLUMN_ID = "ID"
COLUMN_RESUME_TEXT = "Resume_str"
COLUMN_RESUME_HTML = "Resume_html"
COLUMN_CATEGORY = "Category"
REQUIRED_COLUMNS = (COLUMN_ID, COLUMN_RESUME_TEXT, COLUMN_RESUME_HTML, COLUMN_CATEGORY)

# Resume_html cells are far larger than the csv default of 131072 characters.
# sys.maxsize overflows the C long used by csv on Windows, so use a fixed value.
CSV_FIELD_SIZE_LIMIT = 2**31 - 1

CATEGORY_INFORMATION_TECHNOLOGY = "INFORMATION-TECHNOLOGY"
CATEGORY_ACCOUNTANT = "ACCOUNTANT"
CATEGORY_FINANCE = "FINANCE"
CATEGORY_HR = "HR"

STRATUM_NAME_INFORMATION_TECHNOLOGY = "information technology"
STRATUM_NAME_ACCOUNTING_AND_FINANCE = "accountant and finance"
STRATUM_NAME_HUMAN_RESOURCES = "human resources"
STRATUM_NAME_DISTRACTORS = "distractors"

SAMPLE_STRATA_COUNT = 4
FOCUSED_STRATA_COUNT = SAMPLE_STRATA_COUNT - 1
DIRECTORY_LISTING_LIMIT = 10


class DatasetStructureError(Exception):
    """The dataset on disk does not match the layout described in SPEC 7.1."""


@dataclass(frozen=True)
class ResumeRecord:
    """One resume of the dataset.

    `pdf_path` is the expected location of the PDF and may not exist on disk.
    """

    id: str
    category: str
    pdf_path: Path
    resume_str: str

    def has_pdf(self) -> bool:
        """Return True when the PDF file exists on disk."""
        return self.pdf_path.is_file()


@dataclass(frozen=True)
class DatasetStats:
    """Summary of a dataset: category distribution and PDF availability."""

    category_distribution: dict[str, int]
    total_records: int
    pdf_count: int
    missing_pdf_count: int


@dataclass(frozen=True)
class Stratum:
    """A named group of categories that is sampled as one pool."""

    name: str
    categories: frozenset[str]


FOCUSED_STRATA = (
    Stratum(
        STRATUM_NAME_INFORMATION_TECHNOLOGY,
        frozenset({CATEGORY_INFORMATION_TECHNOLOGY}),
    ),
    Stratum(
        STRATUM_NAME_ACCOUNTING_AND_FINANCE,
        frozenset({CATEGORY_ACCOUNTANT, CATEGORY_FINANCE}),
    ),
    Stratum(STRATUM_NAME_HUMAN_RESOURCES, frozenset({CATEGORY_HR})),
)
FOCUSED_CATEGORIES = frozenset().union(*(stratum.categories for stratum in FOCUSED_STRATA))


def describe_directory(directory: Path) -> str:
    """Describe what a directory contains, for use in error messages."""
    if not directory.is_dir():
        return f"{directory} does not exist"
    entry_names = sorted(entry.name for entry in directory.iterdir())
    if not entry_names:
        return f"{directory} is empty"
    shown = ", ".join(entry_names[:DIRECTORY_LISTING_LIMIT])
    suffix = ", ..." if len(entry_names) > DIRECTORY_LISTING_LIMIT else ""
    return f"{directory} contains: {shown}{suffix}"


def read_csv_header(csv_path: Path) -> list[str]:
    """Return the header row of the resume CSV (empty list for an empty file)."""
    with csv_path.open(encoding=CSV_ENCODING, newline="") as csv_file:
        return next(csv.reader(csv_file), [])


def validate_dataset_layout(dataset_root: Path) -> tuple[Path, Path]:
    """Check the dataset layout and return the CSV path and the PDF directory.

    Args:
        dataset_root: Directory that holds `Resume/` and `data/data/`.

    Raises:
        DatasetStructureError: The CSV or PDF directory is missing, or the CSV
            lacks a required column.
    """
    csv_path = dataset_root / RESUME_CSV_RELATIVE_PATH
    pdf_directory = dataset_root / PDF_DIRECTORY_RELATIVE_PATH
    if not csv_path.is_file():
        raise DatasetStructureError(
            f"Expected resume CSV at {csv_path}, but it was not found. "
            f"Dataset root: {describe_directory(dataset_root)}."
        )
    if not pdf_directory.is_dir():
        raise DatasetStructureError(
            f"Expected PDF directory at {pdf_directory}, but it was not found. "
            f"Dataset root: {describe_directory(dataset_root)}."
        )
    header = read_csv_header(csv_path)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing_columns:
        raise DatasetStructureError(
            f"Resume CSV {csv_path} lacks required column(s) {missing_columns}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}; found: {header}."
        )
    return csv_path, pdf_directory


def read_records(csv_path: Path, pdf_directory: Path) -> Iterator[ResumeRecord]:
    """Lazily yield one record per CSV row, dropping the heavy HTML column."""
    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    with csv_path.open(encoding=CSV_ENCODING, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            category = row[COLUMN_CATEGORY]
            resume_id = row[COLUMN_ID]
            yield ResumeRecord(
                id=resume_id,
                category=category,
                pdf_path=pdf_directory / category / f"{resume_id}{PDF_FILE_SUFFIX}",
                resume_str=row[COLUMN_RESUME_TEXT],
            )


def iter_resume_records(dataset_root: Path = DEFAULT_DATASET_ROOT) -> Iterator[ResumeRecord]:
    """Lazily iterate over the resumes of a dataset.

    The layout is validated immediately (not on the first `next()`), so a
    wrong structure fails fast. `Resume_html` is never retained.

    Args:
        dataset_root: Directory that holds `Resume/` and `data/data/`.

    Raises:
        DatasetStructureError: The dataset layout differs from SPEC 7.1.
    """
    csv_path, pdf_directory = validate_dataset_layout(dataset_root)
    return read_records(csv_path, pdf_directory)


def compute_dataset_stats(records: Iterable[ResumeRecord]) -> DatasetStats:
    """Count records per category and how many PDFs exist on disk."""
    distribution: Counter[str] = Counter()
    pdf_count = 0
    for record in records:
        distribution[record.category] += 1
        pdf_count += record.has_pdf()
    total_records = sum(distribution.values())
    return DatasetStats(
        category_distribution=dict(sorted(distribution.items())),
        total_records=total_records,
        pdf_count=pdf_count,
        missing_pdf_count=total_records - pdf_count,
    )


def stratum_sizes(sample_size: int) -> tuple[int, int]:
    """Return (size of each focused stratum, size of the distractor stratum)."""
    focused_size = sample_size // SAMPLE_STRATA_COUNT
    return focused_size, sample_size - focused_size * FOCUSED_STRATA_COUNT


def require_pool_size(pool: list[ResumeRecord], count: int, stratum_name: str) -> None:
    """Fail clearly when a stratum has fewer eligible records than requested."""
    if len(pool) < count:
        raise ValueError(
            f"Stratum '{stratum_name}' needs {count} resumes with a PDF "
            f"but only {len(pool)} are available."
        )


def sample_stratified(
    records: Iterable[ResumeRecord], sample_size: int, seed: int
) -> list[ResumeRecord]:
    """Draw a stratified, reproducible sample (SPEC 7.3).

    One quarter IT, one quarter ACCOUNTANT plus FINANCE (one combined pool),
    one quarter HR; the remainder (including `sample_size % 4`) are distractors
    from all other categories. Only records whose PDF exists are eligible. The
    result depends only on the eligible set and the seed, not on input order.

    Args:
        records: Candidate records.
        sample_size: Total number of records to return; must be positive.
        seed: Seed of the local random generator.

    Raises:
        ValueError: `sample_size` is not positive or a stratum is too small.
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")
    eligible = sorted(
        (record for record in records if record.has_pdf()),
        key=lambda record: (record.id, record.category),
    )
    rng = random.Random(seed)
    focused_size, distractor_size = stratum_sizes(sample_size)
    sample: list[ResumeRecord] = []
    for stratum in FOCUSED_STRATA:
        pool = [record for record in eligible if record.category in stratum.categories]
        require_pool_size(pool, focused_size, stratum.name)
        sample += rng.sample(pool, focused_size)
    distractors = [
        record for record in eligible if record.category not in FOCUSED_CATEGORIES
    ]
    require_pool_size(distractors, distractor_size, STRATUM_NAME_DISTRACTORS)
    sample += rng.sample(distractors, distractor_size)
    return sample
