"""Source-scanning checks for the architecture rules (plan section 5.3).

The scanner walks `app/**/*.py` with plain regular expressions, so it needs no
extra dependency. The scanning helper is unit-tested against violating
sources, otherwise a check that can never fail would prove nothing.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
APP_DIRECTORY = REPOSITORY_ROOT / "app"

RETENTION_FILE = "app/privacy/retention.py"
EMBED_FILE = "app/scoring/embed.py"
LLM_FREE_DIRECTORIES = ("app/ingest/", "app/scoring/")
INGEST_DIRECTORY = "app/ingest/"
PIPELINE_DIRECTORY = "app/pipeline/"
WEB_DIRECTORY = "app/web/"
PIPELINE_MODULE = "app/pipeline/upload.py"
SUMMARY_MODULE = "app/pipeline/summaries.py"
DISK_WRITE_PATTERN = re.compile(
    r"\bopen\s*\(|write_bytes|write_text|\btempfile\b|NamedTemporaryFile"
)


@dataclass(frozen=True)
class Rule:
    """A forbidden source pattern and the files it applies to."""

    name: str
    forbidden_pattern: re.Pattern[str]
    applies_to: Callable[[str], bool]


def is_not_retention(relative_path: str) -> bool:
    """True for every file except the retention module."""
    return relative_path != RETENTION_FILE


def is_retention(relative_path: str) -> bool:
    """True only for the retention module."""
    return relative_path == RETENTION_FILE


def is_not_embed(relative_path: str) -> bool:
    """True for every file except the embedding module."""
    return relative_path != EMBED_FILE


def is_llm_free_directory(relative_path: str) -> bool:
    """True for files under `app/ingest/` and `app/scoring/`."""
    return relative_path.startswith(LLM_FREE_DIRECTORIES)


def is_ingest_directory(relative_path: str) -> bool:
    """True for files under `app/ingest/`."""
    return relative_path.startswith(INGEST_DIRECTORY)


def is_pipeline(relative_path: str) -> bool:
    """True for every module of the `app/pipeline/` package."""
    return relative_path.startswith(PIPELINE_DIRECTORY)


def is_web(relative_path: str) -> bool:
    """True for every module of the `app/web/` package."""
    return relative_path.startswith(WEB_DIRECTORY)


RULES = (
    Rule(
        name="deletion outside retention",
        forbidden_pattern=re.compile(
            r"session\.delete\(|\bdelete\s*\(|DELETE\s+FROM"
            r"|from\s+sqlalchemy[\w.]*\s+import[^\n]*\bdelete\b",
            re.IGNORECASE,
        ),
        applies_to=is_not_retention,
    ),
    Rule(
        name="datetime.now in retention",
        forbidden_pattern=re.compile(r"datetime\s*\.\s*now\s*\("),
        applies_to=is_retention,
    ),
    Rule(
        name="SentenceTransformer outside embed",
        forbidden_pattern=re.compile(r"SentenceTransformer\s*\("),
        applies_to=is_not_embed,
    ),
    Rule(
        name="llm import from ingest or scoring",
        forbidden_pattern=re.compile(
            r"^\s*(from\s+app\.llm\b|import\s+app\.llm\b"
            r"|from\s+app\s+import\s+[^\n]*\bllm\b|from\s+\.+llm\b)",
            re.MULTILINE,
        ),
        applies_to=is_llm_free_directory,
    ),
    Rule(
        name="disk write in ingest",
        forbidden_pattern=DISK_WRITE_PATTERN,
        applies_to=is_ingest_directory,
    ),
    Rule(
        name="disk write in pipeline",
        forbidden_pattern=DISK_WRITE_PATTERN,
        applies_to=is_pipeline,
    ),
    Rule(
        name="disk write in web",
        forbidden_pattern=DISK_WRITE_PATTERN,
        applies_to=is_web,
    ),
    # The green pipeline calls the LLM only through `summarize_candidate`
    # (the top-k summary); the baseline scoring task and raw client calls are
    # not allowed in it.
    Rule(
        name="baseline llm in green pipeline",
        forbidden_pattern=re.compile(
            r"score_cv_with_llm|BaselinePromptInput|build_baseline_prompt|\.generate\s*\("
        ),
        applies_to=is_pipeline,
    ),
)


def find_violations(relative_path: str, source: str) -> list[str]:
    """Return the names of the rules that the given source breaks.

    Args:
        relative_path: POSIX path relative to the repository root.
        source: Full text of the file.
    """
    return [
        rule.name
        for rule in RULES
        if rule.applies_to(relative_path) and rule.forbidden_pattern.search(source)
    ]


def scan_directory(directory: Path, root: Path) -> dict[str, list[str]]:
    """Scan every Python file under a directory and map file to broken rules.

    Args:
        directory: Directory to walk recursively.
        root: Root that reported paths are made relative to.
    """
    violations: dict[str, list[str]] = {}
    for python_file in directory.rglob("*.py"):
        relative_path = python_file.relative_to(root).as_posix()
        broken_rules = find_violations(
            relative_path, python_file.read_text(encoding="utf-8")
        )
        if broken_rules:
            violations[relative_path] = broken_rules
    return violations


def test_app_source_respects_architecture_rules() -> None:
    """No file under `app/` breaks any architecture rule."""
    assert scan_directory(APP_DIRECTORY, REPOSITORY_ROOT) == {}


@pytest.mark.parametrize(
    ("relative_path", "source", "expected_rule"),
    [
        ("app/web/routes.py", "session.delete(row)", "deletion outside retention"),
        ("app/models.py", "stmt = delete(Section)", "deletion outside retention"),
        ("app/models.py", "from sqlalchemy import delete", "deletion outside retention"),
        ("app/db.py", 'run("delete from sections")', "deletion outside retention"),
        (RETENTION_FILE, "now = datetime.now()", "datetime.now in retention"),
        ("app/pipeline/scoring.py", "SentenceTransformer(name)", "SentenceTransformer outside embed"),
        ("app/ingest/pdf.py", "from app.llm.ollama import Client", "llm import from ingest or scoring"),
        ("app/scoring/matrix.py", "import app.llm", "llm import from ingest or scoring"),
        ("app/scoring/filters.py", "from ..llm import x", "llm import from ingest or scoring"),
        ("app/ingest/pdf.py", 'with open(path, "wb") as handle:', "disk write in ingest"),
        ("app/ingest/pdf.py", "Path(name).write_bytes(pdf_bytes)", "disk write in ingest"),
        ("app/ingest/pdf.py", "Path(name).write_text(text)", "disk write in ingest"),
        ("app/ingest/pdf.py", "import tempfile", "disk write in ingest"),
        ("app/ingest/normalize.py", "handle = NamedTemporaryFile()", "disk write in ingest"),
        (PIPELINE_MODULE, "Path(name).write_bytes(upload.pdf_bytes)", "disk write in pipeline"),
        ("app/web/routes_candidate.py", "Path(name).write_bytes(pdf_bytes)", "disk write in web"),
        ("app/web/routes_candidate.py", "import tempfile", "disk write in web"),
        ("app/web/routes_candidate.py", 'open(path, "wb")', "disk write in web"),
        ("app/pipeline/confirm.py", "import tempfile", "disk write in pipeline"),
        ("app/pipeline/profile_edits.py", 'open(path, "wb")', "disk write in pipeline"),
        (SUMMARY_MODULE, "score_cv_with_llm(client, prompt_input, usage)", "baseline llm in green pipeline"),
        (SUMMARY_MODULE, "client.generate(prompt)", "baseline llm in green pipeline"),
        ("app/pipeline/scoring.py", "build_baseline_prompt(prompt_input)", "baseline llm in green pipeline"),
        ("app/pipeline/confirm.py", "BaselinePromptInput(requirements, cv_text)", "baseline llm in green pipeline"),
        ("app/pipeline/upload.py", "session.delete(application)", "deletion outside retention"),
    ],
)
def test_scanner_detects_violations(
    relative_path: str, source: str, expected_rule: str
) -> None:
    """The scanning helper flags each forbidden pattern."""
    assert expected_rule in find_violations(relative_path, source)


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        (RETENTION_FILE, "session.delete(row)\nDELETE FROM sections"),
        (RETENTION_FILE, "def purge(now):\n    return now"),
        (EMBED_FILE, "model = SentenceTransformer(name)"),
        (SUMMARY_MODULE, "from app.llm.ollama import Client"),
        (SUMMARY_MODULE, "from app.llm.tasks import summarize_candidate"),
        ("bench/baseline.py", "score_cv_with_llm(client, prompt_input, usage)"),
        ("app/ingest/pdf.py", "from app.config import Settings"),
        ("app/ingest/pdf.py", "reader = PdfReader(io.BytesIO(pdf_bytes))"),
        ("bench/export.py", 'Path(name).write_bytes(data)\nopen(path, "wb")'),
        ("tests/pdf_fixtures.py", "import tempfile"),
    ],
)
def test_scanner_allows_permitted_places(relative_path: str, source: str) -> None:
    """The scanning helper does not flag code in its permitted location."""
    assert find_violations(relative_path, source) == []


def test_directory_walk_reports_violation_with_posix_path(tmp_path: Path) -> None:
    """The directory walk finds a violating file in a nested temp `app/` tree."""
    web_directory = tmp_path / "app" / "web"
    web_directory.mkdir(parents=True)
    (web_directory / "offender.py").write_text("session.delete(row)\n", encoding="utf-8")
    (web_directory / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
    violations = scan_directory(tmp_path / "app", tmp_path)
    assert violations == {"app/web/offender.py": ["deletion outside retention"]}


def test_pipeline_is_a_package_and_every_module_is_scanned() -> None:
    """The pipeline-specific rules reach all modules; no stray `app/pipeline.py` remains."""
    package = APP_DIRECTORY / "pipeline"
    assert (package / "__init__.py").is_file()
    assert not (APP_DIRECTORY / "pipeline.py").exists()
    relative_paths = [
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in package.glob("*.py")
    ]
    assert len(relative_paths) > 1
    for relative_path in relative_paths:
        assert is_pipeline(relative_path)
        assert "disk write in pipeline" in find_violations(relative_path, "import tempfile")
