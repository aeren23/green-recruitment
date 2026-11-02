"""The baseline's on-disk store: copied PDFs and one JSON Lines results file.

This is the conventional pipeline's storage (SPEC 7.6). It holds CV bytes, CV
text and LLM output on purpose, under `data/bench/baseline_store/`, which is
git-ignored. Nothing in this module logs, prints or returns that content: the
callers get paths, sizes and ids only.

Each run owns one folder. While it runs the folder is called `in_progress`
(the `BenchmarkRun` id is only known after the run is saved) and afterwards it
is renamed to the run id. At the start of a run every earlier folder is
deleted (decision T10-2). This is plain file-system housekeeping of a
benchmark scratch area, not the application's database retention.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

IN_PROGRESS_DIRECTORY = "in_progress"
PDF_NAME_TEMPLATE = "{ordinal:06d}_{record_id}.pdf"
RESULTS_FILE_NAME = "results.jsonl"
TEXT_ENCODING = "utf-8"
LINE_END = "\n"


@dataclass(frozen=True)
class StoredResult:
    """One line of the results file: what the conventional pipeline keeps.

    Attributes:
        ordinal: Position of the application in the scenario (0-based).
        record_id: Dataset record id of the CV.
        posting_title: Posting the CV applied to.
        text: The full extracted CV text.
        score: The LLM score, or `None` when its answer stayed malformed.
        reason: The LLM's reason, when it gave one.
    """

    ordinal: int
    record_id: str
    posting_title: str
    text: str
    score: int | None
    reason: str | None


class BaselineFiles:
    """The folder of one baseline run.

    Args:
        root: The `baseline_store` folder that holds one sub-folder per run.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._work = root / IN_PROGRESS_DIRECTORY

    @property
    def results_path(self) -> Path:
        """The JSON Lines file of the current run."""
        return self._work / RESULTS_FILE_NAME

    def begin(self) -> None:
        """Delete the folders of earlier runs and open a fresh one (T10-2)."""
        self._root.mkdir(parents=True, exist_ok=True)
        for entry in self._root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        self._work.mkdir()

    def write_pdf(self, ordinal: int, record_id: str, pdf_bytes: bytes) -> Path:
        """Copy one CV under a file name that is unique per application."""
        target = self._work / PDF_NAME_TEMPLATE.format(ordinal=ordinal, record_id=record_id)
        target.write_bytes(pdf_bytes)
        return target

    def append_result(self, result: StoredResult) -> None:
        """Append one line; earlier lines are never rewritten."""
        line = json.dumps(asdict(result), ensure_ascii=False)
        with self.results_path.open("a", encoding=TEXT_ENCODING) as handle:
            handle.write(line + LINE_END)

    def size_bytes(self) -> int:
        """Total size of the current run's folder (PDFs and results file)."""
        return sum(file.stat().st_size for file in self._work.rglob("*") if file.is_file())

    def publish(self, run_id: int) -> Path:
        """Rename the finished folder after its `BenchmarkRun` id."""
        target = self._root / str(run_id)
        self._work.rename(target)
        return target

    def discard(self) -> None:
        """Remove the unfinished folder of a run that did not complete."""
        shutil.rmtree(self._work, ignore_errors=True)
