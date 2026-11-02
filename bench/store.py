"""Where a benchmark run is stored: `bench.db` rows and the rankings file.

`BenchmarkRun` and `StageMeasurement` rows go into `data/bench/bench.db`
(SPEC 5). The final rankings (decision T9-1) go to
`data/bench/rankings/<run_id>.json`, as dataset record ids and scores only, so
`precision@k` can be computed after retention has deleted the applications.
The same module reads the runs back and is the only place that updates the
quality columns (`precision_at_k`, `overlap_at_k`) of a stored run.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db import create_benchmark_schema, create_db_engine, create_session_factory
from app.models import BenchmarkRun, PipelineName, StageMeasurement
from bench.results import RankedEntry, Rankings, RunRecord

DEFAULT_BENCH_DIRECTORY = Path("data/bench")
BENCH_DB_NAME = "bench.db"
GREEN_DB_NAME = "bench_green.db"
BASELINE_STORE_NAME = "baseline_store"
IDLE_POWER_NAME = "idle_power.json"
RANKINGS_DIRECTORY_NAME = "rankings"
RANKINGS_SUFFIX = ".json"
JSON_ENCODING = "utf-8"
JSON_INDENT = 2
RANKING_RUN_ID = "run_id"
RANKING_PIPELINE = "pipeline"
RANKING_POSTINGS = "postings"
RANKING_RECORD_ID = "record_id"
RANKING_SCORE = "score"
PRECISION_ATTRIBUTE = "precision_at_k"
OVERLAP_ATTRIBUTE = "overlap_at_k"


@dataclass(frozen=True)
class BenchPaths:
    """File locations under one benchmark directory."""

    directory: Path = DEFAULT_BENCH_DIRECTORY

    @property
    def bench_db(self) -> Path:
        """The shared results database."""
        return self.directory / BENCH_DB_NAME

    @property
    def green_db(self) -> Path:
        """The green runner's scratch database, recreated for every run."""
        return self.directory / GREEN_DB_NAME

    @property
    def baseline_store(self) -> Path:
        """Folder of the baseline's conventional store (SPEC 7.6), one folder per run."""
        return self.directory / BASELINE_STORE_NAME

    @property
    def idle_power(self) -> Path:
        """The idle power record written by `bench idle`."""
        return self.directory / IDLE_POWER_NAME

    @property
    def rankings_directory(self) -> Path:
        """Folder of the per-run rankings files."""
        return self.directory / RANKINGS_DIRECTORY_NAME


def sqlite_url(path: Path) -> str:
    """SQLAlchemy URL of a SQLite file."""
    return f"sqlite:///{path.as_posix()}"


def rankings_payload(run_id: int, record: RunRecord) -> dict[str, object]:
    """The JSON document of a run's rankings: ids and scores only."""
    postings = {
        title: [
            {RANKING_RECORD_ID: entry.record_id, RANKING_SCORE: entry.score}
            for entry in entries
        ]
        for title, entries in record.result.rankings.items()
    }
    return {
        RANKING_RUN_ID: run_id,
        RANKING_PIPELINE: record.pipeline.value,
        RANKING_POSTINGS: postings,
    }


class RankingsFileError(ValueError):
    """A rankings file exists but does not have the documented shape."""


@dataclass(frozen=True)
class StoredStage:
    """One stored `StageMeasurement` row."""

    stage: str
    energy_wh: float
    duration_s: float
    count: int


@dataclass(frozen=True)
class StoredRun:
    """One stored `BenchmarkRun` row with its stages, detached from the session."""

    run_id: int
    pipeline: PipelineName
    scenario: Mapping[str, Any]
    energy_wh_gross: float
    energy_wh_net: float
    co2_g: float
    duration_s: float
    idle_power_w: float
    storage_bytes_active: int
    storage_bytes_after_retention: int
    llm_calls: int
    llm_prompt_tokens: int
    llm_completion_tokens: int
    parse_cache_hits: int
    embed_cache_hits: int
    precision_at_k: float | None
    overlap_at_k: float | None
    measurement_notes: str
    stages: tuple[StoredStage, ...]


def to_stored_run(row: BenchmarkRun) -> StoredRun:
    """Copy a `BenchmarkRun` row (and its stages) into a plain dataclass."""
    stages = tuple(
        StoredStage(item.stage, item.energy_wh, item.duration_s, item.count)
        for item in row.stage_measurements
    )
    return StoredRun(
        run_id=row.id,
        pipeline=row.pipeline,
        scenario=dict(row.scenario_json),
        energy_wh_gross=row.energy_wh_gross,
        energy_wh_net=row.energy_wh_net,
        co2_g=row.co2_g,
        duration_s=row.duration_s,
        idle_power_w=row.idle_power_w,
        storage_bytes_active=row.storage_bytes_active,
        storage_bytes_after_retention=row.storage_bytes_after_retention,
        llm_calls=row.llm_calls,
        llm_prompt_tokens=row.llm_prompt_tokens,
        llm_completion_tokens=row.llm_completion_tokens,
        parse_cache_hits=row.parse_cache_hits,
        embed_cache_hits=row.embed_cache_hits,
        precision_at_k=row.precision_at_k,
        overlap_at_k=row.overlap_at_k,
        measurement_notes=row.measurement_notes,
        stages=stages,
    )


def parse_rankings_payload(payload: Mapping[str, Any]) -> Rankings:
    """Rankings from a rankings document; only record ids and scores are read."""
    try:
        return {
            title: [
                RankedEntry(item[RANKING_RECORD_ID], item[RANKING_SCORE])
                for item in entries
            ]
            for title, entries in payload[RANKING_POSTINGS].items()
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise RankingsFileError(f"rankings document is malformed: {error!r}") from error


def build_run_row(record: RunRecord) -> BenchmarkRun:
    """The `BenchmarkRun` row (with its stage rows) of a run record."""
    result, totals = record.result, record.totals
    row = BenchmarkRun(
        pipeline=record.pipeline,
        scenario_json=record.scenario,
        energy_wh_gross=totals.energy_wh_gross,
        energy_wh_net=totals.energy_wh_net,
        co2_g=totals.co2_g,
        duration_s=totals.duration_s,
        idle_power_w=record.idle_power_w,
        storage_bytes_active=result.storage_bytes_active,
        storage_bytes_after_retention=result.storage_bytes_after_retention,
        llm_calls=result.llm_calls,
        llm_prompt_tokens=result.llm_prompt_tokens,
        llm_completion_tokens=result.llm_completion_tokens,
        parse_cache_hits=result.parse_cache_hits,
        embed_cache_hits=result.embed_cache_hits,
        precision_at_k=None,
        overlap_at_k=None,
        measurement_notes=record.measurement_notes,
    )
    row.stage_measurements = [
        StageMeasurement(
            stage=name,
            energy_wh=stage.energy_wh,
            duration_s=stage.duration_s,
            count=stage.count,
        )
        for name, stage in sorted(record.stages.items())
    ]
    return row


class BenchmarkStore:
    """Writes finished runs to `bench.db` and the rankings directory.

    Args:
        paths: Where the files live.
    """

    def __init__(self, paths: BenchPaths) -> None:
        self._paths = paths

    def save(self, record: RunRecord) -> int:
        """Store the run rows, then the rankings file named after the run id.

        Returns:
            The id of the new `BenchmarkRun`.
        """
        run_id = self._save_rows(record)
        self._write_rankings(run_id, record)
        return run_id

    def _save_rows(self, record: RunRecord) -> int:
        """Insert the run and its stage rows in one transaction."""
        engine = create_db_engine(sqlite_url(self._paths.bench_db))
        try:
            create_benchmark_schema(engine)
            with create_session_factory(engine)() as session:
                row = build_run_row(record)
                session.add(row)
                session.commit()
                return row.id
        finally:
            engine.dispose()

    def _write_rankings(self, run_id: int, record: RunRecord) -> None:
        """Write `rankings/<run_id>.json`."""
        directory = self._paths.rankings_directory
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(rankings_payload(run_id, record), indent=JSON_INDENT)
        target = directory / f"{run_id}{RANKINGS_SUFFIX}"
        target.write_text(payload, encoding=JSON_ENCODING)

    def load_runs(self) -> list[StoredRun]:
        """All stored runs, oldest first. An absent `bench.db` gives an empty list."""
        if not self._paths.bench_db.is_file():
            return []
        engine = create_db_engine(sqlite_url(self._paths.bench_db))
        try:
            with create_session_factory(engine)() as session:
                rows = session.scalars(select(BenchmarkRun).order_by(BenchmarkRun.id))
                return [to_stored_run(row) for row in rows]
        finally:
            engine.dispose()

    def load_rankings(self, run_id: int) -> Rankings | None:
        """The rankings of a run, or `None` when it has no rankings file.

        Raises:
            RankingsFileError: The file is not valid JSON of the documented shape.
        """
        source = self._paths.rankings_directory / f"{run_id}{RANKINGS_SUFFIX}"
        if not source.is_file():
            return None
        try:
            payload = json.loads(source.read_text(encoding=JSON_ENCODING))
        except json.JSONDecodeError as error:
            raise RankingsFileError(f"rankings file of run {run_id} is not JSON") from error
        return parse_rankings_payload(payload)

    def save_precision(self, values: Mapping[int, float | None]) -> None:
        """Set `precision_at_k` of the given runs (a `None` clears the value)."""
        self._update_runs(PRECISION_ATTRIBUTE, values)

    def save_overlap(self, values: Mapping[int, float | None]) -> None:
        """Set `overlap_at_k` of the given runs (a `None` clears the value)."""
        self._update_runs(OVERLAP_ATTRIBUTE, values)

    def _update_runs(self, attribute: str, values: Mapping[int, float | None]) -> None:
        """Write one quality column for several runs in one transaction."""
        if not values:
            return
        engine = create_db_engine(sqlite_url(self._paths.bench_db))
        try:
            with create_session_factory(engine)() as session:
                for row in session.scalars(
                    select(BenchmarkRun).where(BenchmarkRun.id.in_(list(values)))
                ):
                    setattr(row, attribute, values[row.id])
                session.commit()
        finally:
            engine.dispose()
