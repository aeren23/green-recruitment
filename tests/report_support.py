"""Builders for the quality and report tests: stored runs, notes and rankings files.

Everything is synthetic; no run is measured and no dataset is needed.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.db import create_benchmark_schema, create_db_engine, create_session_factory
from app.models import BenchmarkRun, PipelineName, StageMeasurement
from bench.quality import QualityContext
from bench.results import RankedEntry
from bench.store import BenchPaths, StoredRun, StoredStage, sqlite_url

POSTING_IT = "Backend Developer"
POSTING_HR = "HR Specialist"
POSTING_ACCOUNTANT = "Accountant"
CATEGORY_IT = "INFORMATION-TECHNOLOGY"
CATEGORY_HR = "HR"
CATEGORY_CHEF = "CHEF"
DEFAULT_APPLICATIONS = 100
DEFAULT_SCENARIO: dict[str, Any] = {
    "jobs_per_cv": 2,
    "models": {"embedding": "e5", "llm": "qwen2.5:3b"},
    "n": 50,
    "resubmit_rate": 0.1,
    "seed": 42,
    "top_k": 5,
}
SECOND_SCENARIO: dict[str, Any] = {**DEFAULT_SCENARIO, "jobs_per_cv": 1, "resubmit_rate": 0.0}
FINAL_NOTES: dict[str, str] = {
    "cpu_method": "CPU(Windows Emi)",
    "git": "abc1234",
    "git_dirty": "false",
    "power_plan": "Dengeli",
    "onedrive_running": "false",
    "llm_status": "available",
    "applications": str(DEFAULT_APPLICATIONS),
    "gpu_stage_sum_over_stream": "0.950000",
    "unstaged_share": "0.010000",
}


def build_notes(**overrides: str) -> str:
    """Final-condition notes with some keys replaced."""
    return "; ".join(f"{key}={value}" for key, value in {**FINAL_NOTES, **overrides}.items())


def make_run(
    run_id: int,
    pipeline: PipelineName,
    notes: str | None = None,
    **fields: Any,
) -> StoredRun:
    """A stored run with round numbers; `fields` replace any column."""
    base = StoredRun(
        run_id=run_id,
        pipeline=pipeline,
        scenario=dict(DEFAULT_SCENARIO),
        energy_wh_gross=10.0,
        energy_wh_net=4.0,
        co2_g=5.0,
        duration_s=200.0,
        idle_power_w=36.0,
        storage_bytes_active=2000,
        storage_bytes_after_retention=1000,
        llm_calls=40,
        llm_prompt_tokens=4000,
        llm_completion_tokens=400,
        parse_cache_hits=0,
        embed_cache_hits=0,
        precision_at_k=0.5,
        overlap_at_k=None,
        measurement_notes=build_notes() if notes is None else notes,
        stages=(StoredStage("extract", 10.0, 1.0, 10),),
    )
    return replace(base, **fields)


def entries(*record_ids: str) -> list[RankedEntry]:
    """A ranking in the given order with descending dummy scores."""
    return [RankedEntry(record_id, 1.0 - index / 100) for index, record_id in enumerate(record_ids)]


def quality_context(
    categories: Mapping[str, str], k: int = 10
) -> QualityContext:
    """A context for the three seeded postings."""
    return QualityContext(
        categories=categories,
        relevant_categories={
            POSTING_IT: frozenset({CATEGORY_IT}),
            POSTING_HR: frozenset({CATEGORY_HR}),
            POSTING_ACCOUNTANT: frozenset({"ACCOUNTANT", "FINANCE"}),
        },
        k=k,
    )


def insert_runs(paths: BenchPaths, runs: Sequence[StoredRun]) -> None:
    """Write stored runs (and their stages) into a test `bench.db`."""
    engine = create_db_engine(sqlite_url(paths.bench_db))
    try:
        create_benchmark_schema(engine)
        with create_session_factory(engine)() as session:
            session.add_all(to_row(run) for run in runs)
            session.commit()
    finally:
        engine.dispose()


def to_row(run: StoredRun) -> BenchmarkRun:
    """The ORM row of a stored run, keeping its id."""
    row = BenchmarkRun(
        id=run.run_id,
        pipeline=run.pipeline,
        scenario_json=dict(run.scenario),
        energy_wh_gross=run.energy_wh_gross,
        energy_wh_net=run.energy_wh_net,
        co2_g=run.co2_g,
        duration_s=run.duration_s,
        idle_power_w=run.idle_power_w,
        storage_bytes_active=run.storage_bytes_active,
        storage_bytes_after_retention=run.storage_bytes_after_retention,
        llm_calls=run.llm_calls,
        llm_prompt_tokens=run.llm_prompt_tokens,
        llm_completion_tokens=run.llm_completion_tokens,
        parse_cache_hits=run.parse_cache_hits,
        embed_cache_hits=run.embed_cache_hits,
        precision_at_k=run.precision_at_k,
        overlap_at_k=run.overlap_at_k,
        measurement_notes=run.measurement_notes,
    )
    row.stage_measurements = [
        StageMeasurement(
            stage=stage.stage,
            energy_wh=stage.energy_wh,
            duration_s=stage.duration_s,
            count=stage.count,
        )
        for stage in run.stages
    ]
    return row


def write_rankings_file(
    paths: BenchPaths, run_id: int, postings: Mapping[str, Sequence[str]]
) -> None:
    """Write `rankings/<run_id>.json` with extra text fields a reader must ignore."""
    payload = {
        "run_id": run_id,
        "pipeline": "x",
        "postings": {
            title: [
                {"record_id": record_id, "score": 0.5, "reason": "SECRET-CV-TEXT"}
                for record_id in ids
            ]
            for title, ids in postings.items()
        },
    }
    directory = paths.rankings_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{run_id}.json").write_text(json.dumps(payload), encoding="utf-8")
