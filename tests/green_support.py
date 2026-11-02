"""Helpers for the green panel tests: a synthetic `bench.db` and the bench path override."""

from pathlib import Path

from fastapi import FastAPI
from report_support import insert_runs, make_run

from app.models import PipelineName
from app.web.routes_green import get_bench_db_path
from bench.store import BenchPaths

BASELINE_NET_ENERGY = 8.0
GREEN_NET_ENERGY = 4.0
NOT_FINAL_POWER_PLAN = "Yuksek performans"


def build_bench_db(directory: Path, final_conditions: bool = True) -> Path:
    """A `bench.db` with one baseline and one green run of the same scenario."""
    paths = BenchPaths(directory)
    notes = None if final_conditions else f"power_plan={NOT_FINAL_POWER_PLAN}"
    insert_runs(
        paths,
        [
            make_run(1, PipelineName.BASELINE, notes, energy_wh_net=BASELINE_NET_ENERGY),
            make_run(
                2,
                PipelineName.GREEN,
                notes,
                energy_wh_net=GREEN_NET_ENERGY,
                llm_calls=5,
                storage_bytes_after_retention=500,
            ),
        ],
    )
    return paths.bench_db


def use_bench_db(application: FastAPI, bench_db: Path) -> None:
    """Point the panel at a bench database; cleared by the test's fixture teardown."""
    application.dependency_overrides[get_bench_db_path] = lambda: bench_db
