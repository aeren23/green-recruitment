"""`python -m bench run`: build a scenario and run a pipeline over it."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import Settings
from app.green.tracker import CodeCarbonStageTracker
from app.llm.ollama import OllamaClient, OllamaConfig
from app.models import PipelineName
from app.scoring.embed import create_embedder
from bench.cuda_context import TorchCudaContextHolder
from bench.dataset import DEFAULT_DATASET_ROOT, DatasetStructureError, iter_resume_records
from bench.environment import SystemEnvironmentProbe
from bench.gpu_sampler import NvidiaSmiSampler
from bench.green import GreenRunner
from bench.idle import IdleDataError
from bench.results import RunPlan
from bench.run_summary import format_run_summary
from bench.runner import RunnerServices
from bench.scenario import (
    DEFAULT_JOBS_PER_CV,
    DEFAULT_RESUBMIT_RATE,
    ScenarioConfig,
    ScenarioConfigError,
    ScenarioModels,
    build_scenario,
)
from bench.store import DEFAULT_BENCH_DIRECTORY, BenchPaths

RUN_COMMAND = "run"
RUN_HELP = "Run a pipeline over a seeded scenario and store the measurements."
HANDLER_DESTINATION = "handler"
DEFAULT_N = 50
DEFAULT_SEED = 42
DEFAULT_REPEATS = 1
MIN_COUNT = 1
EXIT_OK = 0
EXIT_ERROR = 1
ERROR_PREFIX = "Benchmark error: "
BASELINE_NOT_IMPLEMENTED = "The baseline pipeline is not implemented until T-10."
OLLAMA_PS_HEADING = "ollama ps (taken right after the run):"


def positive_int(value: str) -> int:
    """argparse type for options that must be at least 1."""
    number = int(value)
    if number < MIN_COUNT:
        raise argparse.ArgumentTypeError(f"{value} must be at least {MIN_COUNT}")
    return number


def register_run(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the `run` subcommand to the parser."""
    parser = subparsers.add_parser(RUN_COMMAND, help=RUN_HELP)
    parser.add_argument(
        "--pipeline", choices=[name.value for name in PipelineName], required=True
    )
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--jobs-per-cv", type=positive_int, default=DEFAULT_JOBS_PER_CV)
    parser.add_argument("--resubmit-rate", type=float, default=DEFAULT_RESUBMIT_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.set_defaults(**{HANDLER_DESTINATION: run_run})


def build_config(arguments: argparse.Namespace, settings: Settings) -> ScenarioConfig:
    """The scenario config from the command line and the settings."""
    return ScenarioConfig(
        n=arguments.n,
        seed=arguments.seed,
        top_k=settings.top_k,
        models=ScenarioModels(settings.embed_model, settings.llm_model),
        jobs_per_cv=arguments.jobs_per_cv,
        resubmit_rate=arguments.resubmit_rate,
    )


def build_plan(arguments: argparse.Namespace, settings: Settings) -> RunPlan:
    """Sample the dataset and build the application list."""
    records = list(iter_resume_records(arguments.root))
    config = build_config(arguments, settings)
    applications = build_scenario(records, config)
    needed = {application.record_id for application in applications}
    return RunPlan(config, applications, {r.id: r for r in records if r.id in needed})


def build_services(settings: Settings) -> RunnerServices:
    """The real tracker, CUDA holder, GPU sampler and environment probe."""
    return RunnerServices(
        settings=settings,
        tracker=CodeCarbonStageTracker(settings.country_iso, settings.embed_device),
        cuda_holder=TorchCudaContextHolder(),
        gpu_sampler=NvidiaSmiSampler(),
        environment=SystemEnvironmentProbe(),
        paths=BenchPaths(DEFAULT_BENCH_DIRECTORY),
        clock=lambda: datetime.now(UTC),
    )


def run_green(arguments: argparse.Namespace, settings: Settings) -> int:
    """Run the green pipeline `--repeats` times over one plan."""
    plan = build_plan(arguments, settings)
    services = build_services(settings)
    config = OllamaConfig(settings.ollama_url, settings.llm_model)
    with httpx.Client() as http_client:
        runner = GreenRunner(
            services, create_embedder(settings), OllamaClient(http_client, config)
        )
        for _ in range(arguments.repeats):
            print(format_run_summary(runner.run(plan)))
        print(OLLAMA_PS_HEADING)
        print(services.environment.ollama_ps())
    return EXIT_OK


def run_run(arguments: argparse.Namespace) -> int:
    """Handle `run`: dispatch on the pipeline, turning known errors into exit codes."""
    if arguments.pipeline == PipelineName.BASELINE.value:
        print(BASELINE_NOT_IMPLEMENTED, file=sys.stderr)
        return EXIT_ERROR
    try:
        return run_green(arguments, Settings())
    except (IdleDataError, DatasetStructureError, ScenarioConfigError, ValueError) as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
