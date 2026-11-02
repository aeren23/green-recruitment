"""`python -m bench run`: build a scenario and run a pipeline over it."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import Settings
from app.green.tracker import CodeCarbonStageTracker
from app.llm.ollama import LlmUnavailableError, OllamaClient, OllamaConfig
from app.models import PipelineName
from app.scoring.embed import create_embedder
from bench.baseline import BaselineRunner
from bench.cuda_context import TorchCudaContextHolder
from bench.dataset import (
    DEFAULT_DATASET_ROOT,
    DatasetStructureError,
    ResumeRecord,
    iter_resume_records,
)
from bench.environment import SystemEnvironmentProbe
from bench.estimate import (
    ScenarioCount,
    count_n300_applications,
    estimate_n300,
    format_estimate,
    per_application_seconds,
)
from bench.gpu_sampler import NvidiaSmiSampler
from bench.green import GreenRunner
from bench.idle import IdleDataError
from bench.results import RunPlan
from bench.run_summary import format_run_summary, format_total_duration
from bench.runner import RunnerServices, RunOutcome
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
RUN_ABORTED_MESSAGE = "The run was aborted and nothing was stored (start Ollama and run again)."
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
    return build_plan_from(list(iter_resume_records(arguments.root)), arguments, settings)


def build_plan_from(
    records: list[ResumeRecord], arguments: argparse.Namespace, settings: Settings
) -> RunPlan:
    """The application list of the command-line scenario over already-read records."""
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


def run_baseline(arguments: argparse.Namespace, settings: Settings) -> int:
    """Run the baseline `--repeats` times; print the duration and the N=300 estimate.

    The N=300 application counts are computed before the first run, so a
    dataset that cannot supply 300 CVs fails fast instead of after the run.
    """
    records = list(iter_resume_records(arguments.root))
    plan = build_plan_from(records, arguments, settings)
    counts = count_n300_applications(records, plan.config)
    services = build_services(settings)
    config = OllamaConfig(settings.ollama_url, settings.llm_model)
    with httpx.Client() as http_client:
        runner = BaselineRunner(services, OllamaClient(http_client, config))
        for _ in range(arguments.repeats):
            print(format_baseline_report(runner.run(plan), counts))
        print(OLLAMA_PS_HEADING)
        print(services.environment.ollama_ps())
    return EXIT_OK


def format_baseline_report(
    outcome: RunOutcome, counts: tuple[ScenarioCount, ...]
) -> str:
    """The run summary, the total duration and the N=300 estimate."""
    per_application_s = per_application_seconds(
        outcome.record.stages, outcome.record.result.counts.submitted
    )
    estimate = format_estimate(estimate_n300(per_application_s, counts))
    return "\n".join([format_run_summary(outcome), format_total_duration(outcome), estimate])


def run_run(arguments: argparse.Namespace) -> int:
    """Handle `run`: dispatch on the pipeline, turning known errors into exit codes."""
    try:
        if arguments.pipeline == PipelineName.BASELINE.value:
            return run_baseline(arguments, Settings())
        return run_green(arguments, Settings())
    except LlmUnavailableError as error:
        print(f"{ERROR_PREFIX}{error} {RUN_ABORTED_MESSAGE}", file=sys.stderr)
        return EXIT_ERROR
    except (IdleDataError, DatasetStructureError, ScenarioConfigError, ValueError) as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
