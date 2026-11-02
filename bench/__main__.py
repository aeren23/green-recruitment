"""Command line entry point: `python -m bench <command>`.

Each command registers itself through a function listed in `COMMAND_REGISTRARS`,
so a new command is added by appending to that tuple without touching the
existing commands.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from bench.dataset import (
    DEFAULT_DATASET_ROOT,
    DatasetStats,
    DatasetStructureError,
    compute_dataset_stats,
    iter_resume_records,
)
from bench.idle import register_idle

PROGRAM_NAME = "python -m bench"
SUBCOMMAND_DESTINATION = "command"
HANDLER_DESTINATION = "handler"
DATASET_STATS_COMMAND = "dataset-stats"
DATASET_STATS_HELP = "Print the category distribution and PDF availability of the dataset."
ROOT_OPTION = "--root"
ROOT_HELP = "Dataset root directory (default: %(default)s)."
ERROR_PREFIX = "Dataset structure error: "

EXIT_OK = 0
EXIT_ERROR = 1

CATEGORY_COLUMN_WIDTH = 24

SubparserRegistrar = Callable[["argparse._SubParsersAction[argparse.ArgumentParser]"], None]


def format_dataset_stats(stats: DatasetStats) -> str:
    """Render dataset statistics as a plain-text report."""
    lines = ["Category distribution:"]
    for category, count in stats.category_distribution.items():
        lines.append(f"  {category:<{CATEGORY_COLUMN_WIDTH}}{count}")
    lines.append(f"Total records: {stats.total_records}")
    lines.append(f"PDF count: {stats.pdf_count}")
    lines.append(f"Missing PDF count: {stats.missing_pdf_count}")
    return "\n".join(lines)


def run_dataset_stats(arguments: argparse.Namespace) -> int:
    """Handle `dataset-stats`: print statistics or a clean structure error."""
    try:
        stats = compute_dataset_stats(iter_resume_records(arguments.root))
    except DatasetStructureError as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
    print(format_dataset_stats(stats))
    return EXIT_OK


def register_dataset_stats(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the `dataset-stats` subcommand to the parser."""
    parser = subparsers.add_parser(DATASET_STATS_COMMAND, help=DATASET_STATS_HELP)
    parser.add_argument(
        ROOT_OPTION, type=Path, default=DEFAULT_DATASET_ROOT, help=ROOT_HELP
    )
    parser.set_defaults(**{HANDLER_DESTINATION: run_dataset_stats})


COMMAND_REGISTRARS: tuple[SubparserRegistrar, ...] = (
    register_dataset_stats,
    register_idle,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every registered subcommand."""
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    subparsers = parser.add_subparsers(dest=SUBCOMMAND_DESTINATION, required=True)
    for register in COMMAND_REGISTRARS:
        register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    arguments = build_parser().parse_args(argv)
    return getattr(arguments, HANDLER_DESTINATION)(arguments)


if __name__ == "__main__":
    sys.exit(main())
