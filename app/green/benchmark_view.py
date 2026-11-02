"""The latest benchmark comparison for the green panel (SPEC 5, 9.1).

`bench.db` is opened read-only: the panel never changes a measurement. The
comparison itself is computed by the same report code as `bench compare`, so the
panel and `report.md` cannot disagree.
"""

import logging
from pathlib import Path

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from bench.report import build_report
from bench.report_model import GroupReport
from bench.store import read_only_engine, read_runs

logger = logging.getLogger(__name__)


def latest_group(groups: tuple[GroupReport, ...]) -> GroupReport | None:
    """The group that holds the most recently stored run, or `None` without groups."""
    if not groups:
        return None
    return max(groups, key=lambda group: max(group.green_run_ids + group.baseline_run_ids))


def load_latest_comparison(bench_db: Path, quality_k: int) -> GroupReport | None:
    """Read `bench.db` read-only and return the latest comparable group.

    Args:
        bench_db: Path of `data/bench/bench.db`.
        quality_k: The quality cut-off the report is built for.

    Returns:
        The latest group that has both pipelines, or `None` when the file is
        missing, empty, unreadable or holds no comparable group.
    """
    if not bench_db.is_file():
        return None
    engine = read_only_engine(bench_db)
    try:
        with Session(engine) as session:
            runs = read_runs(session)
    except OperationalError as error:
        logger.warning("bench.db could not be read: %s", type(error).__name__)
        return None
    finally:
        engine.dispose()
    return latest_group(build_report(runs, quality_k).groups)
