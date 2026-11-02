"""The green panel (SPEC 9.1): `GET /green` is public, `POST /green/retention` is not.

The retention button applies R1, R4 and R5 at the request's clock shifted by
the chosen number of days (the demo's "fast-forward"), then vacuums the demo
database. The counts travel back as integer query values, never as text.

KVKK and logging: only counts are logged.
"""

import logging
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.db import vacuum_database
from app.green.benchmark_view import load_latest_comparison
from app.green.counters import derive_llm_calls_avoided, read_counters
from app.privacy.retention import RetentionPolicy, run
from app.web import green_views, messages, paths
from app.web.auth import REDIRECT_AFTER_POST, require_employer
from app.web.dependencies import WebContext, get_web_context
from app.web.templating import render_page
from bench.store import BENCH_DB_NAME, DEFAULT_BENCH_DIRECTORY

logger = logging.getLogger(__name__)

GREEN_TEMPLATE = "green.html"
UNPROCESSABLE_STATUS = 422
MIN_SHIFT_DAYS = 0
MAX_SHIFT_DAYS = 365
MAX_QUERY_COUNT = 10**9
RESULT_PARAMETERS = (
    paths.RETENTION_SHIFT_PARAMETER,
    paths.RETENTION_DRAFTS_PARAMETER,
    paths.RETENTION_SHORTLISTS_PARAMETER,
    paths.RETENTION_DECISIONS_PARAMETER,
)

green_router = APIRouter()
retention_router = APIRouter(dependencies=[Depends(require_employer)])


def get_bench_db_path() -> Path:
    """Where `bench.db` lives; tests override this dependency."""
    return DEFAULT_BENCH_DIRECTORY / BENCH_DB_NAME


def parse_shift_days(raw_value: str) -> int | None:
    """The fast-forward days, or `None` when not a whole number in 0..365."""
    try:
        days = int(raw_value.strip())
    except ValueError:
        return None
    return days if MIN_SHIFT_DAYS <= days <= MAX_SHIFT_DAYS else None


def query_count(request: Request, name: str) -> int | None:
    """A non-negative integer query value, or `None` when absent or malformed."""
    raw_value = request.query_params.get(name, "")
    if not raw_value.isascii() or not raw_value.isdigit():
        return None
    return min(int(raw_value), MAX_QUERY_COUNT)


def retention_notice(request: Request) -> str | None:
    """The result notice of a finished retention run, from its integer flags."""
    counts = [query_count(request, name) for name in RESULT_PARAMETERS]
    if any(count is None for count in counts):
        return None
    days, drafts, shortlists, decisions = counts
    return messages.RETENTION_RESULT.format(
        days=days, drafts=drafts, shortlists=shortlists, decisions=decisions
    )


def render_panel(
    context: WebContext,
    bench_db: Path,
    notice: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> Response:
    """Render the panel with live counters, the comparison and page weight."""
    group = load_latest_comparison(bench_db, context.settings.quality_k)
    values = {
        "counters": green_views.build_counter_lines(
            read_counters(context.session), derive_llm_calls_avoided(context.session)
        ),
        "comparison": None if group is None else green_views.build_comparison_view(group),
        "page_weights": green_views.build_page_weight_lines(
            context.request.app.state.page_weights.snapshot()
        ),
        "notice": notice,
        "error": error,
        "retention_path": paths.GREEN_RETENTION_PATH,
        "days_field": paths.RETENTION_DAYS_FIELD,
        "days_max": MAX_SHIFT_DAYS,
    }
    return render_page(context.request, GREEN_TEMPLATE, values, status_code)


@green_router.get(paths.GREEN_PATH, response_class=HTMLResponse)
def show_green_panel(
    context: WebContext = Depends(get_web_context),
    bench_db: Path = Depends(get_bench_db_path),
) -> Response:
    """Live counters, the latest benchmark comparison and page weight."""
    return render_panel(context, bench_db, notice=retention_notice(context.request))


@retention_router.post(paths.GREEN_RETENTION_PATH)
def run_retention(
    days: str = Form("0"),
    context: WebContext = Depends(get_web_context),
    bench_db: Path = Depends(get_bench_db_path),
) -> Response:
    """Run retention at `now + days`, vacuum the demo database, redirect back."""
    shifted_days = parse_shift_days(days)
    if shifted_days is None:
        error = messages.RETENTION_DAYS_INVALID.format(limit=MAX_SHIFT_DAYS)
        return render_panel(context, bench_db, error=error, status_code=UNPROCESSABLE_STATUS)
    policy = RetentionPolicy.from_settings(context.settings)
    report = run(context.session, context.now + timedelta(days=shifted_days), policy)
    context.session.commit()
    vacuum_database(context.request.app.state.engine)
    logger.info(
        "Panel retention shifted=%d drafts=%d shortlisted=%d decisions=%d",
        shifted_days,
        report.expired_drafts,
        report.released_shortlists,
        report.expired_decisions,
    )
    target = paths.green_with_retention_result(
        shifted_days,
        report.expired_drafts,
        report.released_shortlists,
        report.expired_decisions,
    )
    return RedirectResponse(target, status_code=REDIRECT_AFTER_POST)
