"""Employer routes of SPEC 9.1: login, postings list and the new-posting form.

Two routers: `login_router` is open, `employer_router` carries the
`require_employer` guard on every route, so a new employer page is protected by
default. The posting detail page (`/employer/jobs/{id}`) is T-15; the list
already links to it.

KVKK and logging: form contents and tokens are never logged.
"""

import logging

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select

from app.config import Settings
from app.models import REQUIREMENT_WEIGHT_MAX, REQUIREMENT_WEIGHT_MIN, JobPosting
from app.postings import create_posting
from app.web import messages, paths
from app.web.auth import (
    REDIRECT_AFTER_POST,
    clear_session_cookie,
    is_employer,
    require_employer,
    set_session_cookie,
    tokens_match,
)
from app.web.dependencies import WebContext, get_settings, get_web_context
from app.web.pagination import paginate, parse_page_number
from app.web.posting_form import (
    PostingFormValues,
    build_posting_input,
    pair_rows,
)
from app.web.templating import render_page

logger = logging.getLogger(__name__)

EMPLOYER_PREFIX = paths.EMPLOYER_HOME_PATH
LOGIN_TEMPLATE = "employer/login.html"
POSTINGS_TEMPLATE = "employer/postings.html"
FORM_TEMPLATE = "employer/job_form.html"
UNPROCESSABLE_STATUS = 422
UNAUTHORIZED_STATUS = 401
FIELD_TEXT = "requirement_text"
FIELD_WEIGHT = "requirement_weight"
FIELD_REMOTE_OK = "remote_ok"
FIELD_NAMES = ("title", "description", "languages", "location", "min_years")

login_router = APIRouter(prefix=EMPLOYER_PREFIX)
employer_router = APIRouter(
    prefix=EMPLOYER_PREFIX, dependencies=[Depends(require_employer)]
)


@login_router.get("/login", response_class=HTMLResponse)
def show_login(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    """Show the token form; an already logged-in employer goes to the list."""
    if is_employer(request, settings):
        return RedirectResponse(paths.EMPLOYER_HOME_PATH, REDIRECT_AFTER_POST)
    return render_page(request, LOGIN_TEMPLATE, {"error": None})


@login_router.post("/login", response_class=HTMLResponse)
def submit_login(
    request: Request,
    token: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Check the token; on success set the cookie, otherwise show an error."""
    if not tokens_match(token, settings.employer_token):
        logger.warning("Employer login failed")
        return render_page(
            request, LOGIN_TEMPLATE, {"error": messages.LOGIN_FAILED}, UNAUTHORIZED_STATUS
        )
    response = RedirectResponse(paths.EMPLOYER_HOME_PATH, REDIRECT_AFTER_POST)
    set_session_cookie(response, settings)
    return response


@employer_router.post("/logout")
def logout() -> Response:
    """Expire the employer cookie and go back to the login form."""
    response = RedirectResponse(paths.EMPLOYER_LOGIN_PATH, REDIRECT_AFTER_POST)
    clear_session_cookie(response)
    return response


@employer_router.get("", response_class=HTMLResponse)
def list_postings(
    context: WebContext = Depends(get_web_context),
    page: str | None = Query(None),
    notice: str | None = Query(None),
) -> Response:
    """The postings, newest first, one page at a time."""
    return render_postings_list(
        context, page, notice=messages.POSTING_NOTICES.get(notice or "")
    )


def render_postings_list(
    context: WebContext,
    page_number: str | None,
    notice: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> Response:
    """Render the postings list, optionally with a notice or an error message."""
    statement = select(JobPosting).order_by(
        JobPosting.created_at.desc(), JobPosting.id.desc()
    )
    postings = paginate(context.session, statement, parse_page_number(page_number))
    values = {
        "page": postings,
        "list_path": paths.EMPLOYER_HOME_PATH,
        "notice": notice,
        "error": error,
    }
    return render_page(context.request, POSTINGS_TEMPLATE, values, status_code)


@employer_router.get("/jobs/new", response_class=HTMLResponse)
def show_job_form(context: WebContext = Depends(get_web_context)) -> Response:
    """An empty new-posting form."""
    return _render_form(context.request, PostingFormValues(), [])


async def read_form_values(request: Request) -> PostingFormValues:
    """Read the posted form into raw values (a dependency of the submit route)."""
    form = await request.form()
    text_fields = {name: str(form.get(name, "")) for name in FIELD_NAMES}
    rows = pair_rows(
        [str(item) for item in form.getlist(FIELD_TEXT)],
        [str(item) for item in form.getlist(FIELD_WEIGHT)],
    )
    return PostingFormValues(
        remote_ok=form.get(FIELD_REMOTE_OK) is not None, rows=rows, **text_fields
    )


@employer_router.post("/jobs/new", response_class=HTMLResponse)
def submit_job_form(
    context: WebContext = Depends(get_web_context),
    values: PostingFormValues = Depends(read_form_values),
) -> Response:
    """Validate the form and create the posting, or show the form again."""
    result = build_posting_input(values, context.now)
    if result.posting_input is None:
        return _render_form(
            context.request, values, list(result.errors), UNPROCESSABLE_STATUS
        )
    posting = create_posting(context.session, context.embedder, result.posting_input)
    context.session.commit()
    logger.info("Employer created posting id=%d", posting.id)
    return RedirectResponse(paths.EMPLOYER_HOME_PATH, REDIRECT_AFTER_POST)


def _render_form(
    request: Request,
    values: PostingFormValues,
    errors: list[str],
    status_code: int = 200,
) -> Response:
    """Render the form with the entered values and any error messages."""
    weights = [
        (weight, messages.WEIGHT_LABELS[weight])
        for weight in range(REQUIREMENT_WEIGHT_MIN, REQUIREMENT_WEIGHT_MAX + 1)
    ]
    page_values = {
        "form": values.with_padded_rows(),
        "errors": errors,
        "weight_options": weights,
    }
    return render_page(request, FORM_TEMPLATE, page_values, status_code)
