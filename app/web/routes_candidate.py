"""Candidate routes of SPEC 9.1: postings, upload, preview and confirm, status.

There is no candidate login (T13-5): the unguessable token in the URL is the
access right, so draft and status pages are sent with `Cache-Control:
no-store`. The handlers only translate HTTP into `app.pipeline` calls and the
result into Turkish pages; they hold no business logic and never touch the
PDF beyond handing its bytes to `submit_upload`.

KVKK and logging: tokens, form contents and file contents are never logged.
"""

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.ingest.pdf import TextTooShortError, UnreadablePdfError
from app.models import JobPosting, JobStatus
from app.pipeline import (
    ConfirmInput,
    DraftNotFoundError,
    DraftPreview,
    DraftState,
    PostingClosedError,
    UploadInput,
    UploadOutcomeKind,
    confirm_draft,
    look_up_status,
    open_draft,
    submit_upload,
)
from app.privacy.retention import ApplicationNotDraftError, PostingNotFoundError
from app.web import messages, paths
from app.web.auth import REDIRECT_AFTER_POST
from app.web.background import schedule_summary_refresh
from app.web.dependencies import WebContext, get_web_context
from app.web.draft_form import (
    BLANK_SECTION_ROWS,
    SECTION_TEXT_FIELD,
    SECTION_TYPE_FIELD,
    DraftFormValues,
    build_edits,
    is_well_formed,
    pair_section_rows,
    values_from_preview,
)
from app.web.pagination import paginate, parse_page_number
from app.web.pipeline_context import pipeline_context_of
from app.web.templating import render_page

logger = logging.getLogger(__name__)

JOBS_TEMPLATE = "candidate/jobs.html"
JOB_TEMPLATE = "candidate/job.html"
PREVIEW_TEMPLATE = "candidate/preview.html"
STATUS_TEMPLATE = "candidate/status.html"
NOTICE_TEMPLATE = "candidate/notice.html"
UPLOAD_FIELD = "cv"
SIMPLE_FIELDS = {
    "years": "years_experience",
    "languages": "languages",
    "location": "location",
    "contact_name": "contact_name",
    "contact_email": "contact_email",
    "contact_phone": "contact_phone",
}
NOT_FOUND_STATUS = 404
CONFLICT_STATUS = 409
GONE_STATUS = 410
UNPROCESSABLE_STATUS = 422
CACHE_CONTROL_HEADER = "Cache-Control"
NO_STORE = "no-store"

UPLOAD_ERRORS: dict[type[Exception], tuple[int, str]] = {
    UnreadablePdfError: (UNPROCESSABLE_STATUS, messages.UPLOAD_UNREADABLE),
    TextTooShortError: (UNPROCESSABLE_STATUS, messages.UPLOAD_TOO_SHORT),
    PostingClosedError: (CONFLICT_STATUS, messages.POSTING_CLOSED),
}

candidate_router = APIRouter()


@dataclass(frozen=True)
class DraftSubmission:
    """The posted preview form together with the draft token of the URL."""

    token: str
    values: DraftFormValues
    well_formed: bool = True


async def read_upload(job_id: int, request: Request) -> UploadInput:
    """Read the posted CV into memory (a dependency of the apply route).

    The file is read from the in-memory multipart part; the size limit
    middleware guarantees the body is below Starlette's disk threshold.
    """
    form = await request.form()
    part = form.get(UPLOAD_FIELD)
    pdf_bytes = await part.read() if isinstance(part, StarletteUploadFile) else b""
    await form.close()
    return UploadInput(job_id=job_id, pdf_bytes=pdf_bytes)


async def read_draft_submission(token: str, request: Request) -> DraftSubmission:
    """Read the posted preview form into raw values (a dependency of the confirm route)."""
    form = await request.form()
    simple = {name: str(form.get(field, "")) for name, field in SIMPLE_FIELDS.items()}
    rows = pair_section_rows(
        [str(item) for item in form.getlist(SECTION_TYPE_FIELD)],
        [str(item) for item in form.getlist(SECTION_TEXT_FIELD)],
    )
    values = DraftFormValues(rows=rows, **simple)
    return DraftSubmission(token, values, is_well_formed(form.keys()))


def render_private_page(
    request: Request, template: str, values: dict[str, Any], status_code: int = 200
) -> HTMLResponse:
    """Render a page that holds a token or personal data; it must not be cached."""
    response = render_page(request, template, values, status_code)
    response.headers[CACHE_CONTROL_HEADER] = NO_STORE
    return response


def render_notice(
    request: Request, heading: str, message: str, link: tuple[str, str], status_code: int
) -> HTMLResponse:
    """A short message page with one link (expired draft, closed posting, 404)."""
    values = {
        "heading": heading,
        "message": message,
        "link_href": link[0],
        "link_text": link[1],
    }
    return render_page(request, NOTICE_TEMPLATE, values, status_code)


def not_found_page(request: Request) -> HTMLResponse:
    """The Turkish 404 page for an unknown posting."""
    link = (paths.HOME_PATH, messages.BACK_TO_POSTINGS)
    heading, message = messages.NOT_FOUND_HEADING, messages.JOB_NOT_FOUND
    return render_notice(request, heading, message, link, NOT_FOUND_STATUS)


def redirect(path: str) -> RedirectResponse:
    """A 303 redirect (a form post is answered with a GET)."""
    return RedirectResponse(path, REDIRECT_AFTER_POST)


def retention_values(context: WebContext) -> dict[str, int]:
    """The retention periods the disclosure and the preview summary quote."""
    settings = context.settings
    return {
        "draft_ttl_hours": settings.draft_ttl_hours,
        "decision_retention_days": settings.decision_retention_days,
        "shortlist_retention_days": settings.shortlist_retention_days,
    }


@candidate_router.get("/", response_class=HTMLResponse)
def list_open_postings(
    context: WebContext = Depends(get_web_context), page: str | None = Query(None)
) -> Response:
    """The open postings, newest first, one page at a time."""
    statement = (
        select(JobPosting)
        .where(JobPosting.status == JobStatus.OPEN)
        .order_by(JobPosting.created_at.desc(), JobPosting.id.desc())
    )
    postings = paginate(context.session, statement, parse_page_number(page))
    values = {"page": postings, "list_path": paths.HOME_PATH}
    return render_page(context.request, JOBS_TEMPLATE, values)


def render_job_page(
    context: WebContext, posting: JobPosting, error: str | None = None, status_code: int = 200
) -> HTMLResponse:
    """The posting detail page, with an upload error when there is one."""
    values = {
        "posting": posting,
        "error": error,
        "is_open": posting.status is JobStatus.OPEN,
        "retention": retention_values(context),
        "language_labels": messages.LANGUAGE_LABELS,
        "upload_limit": messages.format_size(context.settings.max_upload_bytes),
    }
    return render_page(context.request, JOB_TEMPLATE, values, status_code)


@candidate_router.get("/jobs/{job_id}", response_class=HTMLResponse)
def show_job(job_id: int, context: WebContext = Depends(get_web_context)) -> Response:
    """Posting detail: hard filters, disclosure text and the upload form."""
    posting = context.session.get(JobPosting, job_id)
    if posting is None:
        return not_found_page(context.request)
    return render_job_page(context, posting)


@candidate_router.post("/jobs/{job_id}/apply", response_class=HTMLResponse)
def apply_to_job(
    context: WebContext = Depends(get_web_context),
    upload: UploadInput = Depends(read_upload),
) -> Response:
    """Turn the uploaded PDF into a draft and go to its preview, or show an error."""
    job_id = upload.job_id
    try:
        outcome = submit_upload(pipeline_context_of(context), upload)
    except PostingNotFoundError:
        return not_found_page(context.request)
    except tuple(UPLOAD_ERRORS) as error:
        status_code, message = UPLOAD_ERRORS[type(error)]
        posting = context.session.get(JobPosting, job_id)
        return render_job_page(context, posting, message, status_code)
    if outcome.kind is UploadOutcomeKind.DUPLICATE_APPLICATION:
        posting = context.session.get(JobPosting, job_id)
        message = messages.DUPLICATE_APPLICATION
        return render_job_page(context, posting, message, CONFLICT_STATUS)
    context.session.commit()
    target = paths.draft_path(outcome.draft_token, outcome.cache_hit, outcome.no_headings)
    return redirect(target)


def try_open_draft(context: WebContext, token: str) -> DraftPreview | None:
    """The draft behind a token, or `None` when it is gone or already confirmed."""
    try:
        return open_draft(pipeline_context_of(context), token)
    except (DraftNotFoundError, ApplicationNotDraftError):
        return None


def expired_page(context: WebContext, preview: DraftPreview) -> HTMLResponse:
    """The expired-draft page (T13-2); the draft was removed through retention."""
    context.session.commit()
    link = (paths.job_path(preview.job_id), messages.UPLOAD_AGAIN)
    heading, message = messages.DRAFT_EXPIRED_HEADING, messages.DRAFT_EXPIRED
    return render_notice(context.request, heading, message, link, GONE_STATUS)


def render_preview(
    context: WebContext,
    preview: DraftPreview,
    values: DraftFormValues,
    errors: tuple[str, ...] = (),
    status_code: int = 200,
) -> HTMLResponse:
    """The preview and edit form."""
    query = context.request.query_params
    page_values = {
        "preview": preview,
        "form": values.with_blank_rows(),
        "errors": errors,
        "cache_hit": query.get(paths.CACHE_HIT_QUERY_PARAMETER) == paths.FLAG_ON,
        "no_headings": query.get(paths.NO_HEADINGS_QUERY_PARAMETER) == paths.FLAG_ON,
        "retention": retention_values(context),
        "token": context.request.path_params["token"],
        "blank_rows": BLANK_SECTION_ROWS,
    }
    return render_private_page(context.request, PREVIEW_TEMPLATE, page_values, status_code)


@candidate_router.get("/drafts/{token}", response_class=HTMLResponse)
def show_draft(token: str, context: WebContext = Depends(get_web_context)) -> Response:
    """The preview of a draft, pre-filled from the parsed (or cached) profile."""
    preview = try_open_draft(context, token)
    if preview is None:
        return redirect(paths.application_path(token))
    if preview.state is DraftState.EXPIRED:
        return expired_page(context, preview)
    return render_preview(context, preview, values_from_preview(preview))


@candidate_router.post("/drafts/{token}", response_class=HTMLResponse)
def confirm_submission(
    context: WebContext = Depends(get_web_context),
    submission: DraftSubmission = Depends(read_draft_submission),
) -> Response:
    """Apply the candidate's corrections and confirm; summaries come afterwards."""
    token = submission.token
    preview = try_open_draft(context, token)
    if preview is None:
        return redirect(paths.application_path(token))
    if preview.state is DraftState.EXPIRED:
        return expired_page(context, preview)
    if not submission.well_formed:
        return render_preview(
            context,
            preview,
            values_from_preview(preview),
            (messages.FORM_INCOMPLETE,),
            UNPROCESSABLE_STATUS,
        )
    result = build_edits(submission.values)
    if result.edits is None:
        return render_preview(
            context, preview, submission.values, result.errors, UNPROCESSABLE_STATUS
        )
    return confirm_and_redirect(context, ConfirmInput(token, result.edits))


def confirm_and_redirect(context: WebContext, confirm_input: ConfirmInput) -> Response:
    """Run `confirm_draft` without an LLM, commit, and schedule the summaries."""
    try:
        outcome = confirm_draft(pipeline_context_of(context), confirm_input)
    except PostingClosedError:
        link = (paths.HOME_PATH, messages.BACK_TO_POSTINGS)
        heading, message = messages.POSTING_CLOSED_HEADING, messages.POSTING_CLOSED
        return render_notice(context.request, heading, message, link, CONFLICT_STATUS)
    context.session.commit()
    response = redirect(paths.application_path(confirm_input.draft_token))
    response.background = schedule_summary_refresh(context, outcome.affected_job_ids)
    return response


@candidate_router.get("/applications/{token}", response_class=HTMLResponse)
def show_status(token: str, context: WebContext = Depends(get_web_context)) -> Response:
    """The status of an application: the application first, then its decision record."""
    view = look_up_status(context.session, token, context.now)
    values = {
        "view": view,
        "token": token,
        "reasons": messages.ELIMINATION_REASONS,
        "retention": retention_values(context),
    }
    return render_private_page(context.request, STATUS_TEMPLATE, values)
