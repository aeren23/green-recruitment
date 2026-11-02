"""Employer review pages: the ranked list, the application detail and shortlisting.

Every route carries the `require_employer` guard through the router. Viewing a
page never calls the LLM: summaries come from the pipeline (SPEC 6.6) and are
only read here. Contact data is decided in `employer_views.visible_contact`.

KVKK and logging: only application ids are logged.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.models import Application, ApplicationStatus, JobPosting
from app.pipeline.ranking import ranked_statement, score_bounds
from app.pipeline.shortlist import (
    ApplicationNotFoundError,
    ApplicationNotRankedError,
    PostingNotOpenError,
    shortlist_application,
)
from app.web import employer_views, messages, paths
from app.web.auth import REDIRECT_AFTER_POST, require_employer
from app.web.dependencies import WebContext, get_web_context
from app.web.pagination import paginate, parse_page_number
from app.web.routes_employer import EMPLOYER_PREFIX
from app.web.templating import render_page

NOT_FOUND_STATUS = 404
CONFLICT_STATUS = 409
PAGE_QUERY_PARAMETER = "page"
JOB_TEMPLATE = "employer/job.html"
APPLICATION_TEMPLATE = "employer/application.html"
NOTICE_TEMPLATE = "employer/notice.html"
VIEWABLE_STATUSES = (ApplicationStatus.RANKED, ApplicationStatus.SHORTLISTED)

employer_review_router = APIRouter(
    prefix=EMPLOYER_PREFIX, dependencies=[Depends(require_employer)]
)


@employer_review_router.get("/jobs/{job_id}", response_class=HTMLResponse)
def show_job(job_id: int, context: WebContext = Depends(get_web_context)) -> Response:
    """The posting with its ranked candidates, 20 per page, anonymous."""
    posting = context.session.get(JobPosting, job_id)
    if posting is None:
        return _not_found(context, messages.POSTING_NOT_FOUND)
    requested = parse_page_number(context.request.query_params.get(PAGE_QUERY_PARAMETER))
    page = paginate(context.session, ranked_statement(job_id), requested)
    bounds = score_bounds(context.session, job_id)
    values = {
        "posting": posting,
        "filters": employer_views.posting_filter_lines(posting),
        "page": page,
        "rows": employer_views.build_ranked_rows(page.items, bounds),
        "list_path": paths.employer_job_path(job_id),
    }
    return render_page(context.request, JOB_TEMPLATE, values)


@employer_review_router.get("/applications/{application_id}", response_class=HTMLResponse)
def show_application(
    application_id: int, context: WebContext = Depends(get_web_context)
) -> Response:
    """The matrix, the summary and, once shortlisted, the contact data."""
    application = context.session.get(Application, application_id)
    if application is None or application.status not in VIEWABLE_STATUSES:
        return _not_found(context, messages.APPLICATION_NOT_FOUND)
    bounds = score_bounds(context.session, application.job_id)
    detail = employer_views.build_application_detail(context.session, application, bounds)
    return render_page(context.request, APPLICATION_TEMPLATE, {"detail": detail})


@employer_review_router.post("/applications/{application_id}/shortlist")
def shortlist(
    application_id: int, context: WebContext = Depends(get_web_context)
) -> Response:
    """Shortlist a ranked application of an open posting."""
    try:
        shortlist_application(context.session, application_id, context.now)
    except ApplicationNotFoundError:
        return _not_found(context, messages.APPLICATION_NOT_FOUND)
    except ApplicationNotRankedError:
        return _conflict(context, messages.SHORTLIST_REQUIRES_RANKED)
    except PostingNotOpenError:
        return _conflict(context, messages.SHORTLIST_REQUIRES_OPEN_POSTING)
    context.session.commit()
    target = paths.employer_application_path(application_id)
    return RedirectResponse(target, REDIRECT_AFTER_POST)


def _not_found(context: WebContext, message: str) -> Response:
    """The Turkish 404 page of the employer area."""
    values = {
        "heading": messages.NOT_FOUND_HEADING,
        "message": message,
        "link_href": paths.EMPLOYER_HOME_PATH,
        "link_text": messages.BACK_TO_EMPLOYER_POSTINGS,
    }
    return render_page(context.request, NOTICE_TEMPLATE, values, NOT_FOUND_STATUS)


def _conflict(context: WebContext, message: str) -> Response:
    """The Turkish 409 page for a refused shortlisting."""
    values = {
        "heading": messages.SHORTLIST_FAILED_HEADING,
        "message": message,
        "link_href": paths.EMPLOYER_HOME_PATH,
        "link_text": messages.BACK_TO_EMPLOYER_POSTINGS,
    }
    return render_page(context.request, NOTICE_TEMPLATE, values, CONFLICT_STATUS)
