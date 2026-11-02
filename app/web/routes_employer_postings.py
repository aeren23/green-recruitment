"""Employer actions on a posting's lifecycle: close (R3) and finalize (R4).

The router carries the `require_employer` guard on every route. Both actions
finish with a retention run at the request's clock (SPEC 8.3, T-14).

KVKK and logging: only posting ids are logged.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse, Response

from app.privacy.retention import (
    PostingNotClosedError,
    PostingNotFoundError,
    RetentionContext,
    RetentionPolicy,
    close_posting,
    finalize_posting,
    run,
)
from app.web import messages, paths
from app.web.auth import REDIRECT_AFTER_POST, require_employer
from app.web.dependencies import WebContext, get_web_context
from app.web.routes_employer import EMPLOYER_PREFIX, render_postings_list

logger = logging.getLogger(__name__)

NOT_FOUND_STATUS = 404
CONFLICT_STATUS = 409

postings_action_router = APIRouter(
    prefix=EMPLOYER_PREFIX, dependencies=[Depends(require_employer)]
)


@postings_action_router.post("/jobs/{job_id}/close")
def close_job(job_id: int, context: WebContext = Depends(get_web_context)) -> Response:
    """Close a posting (R3), then apply the time-driven rules."""
    retention_context, policy = _retention_inputs(context)
    try:
        close_posting(retention_context, job_id)
    except PostingNotFoundError:
        raise HTTPException(NOT_FOUND_STATUS) from None
    run(context.session, context.now, policy)
    context.session.commit()
    logger.info("Employer closed posting id=%d", job_id)
    return _redirect_with_notice(messages.NOTICE_CLOSED)


@postings_action_router.post("/jobs/{job_id}/finalize")
def finalize_job(
    job_id: int, context: WebContext = Depends(get_web_context)
) -> Response:
    """Finalize a closed posting, then apply R4; an open posting is rejected."""
    retention_context, policy = _retention_inputs(context)
    try:
        finalize_posting(retention_context, job_id)
    except PostingNotFoundError:
        raise HTTPException(NOT_FOUND_STATUS) from None
    except PostingNotClosedError:
        return render_postings_list(
            context,
            page_number=None,
            error=messages.FINALIZE_REQUIRES_CLOSED,
            status_code=CONFLICT_STATUS,
        )
    run(context.session, context.now, policy)
    context.session.commit()
    logger.info("Employer finalized posting id=%d", job_id)
    return _redirect_with_notice(messages.NOTICE_FINALIZED)


def _retention_inputs(context: WebContext) -> tuple[RetentionContext, RetentionPolicy]:
    """The retention context and policy for this request."""
    policy = RetentionPolicy.from_settings(context.settings)
    return RetentionContext(context.session, context.now, policy.decision_retention_days), policy


def _redirect_with_notice(notice: str) -> Response:
    """Back to the postings list, which shows the notice."""
    target = paths.employer_home_with_notice(notice)
    return RedirectResponse(target, REDIRECT_AFTER_POST)
