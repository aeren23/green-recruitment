"""URL paths of the web layer, in one place so routes, templates and tests agree."""

EMPLOYER_HOME_PATH = "/employer"
EMPLOYER_LOGIN_PATH = "/employer/login"
EMPLOYER_LOGOUT_PATH = "/employer/logout"
EMPLOYER_NEW_JOB_PATH = "/employer/jobs/new"
EMPLOYER_JOB_PATH_TEMPLATE = "/employer/jobs/{job_id}"
STATIC_URL_PATH = "/static"
STYLESHEET_URL = "/static/app.css"


def employer_job_path(job_id: int) -> str:
    """Path of the employer's posting detail page (built in T-15)."""
    return EMPLOYER_JOB_PATH_TEMPLATE.format(job_id=job_id)

HOME_PATH = "/"
JOB_PATH_TEMPLATE = "/jobs/{job_id}"
APPLY_PATH_TEMPLATE = "/jobs/{job_id}/apply"
DRAFT_PATH_TEMPLATE = "/drafts/{token}"
APPLICATION_PATH_TEMPLATE = "/applications/{token}"
CACHE_HIT_QUERY_PARAMETER = "cached"
NO_HEADINGS_QUERY_PARAMETER = "noheadings"
FLAG_ON = "1"


def job_path(job_id: int) -> str:
    """Path of a posting's candidate-facing detail page."""
    return JOB_PATH_TEMPLATE.format(job_id=job_id)


def apply_path(job_id: int) -> str:
    """Path the CV upload form posts to."""
    return APPLY_PATH_TEMPLATE.format(job_id=job_id)


def draft_path(token: str, cache_hit: bool = False, no_headings: bool = False) -> str:
    """Path of a draft's preview page, with the upload's notices as query flags.

    The flags carry no personal data; they only tell the page which notice to show.
    """
    path = DRAFT_PATH_TEMPLATE.format(token=token)
    flags = []
    if cache_hit:
        flags.append(f"{CACHE_HIT_QUERY_PARAMETER}={FLAG_ON}")
    if no_headings:
        flags.append(f"{NO_HEADINGS_QUERY_PARAMETER}={FLAG_ON}")
    return f"{path}?{'&'.join(flags)}" if flags else path


def application_path(token: str) -> str:
    """Path of an application's status page."""
    return APPLICATION_PATH_TEMPLATE.format(token=token)
