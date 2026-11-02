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
