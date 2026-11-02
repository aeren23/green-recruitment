"""Wiring of the web layer into the FastAPI application (SPEC 9.2).

GZip compression, the upload size limit, the cacheable static CSS, the employer
login guard and the routers. Later tasks add their routers here.
"""

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from app.web.auth import EmployerLoginRequired, redirect_to_login
from app.web.paths import STATIC_URL_PATH
from app.web.routes_candidate import candidate_router
from app.web.routes_employer import employer_router, login_router
from app.web.routes_employer_postings import postings_action_router
from app.web.static_files import CachedStaticFiles
from app.web.templating import STATIC_DIRECTORY
from app.web.upload_limit import UploadSizeLimitMiddleware

GZIP_MINIMUM_SIZE_BYTES = 500
STATIC_MOUNT_NAME = "static"


def install_web(application: FastAPI) -> None:
    """Add middleware, static files, the login guard handler and the routers."""
    application.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE_BYTES)
    # Added last, so it is the outermost layer and sees the raw request body first.
    application.add_middleware(UploadSizeLimitMiddleware)
    application.mount(
        STATIC_URL_PATH,
        CachedStaticFiles(directory=STATIC_DIRECTORY),
        name=STATIC_MOUNT_NAME,
    )
    application.add_exception_handler(EmployerLoginRequired, redirect_to_login)
    application.include_router(login_router)
    application.include_router(employer_router)
    application.include_router(postings_action_router)
    application.include_router(candidate_router)
