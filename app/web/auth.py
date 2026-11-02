"""Employer authentication (SPEC 2, SC-04): a token form and a derived cookie.

This is deliberately not real authentication. The employer types the shared
`EMPLOYER_TOKEN`; on success the browser receives a cookie whose value is an
HMAC of a fixed purpose label keyed with the token, never the token itself.
A request is authenticated when its cookie equals that derived value.

The cookie path is `/` (not `/employer`) so that `POST /green/retention`
(SPEC 9.1, T-16) can check the same cookie. The cookie is `HttpOnly` and
`SameSite=Lax`; it has no `Secure` flag because the demo is served over plain
HTTP on localhost.
"""

import hashlib
import hmac
import secrets

from fastapi import Depends, Request
from fastapi.responses import RedirectResponse, Response

from app.config import Settings
from app.web.dependencies import get_settings
from app.web.paths import EMPLOYER_LOGIN_PATH

EMPLOYER_COOKIE_NAME = "employer_session"
EMPLOYER_COOKIE_PATH = "/"
EMPLOYER_COOKIE_SAMESITE = "lax"
EMPLOYER_COOKIE_MAX_AGE_SECONDS = 12 * 60 * 60
SESSION_PURPOSE_LABEL = b"green-hiring/employer-session/v1"
TEXT_ENCODING = "utf-8"
REDIRECT_AFTER_POST = 303


class EmployerLoginRequired(Exception):
    """Raised by the guard when a request has no valid employer cookie."""


def tokens_match(supplied: str, expected: str) -> bool:
    """Compare two tokens in constant time.

    Both sides are encoded first: `secrets.compare_digest` raises on non-ASCII
    `str`, and a Turkish character in a wrong token must not cause an error.
    """
    return secrets.compare_digest(
        supplied.encode(TEXT_ENCODING), expected.encode(TEXT_ENCODING)
    )


def derive_session_value(employer_token: str) -> str:
    """The cookie value for a token: HMAC-SHA256 of the purpose label, hex."""
    return hmac.new(
        employer_token.encode(TEXT_ENCODING), SESSION_PURPOSE_LABEL, hashlib.sha256
    ).hexdigest()


def has_valid_session(cookie_value: str | None, settings: Settings) -> bool:
    """True when the cookie carries the value derived from the current token."""
    if cookie_value is None:
        return False
    return tokens_match(cookie_value, derive_session_value(settings.employer_token))


def is_employer(request: Request, settings: Settings) -> bool:
    """True when the request carries a valid employer cookie."""
    return has_valid_session(request.cookies.get(EMPLOYER_COOKIE_NAME), settings)


def require_employer(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Guard dependency: let employers through, send everyone else to the login.

    Raises:
        EmployerLoginRequired: When the cookie is missing or does not match.
    """
    if not is_employer(request, settings):
        raise EmployerLoginRequired


def redirect_to_login(request: Request, exc: Exception) -> Response:
    """Exception handler turning `EmployerLoginRequired` into a redirect."""
    return RedirectResponse(EMPLOYER_LOGIN_PATH, status_code=REDIRECT_AFTER_POST)


def set_session_cookie(response: Response, settings: Settings) -> None:
    """Attach the employer cookie to a response."""
    response.set_cookie(
        EMPLOYER_COOKIE_NAME,
        derive_session_value(settings.employer_token),
        max_age=EMPLOYER_COOKIE_MAX_AGE_SECONDS,
        path=EMPLOYER_COOKIE_PATH,
        httponly=True,
        samesite=EMPLOYER_COOKIE_SAMESITE,
    )


def clear_session_cookie(response: Response) -> None:
    """Expire the employer cookie (same path and flags as when it was set)."""
    response.delete_cookie(
        EMPLOYER_COOKIE_NAME,
        path=EMPLOYER_COOKIE_PATH,
        httponly=True,
        samesite=EMPLOYER_COOKIE_SAMESITE,
    )
