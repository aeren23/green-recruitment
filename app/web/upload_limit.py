"""Request-body size limit for the candidate's form posts (SC-02, SPEC 9.1).

Starlette spools a multipart upload above 1 MB to a temporary file, which would
break the rule that a CV never touches the disk. This pure-ASGI wrapper counts
the body bytes of the limited paths as they arrive and answers 413 with a
Turkish page as soon as the limit is passed, before Starlette parses further.
A declared `Content-Length` over the limit is rejected without reading the
body at all. The limit is `MAX_UPLOAD_BYTES` and applies to the whole request
body (multipart framing included), which is the stricter reading.

`tests/test_candidate_upload_limit.py` asserts that the limit never exceeds
Starlette's in-memory threshold, so a Starlette upgrade cannot silently
re-open the disk path.
"""

import re
from collections.abc import Callable
from typing import Any

from starlette.requests import ClientDisconnect, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.web import messages, paths
from app.web.templating import render_page

LIMITED_METHOD = "POST"
LIMITED_PATH_PATTERN = re.compile(r"^/(?:jobs/\d+/apply|drafts/[^/]+)$")
CONTENT_LENGTH_HEADER = b"content-length"
TOO_LARGE_STATUS = 413
NOTICE_TEMPLATE = "candidate/notice.html"
NO_DECLARED_LENGTH = 0
HTTP_SCOPE_TYPE = "http"
REQUEST_MESSAGE = "http.request"
DISCONNECT_MESSAGE: Message = {"type": "http.disconnect"}

LimitProvider = Callable[[Scope], int]


def limit_from_app_settings(scope: Scope) -> int:
    """The limit of the running application: `MAX_UPLOAD_BYTES` from its settings."""
    return scope["app"].state.settings.max_upload_bytes


def is_limited_request(scope: Scope) -> bool:
    """Whether the request is a form post whose body size is capped."""
    return (
        scope["type"] == HTTP_SCOPE_TYPE
        and scope["method"] == LIMITED_METHOD
        and LIMITED_PATH_PATTERN.match(scope["path"]) is not None
    )


def declared_length(scope: Scope) -> int:
    """The `Content-Length` the client declared, 0 when absent or unreadable."""
    for name, value in scope["headers"]:
        if name == CONTENT_LENGTH_HEADER:
            return int(value) if value.isdigit() else NO_DECLARED_LENGTH
    return NO_DECLARED_LENGTH


async def send_too_large(scope: Scope, send: Send, limit: int) -> None:
    """Answer 413 with the Turkish page; the request body is not read."""
    values: dict[str, Any] = {
        "heading": messages.UPLOAD_TOO_LARGE_HEADING,
        "message": messages.UPLOAD_TOO_LARGE.format(limit=messages.format_size(limit)),
        "link_href": paths.HOME_PATH,
        "link_text": messages.BACK_TO_POSTINGS,
    }
    response = render_page(Request(scope), NOTICE_TEMPLATE, values, TOO_LARGE_STATUS)

    async def no_body() -> Message:
        return DISCONNECT_MESSAGE

    await response(scope, no_body, send)


class BodyGuard:
    """Counts request-body bytes and cuts the request off once the limit is passed.

    After the 413 has been sent, the application sees a disconnect (so the
    multipart parser stops) and whatever it tries to send is dropped.
    """

    def __init__(self, scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        self._scope = scope
        self._receive = receive
        self._send = send
        self._limit = limit
        self._received = 0
        self.rejected = False

    async def receive(self) -> Message:
        """Pass the next body chunk on, or reject the request when it is too big."""
        if self.rejected:
            return DISCONNECT_MESSAGE
        message = await self._receive()
        if message["type"] == REQUEST_MESSAGE:
            self._received += len(message.get("body", b""))
            if self._received > self._limit:
                self.rejected = True
                await send_too_large(self._scope, self._send, self._limit)
                return DISCONNECT_MESSAGE
        return message

    async def send(self, message: Message) -> None:
        """Forward the application's response unless the request was rejected."""
        if not self.rejected:
            await self._send(message)


class UploadSizeLimitMiddleware:
    """ASGI middleware that caps the body of the candidate's form posts."""

    def __init__(self, app: ASGIApp, limit_of: LimitProvider = limit_from_app_settings) -> None:
        self.app = app
        self.limit_of = limit_of

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject an oversized limited request, pass everything else through."""
        if not is_limited_request(scope):
            await self.app(scope, receive, send)
            return
        limit = self.limit_of(scope)
        if declared_length(scope) > limit:
            await send_too_large(scope, send, limit)
            return
        guard = BodyGuard(scope, receive, send, limit)
        try:
            await self.app(scope, guard.receive, guard.send)
        except ClientDisconnect:
            if not guard.rejected:
                raise
