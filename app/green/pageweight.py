"""Page weight (SPEC 9.2): the transferred bytes of every response, per route.

`PageWeightMiddleware` is a pure ASGI middleware. It must sit outside the gzip
middleware so that it counts the compressed bytes that actually travel. Figures
are kept in memory only (no table) and grouped by the route's path template,
for example `/employer/jobs/{job_id}`, so ids and tokens never become keys.

KVKK: only a path template, a count and a byte total are recorded.
"""

from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

HTTP_SCOPE_TYPE = "http"
RESPONSE_BODY_TYPE = "http.response.body"
ROUTE_SCOPE_KEY = "route"
PATH_TEMPLATE_ATTRIBUTE = "path_format"
UNMATCHED_LABEL = "(eşleşmeyen yol)"


@dataclass(frozen=True)
class PageWeightRow:
    """The figures of one path template.

    Attributes:
        path: The route's path template.
        response_count: How many responses were counted.
        total_bytes: Body bytes over all those responses.
    """

    path: str
    response_count: int
    total_bytes: int

    @property
    def average_bytes(self) -> float:
        """Mean body bytes per response."""
        return self.total_bytes / self.response_count


class PageWeightStats:
    """Thread-safe in-memory totals per path template."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._totals: dict[str, tuple[int, int]] = {}

    def record(self, path_template: str, body_bytes: int) -> None:
        """Count one response of `body_bytes` bytes under a path template."""
        with self._lock:
            count, total = self._totals.get(path_template, (0, 0))
            self._totals[path_template] = (count + 1, total + body_bytes)

    def snapshot(self) -> list[PageWeightRow]:
        """The rows so far, sorted by path."""
        with self._lock:
            items = sorted(self._totals.items())
        return [PageWeightRow(path, count, total) for path, (count, total) in items]


def resolve_path_template(scope: Scope) -> str:
    """The matched route's path template, or a fixed label when nothing matched.

    The raw request path is never used: it could carry ids or tokens.
    """
    route = scope.get(ROUTE_SCOPE_KEY)
    template = getattr(route, PATH_TEMPLATE_ATTRIBUTE, None)
    return template if isinstance(template, str) else UNMATCHED_LABEL


class PageWeightMiddleware:
    """Records the size of every HTTP response body in `stats`.

    Args:
        app: The wrapped ASGI application.
        stats: Where the figures are kept.
        mount_templates: Path prefixes of mounted sub-applications (for example
            static files) mapped to the label their responses are counted under.
    """

    def __init__(
        self,
        app: ASGIApp,
        stats: PageWeightStats,
        mount_templates: dict[str, str] | None = None,
    ) -> None:
        self._app = app
        self._stats = stats
        self._mount_templates = mount_templates or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request on and add up the body bytes of its response."""
        if scope["type"] != HTTP_SCOPE_TYPE:
            await self._app(scope, receive, send)
            return
        body_bytes = 0
        request_path = scope.get("path", "")

        async def counting_send(message: Message) -> None:
            nonlocal body_bytes
            if message["type"] == RESPONSE_BODY_TYPE:
                body_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self._app(scope, receive, counting_send)
        finally:
            self._stats.record(self._label_of(scope, request_path), body_bytes)

    def _label_of(self, scope: Scope, request_path: str) -> str:
        """The label of the request: a mount label, else the route template.

        `request_path` is read before routing, because a mount may rewrite the
        scope's path while the request is handled.
        """
        for prefix, label in self._mount_templates.items():
            if request_path.startswith(prefix):
                return label
        return resolve_path_template(scope)
