"""Static file serving with the cache header SPEC 9.2 asks for."""

from typing import Any

from starlette.responses import Response
from starlette.staticfiles import StaticFiles

CACHE_CONTROL_HEADER = "Cache-Control"
CACHE_CONTROL_VALUE = "public, max-age=86400"


class CachedStaticFiles(StaticFiles):
    """`StaticFiles` that marks every served file as cacheable for a day."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        """Build the normal file response, then add `Cache-Control`."""
        response = super().file_response(*args, **kwargs)
        response.headers[CACHE_CONTROL_HEADER] = CACHE_CONTROL_VALUE
        return response
