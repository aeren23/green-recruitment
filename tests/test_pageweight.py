"""Page weight middleware (SPEC 9.2): path templates, compressed bytes, averages."""

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from app.green.pageweight import (
    UNMATCHED_LABEL,
    PageWeightMiddleware,
    PageWeightStats,
)

LARGE_BODY = "green hiring pipeline " * 2000
SMALL_BODY = "ok"
GZIP_MINIMUM_BYTES = 500


def build_app(stats: PageWeightStats) -> FastAPI:
    """An app whose pages are gzipped and counted outside the gzip layer."""
    application = FastAPI()

    @application.get("/items/{item_id}")
    def item(item_id: int) -> PlainTextResponse:
        return PlainTextResponse(SMALL_BODY)

    @application.get("/large")
    def large() -> PlainTextResponse:
        return PlainTextResponse(LARGE_BODY)

    application.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_BYTES)
    application.add_middleware(PageWeightMiddleware, stats=stats)
    return application


def rows_by_path(stats: PageWeightStats) -> dict[str, tuple[int, int]]:
    """Snapshot as `{path: (count, total bytes)}`."""
    return {row.path: (row.response_count, row.total_bytes) for row in stats.snapshot()}


def test_paths_are_recorded_as_templates_never_as_raw_ids() -> None:
    """Different ids land under one template; an unknown path gets a fixed label."""
    stats = PageWeightStats()
    client = TestClient(build_app(stats))
    client.get("/items/17")
    client.get("/items/4242")
    client.get("/no/such/secret-token-abc")
    recorded = rows_by_path(stats)
    assert set(recorded) == {"/items/{item_id}", UNMATCHED_LABEL}
    assert recorded["/items/{item_id}"][0] == 2
    assert not any("4242" in path or "secret" in path for path in recorded)


def test_compressed_bytes_are_counted() -> None:
    """The recorded size is the gzip size, smaller than the page itself."""
    stats = PageWeightStats()
    client = TestClient(build_app(stats))
    response = client.get("/large")
    assert response.headers["content-encoding"] == "gzip"
    _, total = rows_by_path(stats)["/large"]
    assert 0 < total < len(LARGE_BODY.encode())


def test_average_is_total_over_count() -> None:
    """The average of a path is its byte total divided by its response count."""
    stats = PageWeightStats()
    stats.record("/page", 100)
    stats.record("/page", 300)
    (row,) = stats.snapshot()
    assert (row.response_count, row.total_bytes, row.average_bytes) == (2, 400, 200.0)


def test_mounted_files_use_the_configured_label() -> None:
    """A mounted prefix is counted under its label, not under a raw file name."""
    stats = PageWeightStats()
    application = build_app(stats)
    application.user_middleware.clear()
    application.middleware_stack = None
    application.add_middleware(
        PageWeightMiddleware, stats=stats, mount_templates={"/static/": "/static/{file}"}
    )
    TestClient(application).get("/static/app.css")
    assert set(rows_by_path(stats)) == {"/static/{file}"}
