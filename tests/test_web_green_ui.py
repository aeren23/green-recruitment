"""Green UI rules of SPEC 9.2: page weight, gzip, CSS cache header, no external assets."""

import gzip
import re
from pathlib import Path

import pytest
from conftest import FakeEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from web_support import add_postings, log_in, submit_job

from app.web.paths import (
    EMPLOYER_HOME_PATH,
    EMPLOYER_LOGIN_PATH,
    EMPLOYER_NEW_JOB_PATH,
    STYLESHEET_URL,
)
from app.web.static_files import CACHE_CONTROL_VALUE
from app.web.templating import STATIC_DIRECTORY, TEMPLATE_DIRECTORY

PAGE_BUDGET_BYTES = 50 * 1024
GZIP_MINIMUM_BYTES = 500
FORBIDDEN_ASSET_PATTERN = re.compile(
    r"https?://|(?:src|href)\s*=\s*[\"']?//|<img\b|<script\b[^>]*\bsrc\b|@import|@font-face|url\(",
    re.IGNORECASE,
)


def gzip_size(content: bytes) -> int:
    """Size of the content after gzip compression."""
    return len(gzip.compress(content))


@pytest.fixture
def employer(
    web_app: FastAPI, web_client: TestClient, fake_embedder: FakeEmbedder
) -> TestClient:
    """A logged-in client over a full first page (20 of 25 postings)."""
    add_postings(web_app, fake_embedder, 25)
    log_in(web_client)
    return web_client


def every_t12_page(client: TestClient) -> dict[str, bytes]:
    """The body of each T-12 page, including a form with errors and a full list."""
    client.cookies.clear()
    pages = {"login": client.get(EMPLOYER_LOGIN_PATH).content}
    log_in(client)
    pages["list"] = client.get(EMPLOYER_HOME_PATH).content
    pages["list page 2"] = client.get(EMPLOYER_HOME_PATH, params={"page": 2}).content
    pages["form"] = client.get(EMPLOYER_NEW_JOB_PATH).content
    pages["form with errors"] = submit_job(client, title="", requirements=()).content
    return pages


def test_every_page_with_the_css_is_under_50_kb_gzipped(employer: TestClient) -> None:
    """HTML + CSS per page, each compressed from the actual response bodies."""
    css = employer.get(STYLESHEET_URL).content
    for name, body in every_t12_page(employer).items():
        total = gzip_size(body) + gzip_size(css)
        assert total < PAGE_BUDGET_BYTES, f"{name}: {total} bytes"


def test_large_responses_are_gzipped_and_tiny_ones_are_not(employer: TestClient) -> None:
    """GZipMiddleware is on for pages and CSS; the health body is below the minimum."""
    for path in (EMPLOYER_HOME_PATH, EMPLOYER_NEW_JOB_PATH, STYLESHEET_URL):
        response = employer.get(path, headers={"Accept-Encoding": "gzip"})
        assert response.headers["content-encoding"] == "gzip", path
        assert response.headers["vary"].lower().find("accept-encoding") >= 0
    health = employer.get("/health", headers={"Accept-Encoding": "gzip"})
    assert len(health.content) < GZIP_MINIMUM_BYTES
    assert "content-encoding" not in health.headers


def test_css_is_cacheable_for_a_day_and_served_as_css(web_client: TestClient) -> None:
    """`Cache-Control: public, max-age=86400` on the single stylesheet."""
    response = web_client.get(STYLESHEET_URL)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert CACHE_CONTROL_VALUE == "public, max-age=86400"
    assert response.headers["content-type"].startswith("text/css")


def test_cached_css_answers_a_conditional_request_with_304_and_the_header(
    web_client: TestClient,
) -> None:
    """A revalidation keeps the cache header too."""
    first = web_client.get(STYLESHEET_URL)
    second = web_client.get(STYLESHEET_URL, headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 304
    assert second.headers["cache-control"] == CACHE_CONTROL_VALUE


def test_there_is_exactly_one_stylesheet() -> None:
    """Green rule 3: a single CSS file."""
    assert [path.name for path in STATIC_DIRECTORY.rglob("*") if path.is_file()] == ["app.css"]


def test_css_supports_dark_mode_with_a_system_font_stack() -> None:
    """`prefers-color-scheme` and a system font, nothing downloaded."""
    css = (STATIC_DIRECTORY / "app.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css
    assert "system-ui" in css


def source_files() -> list[Path]:
    """Every template and static file."""
    return [
        path
        for directory in (TEMPLATE_DIRECTORY, STATIC_DIRECTORY)
        for path in directory.rglob("*")
        if path.is_file()
    ]


def test_sources_reference_no_external_or_forbidden_assets() -> None:
    """No URL to another host, image, external script, `@import` or webfont."""
    offenders = {
        path.name: FORBIDDEN_ASSET_PATTERN.findall(path.read_text(encoding="utf-8"))
        for path in source_files()
    }
    assert {name: found for name, found in offenders.items() if found} == {}
    assert len(source_files()) >= 7


def test_rendered_pages_carry_no_script_image_or_external_link(employer: TestClient) -> None:
    """The responses themselves, not only the sources."""
    for name, body in every_t12_page(employer).items():
        text = body.decode("utf-8")
        assert "<script" not in text.lower(), name
        assert FORBIDDEN_ASSET_PATTERN.search(text) is None, name


@pytest.mark.parametrize(
    "offending_source",
    [
        '<img src="logo.png">',
        '<script src="app.js"></script>',
        '<link href="https://cdn.example.com/x.css">',
        '<script src="//cdn.example.com/x.js">',
        '@import "other.css";',
        "@font-face { font-family: X; }",
        "body { background: url(bg.png); }",
    ],
)
def test_the_scan_pattern_flags_each_forbidden_construct(offending_source: str) -> None:
    """A check that cannot fail proves nothing."""
    assert FORBIDDEN_ASSET_PATTERN.search(offending_source) is not None
