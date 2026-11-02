"""Green UI rules for the candidate pages: 50 KB budget, no scripts or external assets, autoescape."""

from datetime import UTC, datetime, timedelta

import pytest
from candidate_support import (
    confirm_form,
    create_open_posting,
    multipart_of_exact_size,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pdf_fixtures import build_pdf
from pipeline_support import ControlledEmbedder, build_cv_pdf
from test_web_green_ui import FORBIDDEN_ASSET_PATTERN, PAGE_BUDGET_BYTES, gzip_size

from app.privacy.retention import RetentionContext, close_posting
from app.web.dependencies import get_now
from app.web.paths import STYLESHEET_URL

ESCAPED_LINE = "<b>bold</b> & <i>italic</i>"
HOSTILE_TITLE = "<u>Hostile</u> title"


def every_candidate_page(
    client: TestClient, application: FastAPI, embedder: ControlledEmbedder
) -> dict[str, bytes]:
    """The body of each candidate page and its error variants."""
    job_id = create_open_posting(application, embedder, "Backend Developer")
    closed_id = create_open_posting(application, embedder, "Closed Job")
    with application.state.session_factory() as session:
        close_posting(RetentionContext(session, datetime.now(UTC), 30), closed_id)
        session.commit()
    pdf = build_cv_pdf("alpha")
    token = token_of(upload_cv(client, job_id, pdf))
    pages = {
        "home": client.get("/").content,
        "job": client.get(f"/jobs/{job_id}").content,
        "job closed": client.get(f"/jobs/{closed_id}").content,
        "job not found": client.get("/jobs/999").content,
        "upload error": upload_cv(client, job_id, b"not a pdf").content,
        "preview": client.get(f"/drafts/{token}?cached=1&noheadings=1").content,
        "preview with errors": client.post(
            f"/drafts/{token}", data=confirm_form([("summary", "x")], years_experience="abc")
        ).content,
        "status draft": client.get(f"/applications/{token}").content,
        "status deleted": client.get("/applications/unknown").content,
    }
    body, content_type = multipart_of_exact_size(pdf, 3 * 1024 * 1024 + 1)
    pages["too large"] = client.post(
        f"/jobs/{job_id}/apply", content=body, headers={"content-type": content_type}
    ).content
    application.dependency_overrides[get_now] = lambda: datetime.now(UTC) + timedelta(hours=25)
    pages["draft expired"] = client.get(f"/drafts/{token}").content
    return pages


def test_every_candidate_page_is_under_50_kb_gzipped_with_the_css(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """HTML plus CSS per page, compressed from the actual response bodies."""
    css = candidate_client.get(STYLESHEET_URL).content
    pages = every_candidate_page(candidate_client, candidate_app, candidate_embedder)
    assert len(pages) >= 10
    for name, body in pages.items():
        assert gzip_size(body + css) < PAGE_BUDGET_BYTES, name


def test_no_candidate_page_has_a_script_or_an_external_resource(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """No JavaScript, no images, no absolute or protocol-relative asset URLs."""
    pages = every_candidate_page(candidate_client, candidate_app, candidate_embedder)
    for name, body in pages.items():
        text = body.decode("utf-8")
        assert "<script" not in text.lower(), name
        assert FORBIDDEN_ASSET_PATTERN.search(text) is None, name


def test_cv_derived_text_and_posting_titles_are_escaped(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """Markup in a CV section or a posting title is shown as text, never as HTML."""
    job_id = create_open_posting(candidate_app, candidate_embedder, HOSTILE_TITLE)
    lines = [
        "Jane Doe", "jane.doe@example.com +90 532 123 45 67",
        "SUMMARY", f"Candidate profile {ESCAPED_LINE} seeking backend engineering roles.",
        "EXPERIENCE", "Software Engineer, Example Corp, 2016 - 2024",
        "Built services and databases for finance teams across the country.",
        "SKILLS", "Python SQL Docker",
    ]
    token = token_of(upload_cv(candidate_client, job_id, build_pdf(lines)))
    for page in (
        candidate_client.get(f"/drafts/{token}").text,
        candidate_client.get(f"/jobs/{job_id}").text,
        candidate_client.get("/").text,
        candidate_client.get(f"/applications/{token}").text,
    ):
        assert "<b>bold</b>" not in page and "<u>Hostile</u>" not in page
    preview = candidate_client.get(f"/drafts/{token}").text
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; &lt;i&gt;italic&lt;/i&gt;" in preview
    assert "&lt;u&gt;Hostile&lt;/u&gt; title" in candidate_client.get(f"/jobs/{job_id}").text


@pytest.mark.parametrize("path", ["/drafts/some-token", "/applications/some-token"])
def test_private_pages_are_not_cacheable(candidate_client: TestClient, path: str) -> None:
    """Pages behind a token carry Cache-Control: no-store (a draft or unknown token alike)."""
    response = candidate_client.get(path, follow_redirects=True)
    assert response.headers["cache-control"] == "no-store"
