"""Helpers for the candidate-flow tests: postings, uploads, forms and raw ASGI calls.

The embedder is `ControlledEmbedder` and the LLM is `FakeLlm`, so nothing here
needs a GPU or Ollama. Postings are created through the real posting service.
"""

import asyncio
import html
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pipeline_support import ControlledEmbedder
from sqlalchemy import select

from app.models import Application, DecisionRecord, JobPosting
from app.postings import PostingInput, RequirementInput, create_posting

POSTING_TIME = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
DEFAULT_REQUIREMENTS = ("Building REST APIs", "Relational databases and SQL")
UPLOAD_FIELD = "cv"
PDF_CONTENT_TYPE = "application/pdf"
TEXTAREA_PATTERN = re.compile(r"<textarea[^>]*>(.*?)</textarea>", re.DOTALL)
INPUT_VALUE_PATTERN = r'<input id="{name}" name="{name}" value="([^"]*)"'
LOCATION_PREFIX = "/drafts/"


def create_open_posting(
    application: FastAPI, embedder: ControlledEmbedder, title: str, **filters: Any
) -> int:
    """Create an open posting through the posting service; return its id."""
    posting_input = PostingInput(
        title=title,
        description=f"{title} description.",
        requirements=[RequirementInput(text, 2) for text in DEFAULT_REQUIREMENTS],
        created_at=POSTING_TIME,
        **filters,
    )
    with application.state.session_factory() as session:
        posting = create_posting(session, embedder, posting_input)
        session.commit()
        return posting.id


def upload_cv(client: TestClient, job_id: int, pdf: bytes) -> Response:
    """POST a CV to the apply route."""
    files = {UPLOAD_FIELD: ("cv.pdf", pdf, PDF_CONTENT_TYPE)}
    return client.post(f"/jobs/{job_id}/apply", files=files)


def token_of(response: Response) -> str:
    """The draft token in the redirect of a successful upload."""
    location = response.headers["location"]
    assert location.startswith(LOCATION_PREFIX), location
    return location[len(LOCATION_PREFIX):].split("?")[0]


def section_texts_of(page: str) -> list[str]:
    """The (unescaped) section texts of a preview page, blank rows included."""
    return [html.unescape(text).strip() for text in TEXTAREA_PATTERN.findall(page)]


def input_value_of(page: str, name: str) -> str:
    """The unescaped value of a named single-line input of the preview page."""
    match = re.search(INPUT_VALUE_PATTERN.format(name=name), page)
    assert match is not None, name
    return html.unescape(match.group(1))


def confirm_form(
    sections: Sequence[tuple[str, str]], **fields: str
) -> dict[str, Any]:
    """Preview-form data: section (type, text) pairs and the single-line fields."""
    data: dict[str, Any] = {
        "section_type": [row_type for row_type, _ in sections],
        "section_text": [text for _, text in sections],
        "years_experience": "",
        "languages": "",
        "location": "",
        "contact_name": "",
        "contact_email": "",
        "contact_phone": "",
    }
    data.update(fields)
    return data


def stored_applications(application: FastAPI) -> list[Application]:
    """All applications in the app's database."""
    with application.state.session_factory() as session:
        return list(session.scalars(select(Application).order_by(Application.id)))


def stored_decisions(application: FastAPI) -> list[DecisionRecord]:
    """All decision records in the app's database."""
    with application.state.session_factory() as session:
        return list(session.scalars(select(DecisionRecord).order_by(DecisionRecord.id)))


def posting_status(application: FastAPI, job_id: int) -> str:
    """The stored status value of a posting."""
    with application.state.session_factory() as session:
        return session.get(JobPosting, job_id).status.value


def application_summary(application: FastAPI, token: str) -> str | None:
    """The stored summary of the application behind a token."""
    with application.state.session_factory() as session:
        row = session.scalar(select(Application).where(Application.draft_token == token))
        return row.summary


def build_multipart_body(pdf: bytes, boundary: str) -> bytes:
    """A multipart body holding one `cv` file part."""
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{UPLOAD_FIELD}"; filename="cv.pdf"\r\n'
        f"Content-Type: {PDF_CONTENT_TYPE}\r\n\r\n"
    ).encode()
    return head + pdf + f"\r\n--{boundary}--\r\n".encode()


def multipart_of_exact_size(pdf: bytes, total_size: int) -> tuple[bytes, str]:
    """A multipart body of exactly `total_size` bytes; the PDF gets trailing newlines."""
    boundary = "candidate-test-boundary"
    base = build_multipart_body(pdf, boundary)
    padding = total_size - len(base)
    assert padding >= 0, "total size is smaller than the PDF itself"
    body = build_multipart_body(pdf + b"\n" * padding, boundary)
    assert len(body) == total_size
    return body, f"multipart/form-data; boundary={boundary}"


class AsgiRecorder:
    """Drives an ASGI app by hand and records what it sent and how often it read."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = list(chunks)
        self.sent: list[dict[str, Any]] = []
        self.receive_calls = 0

    async def receive(self) -> dict[str, Any]:
        """The next body chunk; a disconnect once the chunks are used up."""
        self.receive_calls += 1
        if not self._chunks:
            return {"type": "http.disconnect"}
        body = self._chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(self._chunks)}

    async def send(self, message: dict[str, Any]) -> None:
        """Record a message the application sent."""
        self.sent.append(message)

    @property
    def status(self) -> int | None:
        """The status of the response that was started, if any."""
        starts = [item for item in self.sent if item["type"] == "http.response.start"]
        return starts[0]["status"] if starts else None


def call_asgi(
    application: FastAPI,
    path: str,
    headers: dict[str, str],
    recorder: AsgiRecorder,
) -> None:
    """POST to the application with a hand-built scope (the lifespan must be running)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(application(scope, recorder.receive, recorder.send))


def urlencoded_form(data: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    """A urlencoded body and its content-type header."""
    body = urlencode(data, doseq=True).encode()
    return body, {"content-type": "application/x-www-form-urlencoded"}


SELECTED_TYPE_PATTERN = re.compile(r'<option value="(\w+)" selected>')


def selected_types_of(page: str) -> list[str]:
    """The selected section type of every row of a preview page, blank rows included."""
    return SELECTED_TYPE_PATTERN.findall(page)


def build_person_pdf(name: str, email: str, phone: str, tag: str) -> bytes:
    """A synthetic CV for one distinct fictional person."""
    from pdf_fixtures import build_pdf

    lines = [
        name,
        f"{email} {phone}",
        "SUMMARY",
        f"Candidate profile {tag} seeking backend engineering roles.",
        "EXPERIENCE",
        "Software Engineer, Example Corp, 2016 - 2024",
        f"Built services and databases for finance teams {tag} QSCORE50",
        "SKILLS",
        "Python SQL Docker",
        "EDUCATION",
        "BSc Computer Engineering, Example University, 2010 - 2014",
    ]
    return build_pdf(lines)
