"""Helpers for the web tests: logging in, valid form data and seeded postings."""

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.models import JobPosting
from app.postings import PostingInput, RequirementInput, create_postings
from app.web.paths import EMPLOYER_LOGIN_PATH, EMPLOYER_NEW_JOB_PATH
from conftest import DUMMY_EMPLOYER_TOKEN, FakeEmbedder

NOW = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
ROW_COUNT = 6
DEFAULT_REQUIREMENTS = ("Building REST APIs", "Relational databases and SQL")


def log_in(client: TestClient) -> None:
    """Log in with the right token; the client keeps the cookie."""
    response = client.post(EMPLOYER_LOGIN_PATH, data={"token": DUMMY_EMPLOYER_TOKEN})
    assert response.status_code == 303


def form_data(
    requirements: tuple[tuple[str, str], ...] = (
        (DEFAULT_REQUIREMENTS[0], "3"),
        (DEFAULT_REQUIREMENTS[1], "2"),
    ),
    **fields: str,
) -> dict[str, object]:
    """Valid new-posting form data with six requirement rows (blank-padded)."""
    rows = list(requirements) + [("", "2")] * (ROW_COUNT - len(requirements))
    data: dict[str, object] = {
        "title": "Data Engineer",
        "description": "Build pipelines.",
        "languages": "",
        "location": "",
        "min_years": "",
        "requirement_text": [text for text, _ in rows],
        "requirement_weight": [weight for _, weight in rows],
    }
    data.update(fields)
    return data


def submit_job(client: TestClient, **kwargs: object):
    """POST the new-posting form; `kwargs` override fields of `form_data`."""
    return client.post(EMPLOYER_NEW_JOB_PATH, data=form_data(**kwargs))  # type: ignore[arg-type]


def add_postings(application: FastAPI, embedder: FakeEmbedder, count: int) -> None:
    """Insert `count` postings directly, oldest first, titled `Posting 01`, ..."""
    inputs = [
        PostingInput(
            title=f"Posting {number:02d}",
            description="",
            requirements=[RequirementInput("Python programming", 2)],
            created_at=NOW + timedelta(minutes=number),
        )
        for number in range(1, count + 1)
    ]
    with application.state.session_factory() as session:
        create_postings(session, embedder, inputs)
        session.commit()


def stored_postings(application: FastAPI) -> list[JobPosting]:
    """All postings in the app's database, oldest first, requirements loaded."""
    with application.state.session_factory() as session:
        postings = list(session.scalars(select(JobPosting).order_by(JobPosting.id)))
        for posting in postings:
            len(posting.requirements)  # load before the session closes
        session.expunge_all()
        return postings
