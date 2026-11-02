"""KVKK checks on the candidate pages: token masking in the access log and data isolation."""

import logging
from collections.abc import Iterator

import pytest
from candidate_support import (
    build_person_pdf,
    confirm_form,
    create_open_posting,
    selected_types_of,
    section_texts_of,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import ControlledEmbedder

from app.web.access_log import (
    ACCESS_LOGGER_NAME,
    TokenMaskingFilter,
    install_access_log_masking,
    mask_tokens,
)

SECRET_TOKEN = "k3Y_9qZxAbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
UVICORN_ACCESS_FORMAT = '%s - "%s %s HTTP/%s" %d'


class ListHandler(logging.Handler):
    """Collects the formatted text of every record that reaches it."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the final message, as a handler's formatter would print it."""
        self.lines.append(record.getMessage())


@pytest.fixture
def access_lines() -> Iterator[list[str]]:
    """The access logger without any filter left over from other tests, plus a collector."""
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    for existing in [item for item in access_logger.filters if isinstance(item, TokenMaskingFilter)]:
        access_logger.removeFilter(existing)
    handler = ListHandler()
    access_logger.addHandler(handler)
    previous_level = access_logger.level
    access_logger.setLevel(logging.INFO)
    yield handler.lines
    access_logger.setLevel(previous_level)
    access_logger.removeHandler(handler)
    for existing in [item for item in access_logger.filters if isinstance(item, TokenMaskingFilter)]:
        access_logger.removeFilter(existing)


def log_access(path: str) -> None:
    """Write one record the way uvicorn's access logger does."""
    logging.getLogger(ACCESS_LOGGER_NAME).info(
        UVICORN_ACCESS_FORMAT, "127.0.0.1:5000", "GET", path, "1.1", 200
    )


def test_starting_the_app_masks_draft_and_application_tokens_in_the_access_log(
    access_lines: list[str], candidate_app: FastAPI
) -> None:
    """After start-up an access record never carries the token segment."""
    with TestClient(candidate_app):
        log_access(f"/drafts/{SECRET_TOKEN}")
        log_access(f"/applications/{SECRET_TOKEN}?page=2")
        log_access("/jobs/3")
    assert all(SECRET_TOKEN not in line for line in access_lines)
    assert any('GET /drafts/*** HTTP/1.1" 200' in line for line in access_lines)
    assert any("/applications/***?page=2" in line for line in access_lines)
    assert any("/jobs/3" in line for line in access_lines)


def test_without_the_filter_the_token_would_be_logged(access_lines: list[str]) -> None:
    """Control: the fixture really starts unfiltered, so the masking test means something."""
    log_access(f"/drafts/{SECRET_TOKEN}")
    assert any(SECRET_TOKEN in line for line in access_lines)


def test_installing_twice_adds_one_filter() -> None:
    """Idempotent: several app starts in one process do not stack filters."""
    install_access_log_masking()
    install_access_log_masking()
    filters = logging.getLogger(ACCESS_LOGGER_NAME).filters
    assert sum(isinstance(item, TokenMaskingFilter) for item in filters) == 1


@pytest.mark.parametrize(
    ("raw", "masked"),
    [
        ("/drafts/abc-DEF_123", "/drafts/***"),
        ("/applications/abc?x=1", "/applications/***?x=1"),
        ("/employer/jobs/7", "/employer/jobs/7"),
        ("/drafts", "/drafts"),
    ],
)
def test_mask_tokens(raw: str, masked: str) -> None:
    """Only the segment after /drafts/ and /applications/ is replaced."""
    assert mask_tokens(raw) == masked


def test_each_candidate_sees_only_their_own_data(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """Two candidates: every preview and status page holds only its own contact data."""
    job_id = create_open_posting(candidate_app, candidate_embedder, "Backend Developer")
    people = {
        "ada": ("Ada Lovelace", "ada.only@example.com", "+90 532 111 22 33"),
        "bob": ("Bob Builder", "bob.only@example.com", "+90 533 444 55 66"),
    }
    tokens = {}
    for key, (name, email, phone) in people.items():
        pdf = build_person_pdf(name, email, phone, f"{key}marker")
        tokens[key] = token_of(upload_cv(candidate_client, job_id, pdf))
    for key, (name, email, phone) in people.items():
        other = "bob" if key == "ada" else "ada"
        other_name, other_email, other_phone = people[other]
        preview = candidate_client.get(f"/drafts/{tokens[key]}").text
        assert email in preview and name in preview
        for foreign in (other_name, other_email, other_phone, f"{other}marker"):
            assert foreign not in preview
    for key in people:
        page_types = selected_types_of(candidate_client.get(f"/drafts/{tokens[key]}").text)
        page_texts = section_texts_of(candidate_client.get(f"/drafts/{tokens[key]}").text)
        candidate_client.post(
            f"/drafts/{tokens[key]}", data=confirm_form(list(zip(page_types, page_texts, strict=True)))
        )
    for key, (_, email, phone) in people.items():
        other_email = people["bob" if key == "ada" else "ada"][1]
        status = candidate_client.get(f"/applications/{tokens[key]}").text
        assert other_email not in status and email not in status and phone not in status
