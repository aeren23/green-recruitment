"""Pagination (Green rule 9): 20 rows per page, previous and next links."""

import pytest
from conftest import FakeEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from web_support import add_postings, log_in

from app.models import JobPosting
from app.web.pagination import PAGE_SIZE, paginate, parse_page_number
from app.web.paths import EMPLOYER_HOME_PATH

TOTAL_POSTINGS = 45
LAST_PAGE = 3
LAST_PAGE_ROWS = TOTAL_POSTINGS - 2 * PAGE_SIZE
PREVIOUS_LABEL = "Önceki"
NEXT_LABEL = "Sonraki"


@pytest.fixture
def many_postings(
    web_app: FastAPI, web_client: TestClient, fake_embedder: FakeEmbedder
) -> TestClient:
    """A logged-in client over 45 postings (Posting 01 oldest, Posting 45 newest)."""
    add_postings(web_app, fake_embedder, TOTAL_POSTINGS)
    log_in(web_client)
    return web_client


def titles_on(page_html: str) -> list[str]:
    """The `Posting NN` titles in the order they appear on the page."""
    return [
        f"Posting {number:02d}"
        for number in range(TOTAL_POSTINGS, 0, -1)
        if f"Posting {number:02d}" in page_html
    ]


def test_first_page_holds_the_newest_twenty_and_only_a_next_link(
    many_postings: TestClient,
) -> None:
    """Newest first; no previous link on page 1."""
    html = many_postings.get(EMPLOYER_HOME_PATH).text

    assert titles_on(html) == [f"Posting {n:02d}" for n in range(45, 25, -1)]
    assert NEXT_LABEL in html
    assert PREVIOUS_LABEL not in html
    assert "Sayfa 1 / 3" in html
    assert "page=2" in html


def test_middle_page_has_both_links(many_postings: TestClient) -> None:
    """Page 2: rows 25 down to 6, previous and next."""
    html = many_postings.get(EMPLOYER_HOME_PATH, params={"page": 2}).text

    assert titles_on(html) == [f"Posting {n:02d}" for n in range(25, 5, -1)]
    assert PREVIOUS_LABEL in html and NEXT_LABEL in html
    assert "page=1" in html and "page=3" in html


def test_last_page_has_the_remainder_and_only_a_previous_link(
    many_postings: TestClient,
) -> None:
    """Page 3 holds the 5 oldest rows."""
    html = many_postings.get(EMPLOYER_HOME_PATH, params={"page": LAST_PAGE}).text

    assert len(titles_on(html)) == LAST_PAGE_ROWS
    assert PREVIOUS_LABEL in html
    assert NEXT_LABEL not in html


@pytest.mark.parametrize(
    ("raw_page", "expected_marker"),
    [("0", "Sayfa 1 / 3"), ("-4", "Sayfa 1 / 3"), ("abc", "Sayfa 1 / 3"), ("99", "Sayfa 3 / 3")],
)
def test_out_of_range_or_garbage_page_numbers_are_clamped(
    many_postings: TestClient, raw_page: str, expected_marker: str
) -> None:
    """Never an error page; the nearest real page is shown."""
    response = many_postings.get(EMPLOYER_HOME_PATH, params={"page": raw_page})
    assert response.status_code == 200
    assert expected_marker in response.text


def test_a_single_page_shows_no_pager(
    web_app: FastAPI, web_client: TestClient, fake_embedder: FakeEmbedder
) -> None:
    """Twenty rows fit on one page: no navigation."""
    add_postings(web_app, fake_embedder, PAGE_SIZE)
    log_in(web_client)
    html = web_client.get(EMPLOYER_HOME_PATH).text
    assert len(titles_on(html)) == PAGE_SIZE
    assert "Sayfalama" not in html


def test_paginate_helper_reports_totals_and_neighbours(
    demo_session: Session, fake_embedder: FakeEmbedder
) -> None:
    """The reusable helper, without the web layer."""
    from app.postings import PostingInput, RequirementInput, create_postings
    from web_support import NOW

    inputs = [
        PostingInput(f"P{n}", "", [RequirementInput("SQL", 1)], NOW) for n in range(41)
    ]
    create_postings(demo_session, fake_embedder, inputs)
    statement = select(JobPosting).order_by(JobPosting.id)

    page = paginate(demo_session, statement, 3)

    assert (page.total, page.page_count, page.number) == (41, 3, 3)
    assert len(page.items) == 1
    assert page.has_previous and not page.has_next
    assert (page.previous_number, page.next_number) == (2, 3)
    assert len(paginate(demo_session, statement, 1).items) == PAGE_SIZE
    assert paginate(demo_session, statement.where(JobPosting.id < 0), 1).page_count == 1


@pytest.mark.parametrize(
    ("raw_value", "expected"), [(None, 1), ("", 1), ("7", 7), ("0", 1), ("x", 1), ("-2", 1)]
)
def test_parse_page_number_is_lenient(raw_value: str | None, expected: int) -> None:
    """Only a positive integer string selects a page."""
    assert parse_page_number(raw_value) == expected


def test_page_size_is_twenty() -> None:
    """Green rule 9."""
    assert PAGE_SIZE == 20
