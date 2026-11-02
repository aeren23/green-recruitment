"""The SPEC 3 demo from start to finish (T-16 acceptance).

Fake embedder and fake LLM; no GPU, no Ollama. One logged-in client plays both
the candidate and the employer, which is safe because the candidate pages do
not look at the employer cookie.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from candidate_support import (
    create_open_posting,
    input_value_of,
    posting_status,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from green_support import build_bench_db, use_bench_db
from pipeline_support import ControlledEmbedder
from review_support import (
    CONFLICT,
    REDIRECT,
    build_person_cv,
    confirm_unchanged,
    person_number,
    rank_candidate,
)
from web_support import NOW, log_in

from app.green.counters import (
    count_confirmed_applications,
    derive_llm_calls_avoided,
    read_counters,
)
from app.web import messages
from app.web.dependencies import get_now
from app.web.paths import GREEN_PATH

REQUIREMENTS = ("Building REST APIs", "Relational databases and SQL")
MIN_YEARS = 2.0
CONFIRMED_APPLICATIONS = 3
GOOD_QUALITY = 90
OTHER_QUALITY = 60
EXAMPLE_COMMENT_PATTERN = re.compile(r"Aday #(\d+)")


@pytest.fixture
def demo_client(
    candidate_app: FastAPI, candidate_client: TestClient, tmp_path: Path
) -> Iterator[TestClient]:
    """A logged-in client with a fixed clock and a synthetic `bench.db`."""
    candidate_app.dependency_overrides[get_now] = lambda: NOW
    use_bench_db(candidate_app, build_bench_db(tmp_path / "bench"))
    log_in(candidate_client)
    yield candidate_client
    candidate_app.dependency_overrides.clear()


def test_the_whole_demo_runs_from_start_to_finish(
    candidate_app: FastAPI,
    demo_client: TestClient,
    candidate_embedder: ControlledEmbedder,
) -> None:
    """Steps 1 to 7 of SPEC 3, each checked on the page the user would see."""
    backend = create_open_posting(
        candidate_app, candidate_embedder, "Backend Developer", min_years=MIN_YEARS
    )
    accountant = create_open_posting(candidate_app, candidate_embedder, "Accountant")
    person = person_number(1)
    pdf = build_person_cv(person, "demo", GOOD_QUALITY)

    # 1. The employer sees the posting with its filters and requirements.
    posting_page = demo_client.get(f"/employer/jobs/{backend}")
    assert posting_page.status_code == 200
    assert messages.FILTER_MIN_YEARS in posting_page.text
    assert all(requirement in posting_page.text for requirement in REQUIREMENTS)

    # 2. The candidate uploads, previews (with contact), corrects and confirms.
    token = token_of(upload_cv(demo_client, backend, pdf))
    preview = demo_client.get(f"/drafts/{token}").text
    assert "Dosyanız saklanmadı" in preview
    assert input_value_of(preview, "contact_email") == person.email
    confirm_unchanged(demo_client, token)

    # 3. The same CV to the other posting comes from the cache.
    second = upload_cv(demo_client, accountant, pdf)
    assert second.status_code == REDIRECT
    cached_preview = demo_client.get(second.headers["location"]).text
    assert "Önceki başvurunuzdan alındı, yeniden işlenmedi" in cached_preview
    confirm_unchanged(demo_client, token_of(second))

    # 4. The same CV to the same posting again is blocked.
    blocked = upload_cv(demo_client, backend, pdf)
    assert blocked.status_code == CONFLICT
    assert messages.DUPLICATE_APPLICATION in blocked.text

    # 5. The employer sees an anonymous list and matrix; shortlisting reveals contact.
    other = build_person_cv(person_number(2), "other", OTHER_QUALITY)
    rank_candidate(demo_client, candidate_app, backend, other)
    ranked_list = demo_client.get(f"/employer/jobs/{backend}").text
    assert person.email not in ranked_list and person.name not in ranked_list
    application_id = int(EXAMPLE_COMMENT_PATTERN.findall(ranked_list)[0])
    matrix = demo_client.get(f"/employer/applications/{application_id}").text
    assert REQUIREMENTS[0] in matrix and messages.MATCH_EVIDENCE in matrix
    assert person.email not in matrix
    shortlisted = demo_client.post(f"/employer/applications/{application_id}/shortlist")
    assert shortlisted.status_code == REDIRECT
    contact = demo_client.get(f"/employer/applications/{application_id}").text
    assert person.email in contact

    with candidate_app.state.session_factory() as session:
        confirmed = count_confirmed_applications(session)
        calls = read_counters(session).llm_calls
        assert confirmed == CONFIRMED_APPLICATIONS
        assert derive_llm_calls_avoided(session) == max(confirmed - calls, 0)

    # 6. Closing the posting runs retention and the panel counters update.
    with candidate_app.state.session_factory() as session:
        deleted_before = read_counters(session).records_deleted
    closed = demo_client.post(f"/employer/jobs/{backend}/close")
    assert closed.status_code == REDIRECT
    assert posting_status(candidate_app, backend) == "closed"
    with candidate_app.state.session_factory() as session:
        assert read_counters(session).records_deleted > deleted_before

    # 7. The green panel shows counters, the comparison and the page weight.
    panel = demo_client.get(GREEN_PATH).text
    assert "Ayrıştırma önbelleği isabeti" in panel
    assert "Önlenen LLM çağrısı" in panel
    assert "Net enerji / başvuru (Wh)" in panel
    assert "K1 durumu: <strong>Geçti</strong>" in panel
    assert "/employer/jobs/{job_id}" in panel
    assert f"/employer/jobs/{backend}" not in panel
    assert token not in panel
