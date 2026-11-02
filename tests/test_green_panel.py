"""The green panel (T-16): live counters, read-only comparison, retention button."""

import gzip
import re
import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from candidate_support import (
    create_open_posting,
    stored_applications,
    stored_decisions,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from green_support import build_bench_db, use_bench_db
from pipeline_support import ControlledEmbedder, build_cv_pdf
from web_support import NOW, log_in

from app.green.benchmark_view import load_latest_comparison
from app.green.counters import read_counters
from app.models import DecisionRecord
from app.web import messages
from app.web.dependencies import get_now
from app.web.paths import EMPLOYER_LOGIN_PATH, GREEN_PATH, GREEN_RETENTION_PATH
from bench.store import read_only_engine

PAGE_BUDGET_BYTES = 50 * 1024
REDIRECT = 303
UNPROCESSABLE = 422
FAST_FORWARD_DAYS = 31
DECISION_RETENTION_DAYS = 30
NO_MEASUREMENT_NOTE = "Henüz ölçüm yok"


@pytest.fixture
def panel_app(candidate_app: FastAPI, tmp_path: Path) -> Iterator[FastAPI]:
    """The application with the clock fixed and an empty bench directory."""
    candidate_app.dependency_overrides[get_now] = lambda: NOW
    use_bench_db(candidate_app, tmp_path / "bench" / "bench.db")
    yield candidate_app
    candidate_app.dependency_overrides.clear()


@pytest.fixture
def panel_client(panel_app: FastAPI, candidate_client: TestClient) -> TestClient:
    """A client that is not logged in."""
    return candidate_client


def add_expiring_decision(application: FastAPI, job_id: int) -> None:
    """A decision record that expires `DECISION_RETENTION_DAYS` after NOW."""
    with application.state.session_factory() as session:
        session.add(
            DecisionRecord(
                job_id=job_id,
                raw_hash="ab" * 32,
                reason_code="NOT_SELECTED",
                decided_at=NOW,
                expires_at=NOW + timedelta(days=DECISION_RETENTION_DAYS),
                application_token=None,
            )
        )
        session.commit()


def test_panel_without_bench_db_shows_the_no_measurement_note(
    panel_client: TestClient, tmp_path: Path
) -> None:
    """A missing file gives the note, and the file is not created."""
    response = panel_client.get(GREEN_PATH)
    assert response.status_code == 200
    assert NO_MEASUREMENT_NOTE in response.text
    assert "Canlı sayaçlar" in response.text
    assert "Saklanmayan PDF boyutu" in response.text
    assert not (tmp_path / "bench").exists()


def test_empty_bench_db_file_shows_the_note(
    panel_app: FastAPI, panel_client: TestClient, tmp_path: Path
) -> None:
    """A file without tables is treated like no measurement."""
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    use_bench_db(panel_app, empty)
    assert NO_MEASUREMENT_NOTE in panel_client.get(GREEN_PATH).text


def test_panel_shows_the_comparison_table_and_k1_status(
    panel_app: FastAPI, panel_client: TestClient, tmp_path: Path
) -> None:
    """Metric, baseline, green and difference come from the report code."""
    use_bench_db(panel_app, build_bench_db(tmp_path / "bench"))
    page = panel_client.get(GREEN_PATH).text
    assert NO_MEASUREMENT_NOTE not in page
    assert "Net enerji / başvuru (Wh)" in page
    assert "-50.0%" in page
    assert "K1 durumu: <strong>Geçti</strong>" in page
    assert messages.K1_NOT_FINAL_WARNING not in page


def test_panel_warns_when_conditions_are_not_final(
    panel_app: FastAPI, panel_client: TestClient, tmp_path: Path
) -> None:
    """Runs measured outside the final conditions carry the warning."""
    use_bench_db(panel_app, build_bench_db(tmp_path / "bench", final_conditions=False))
    assert messages.K1_NOT_FINAL_WARNING in panel_client.get(GREEN_PATH).text


def test_bench_db_is_opened_read_only(tmp_path: Path) -> None:
    """Writing through the panel's connection fails and the file stays unchanged."""
    bench_db = build_bench_db(tmp_path / "bench")
    before = bench_db.read_bytes()
    assert load_latest_comparison(bench_db, 10) is not None
    engine = read_only_engine(bench_db)
    try:
        with engine.connect() as connection, pytest.raises(Exception, match="readonly"):
            connection.exec_driver_sql("DELETE FROM benchmark_runs")
            connection.commit()
    finally:
        engine.dispose()
    assert bench_db.read_bytes() == before


def test_panel_stays_within_the_page_budget_and_has_no_script(
    panel_app: FastAPI, panel_client: TestClient, tmp_path: Path
) -> None:
    """SPEC 9.2: HTML plus CSS under 50 KB gzipped; no inline JavaScript."""
    use_bench_db(panel_app, build_bench_db(tmp_path / "bench"))
    page = panel_client.get(GREEN_PATH)
    stylesheet = panel_client.get("/static/app.css")
    assert "<script" not in page.text.lower()
    assert len(gzip.compress(page.content + stylesheet.content)) < PAGE_BUDGET_BYTES


def test_page_weight_rows_show_templates(panel_client: TestClient) -> None:
    """Visiting pages makes their templates appear with an average."""
    panel_client.get("/")
    panel_client.get("/static/app.css")
    page = panel_client.get(GREEN_PATH).text
    assert "/static/{file}" in page
    assert re.search(r"<td>\d+,\d KB</td>", page)


def test_retention_requires_the_employer_cookie(panel_client: TestClient) -> None:
    """Without the cookie the request goes to the login and nothing runs."""
    response = panel_client.post(GREEN_RETENTION_PATH, data={"days": "31"})
    assert response.status_code == REDIRECT
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH


def test_fast_forward_runs_r1_and_r5_and_counts_deletions(
    panel_app: FastAPI,
    panel_client: TestClient,
    candidate_embedder: ControlledEmbedder,
) -> None:
    """+31 days expires an old draft (R1) and a decision record (R5); the notice has counts."""
    job_id = create_open_posting(panel_app, candidate_embedder, "Backend Developer")
    upload_cv(panel_client, job_id, build_cv_pdf("alpha"))
    add_expiring_decision(panel_app, job_id)
    log_in(panel_client)
    with panel_app.state.session_factory() as session:
        deleted_before = read_counters(session).records_deleted
    response = panel_client.post(GREEN_RETENTION_PATH, data={"days": str(FAST_FORWARD_DAYS)})
    assert response.status_code == REDIRECT
    page = panel_client.get(response.headers["location"]).text
    assert "Retention çalıştı (+31 gün)" in page
    assert "taslak (R1) 1" in page
    assert "karar kaydı (R5) 1" in page
    assert stored_applications(panel_app) == []
    assert stored_decisions(panel_app) == []
    with panel_app.state.session_factory() as session:
        assert read_counters(session).records_deleted > deleted_before


def test_without_fast_forward_a_fresh_draft_is_kept(
    panel_app: FastAPI,
    panel_client: TestClient,
    candidate_embedder: ControlledEmbedder,
) -> None:
    """+0 days deletes nothing that is not yet due."""
    job_id = create_open_posting(panel_app, candidate_embedder, "Backend Developer")
    upload_cv(panel_client, job_id, build_cv_pdf("alpha"))
    log_in(panel_client)
    response = panel_client.post(GREEN_RETENTION_PATH, data={"days": "0"})
    assert response.status_code == REDIRECT
    assert len(stored_applications(panel_app)) == 1


@pytest.mark.parametrize("days", ["-1", "366", "abc", "1.5"])
def test_days_outside_the_range_are_rejected_in_turkish(
    panel_app: FastAPI,
    panel_client: TestClient,
    candidate_embedder: ControlledEmbedder,
    days: str,
) -> None:
    """Anything but a whole number from 0 to 365 gives 422 and runs nothing."""
    job_id = create_open_posting(panel_app, candidate_embedder, "Backend Developer")
    upload_cv(panel_client, job_id, build_cv_pdf("alpha"))
    log_in(panel_client)
    response = panel_client.post(GREEN_RETENTION_PATH, data={"days": days})
    assert response.status_code == UNPROCESSABLE
    assert "0 ile 365 arasında" in response.text
    assert len(stored_applications(panel_app)) == 1


def test_forged_notice_values_cannot_inject_text(panel_client: TestClient) -> None:
    """The notice only renders integers; anything else shows no notice."""
    query = {"shifted": "<b>x</b>", "drafts": "1", "shortlists": "1", "decisions": "1"}
    page = panel_client.get(GREEN_PATH, params=query).text
    assert "<b>x</b>" not in page
    assert "Retention çalıştı" not in page
