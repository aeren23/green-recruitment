"""T-15: ranked list, matrix, shortlisting and the KVKK contact rule (SPEC 9.1).

Fake embedder (controlled similarities) and fake LLM; no GPU, no Ollama.
"""

import re
from collections.abc import Iterator

import pytest
from candidate_support import create_open_posting, stored_applications, token_of, upload_cv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import ControlledEmbedder, FakeLlm
from review_support import (
    CONFLICT,
    INCOMPLETE_PROFILE,
    NOT_FOUND,
    REDIRECT,
    build_person_cv,
    person_number,
    rank_candidate,
    stored_contact_values,
)
from sqlalchemy import update
from test_web_green_ui import FORBIDDEN_ASSET_PATTERN, PAGE_BUDGET_BYTES, gzip_size
from web_support import NOW, log_in

from app.models import Application, ApplicationStatus, JobPosting, JobStatus
from app.pipeline.ranking import ScoreBounds
from app.web.dependencies import get_now
from app.web.paths import (
    EMPLOYER_HOME_PATH,
    EMPLOYER_LOGIN_PATH,
    EMPLOYER_NEW_JOB_PATH,
    STYLESHEET_URL,
)

ROW_PATTERN = re.compile(r"Aday #(\d+)(?: <span[^>]*>\(eksik bilgi\)</span>)?</td>\s*<td>(\d+)</td>")
LABEL_PATTERN = re.compile(r"Aday #(\d+)")
PAGE_SIZE = 20


@pytest.fixture
def employer(candidate_app: FastAPI, candidate_client: TestClient) -> Iterator[TestClient]:
    """A logged-in client with a fixed clock."""
    candidate_app.dependency_overrides[get_now] = lambda: NOW
    log_in(candidate_client)
    yield candidate_client
    candidate_app.dependency_overrides.clear()


@pytest.fixture
def job_id(candidate_app: FastAPI, candidate_embedder: ControlledEmbedder) -> int:
    """An open posting without hard filters."""
    return create_open_posting(candidate_app, candidate_embedder, "Backend Developer")


def add_candidates(
    client: TestClient, application: FastAPI, job: int, qualities: list[int]
) -> list[int]:
    """Rank one candidate per quality; return the application ids in order."""
    ids = []
    for index, quality in enumerate(qualities, start=1):
        pdf = build_person_cv(person_number(index), f"cand{index}", quality)
        ids.append(rank_candidate(client, application, job, pdf))
    return ids


def rows_of(page_text: str) -> list[tuple[int, int]]:
    """(application id, relative score) of each row, in page order."""
    return [(int(found), int(score)) for found, score in ROW_PATTERN.findall(page_text)]


def job_path(job: int) -> str:
    """The employer posting page."""
    return f"{EMPLOYER_HOME_PATH}/jobs/{job}"


def application_path(application_id: int) -> str:
    """The employer application page."""
    return f"{EMPLOYER_HOME_PATH}/applications/{application_id}"


def test_relative_score_is_min_max_over_the_whole_posting(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Lowest 0, highest 100, the middle one in between; rows ordered by raw score."""
    first, second, third = add_candidates(employer, candidate_app, job_id, [50, 90, 70])

    page = employer.get(job_path(job_id))

    assert page.status_code == 200
    assert rows_of(page.text) == [(second, 100), (third, 50), (first, 0)]
    assert "göreli" in page.text.lower()


def test_relative_score_uses_all_candidates_not_just_the_page(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """The lowest scorer on page 2 shows 0 although page 1 does not contain it."""
    qualities = [60 + (number % 5) * 5 for number in range(PAGE_SIZE + 1)]
    qualities[-1] = 30
    ids = add_candidates(employer, candidate_app, job_id, qualities)

    second_page = employer.get(f"{job_path(job_id)}?page=2")

    assert rows_of(second_page.text) == [(ids[-1], 0)]


def test_equal_scores_and_a_single_candidate_show_100(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """No division by zero when the score range is empty."""
    (only,) = add_candidates(employer, candidate_app, job_id, [60])
    assert rows_of(employer.get(job_path(job_id)).text) == [(only, 100)]

    pdf = build_person_cv(person_number(2), "second", 60)
    second = rank_candidate(employer, candidate_app, job_id, pdf)

    assert rows_of(employer.get(job_path(job_id)).text) == [(only, 100), (second, 100)]


def test_score_bounds_pure_scaling() -> None:
    """Edge cases of the scaling itself."""
    bounds = ScoreBounds(lowest=0.5, highest=0.9)
    assert bounds.relative(0.5) == 0
    assert bounds.relative(0.9) == 100
    assert bounds.relative(0.7) == pytest.approx(50)
    assert ScoreBounds(lowest=0.7, highest=0.7).relative(0.7) == 100


def test_twenty_one_candidates_make_two_stable_pages(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """20 per page; order is raw score descending, then id, on every request."""
    qualities = [60 + (number % 4) * 10 for number in range(PAGE_SIZE + 1)]
    add_candidates(employer, candidate_app, job_id, qualities)
    stored = stored_applications(candidate_app)
    expected = [row.id for row in sorted(stored, key=lambda row: (-row.score, row.id))]

    first_page = employer.get(job_path(job_id))
    second_page = employer.get(f"{job_path(job_id)}?page=2")
    repeated = employer.get(job_path(job_id))

    listed = [int(found) for found in LABEL_PATTERN.findall(first_page.text)]
    assert listed + [int(found) for found in LABEL_PATTERN.findall(second_page.text)] == expected
    assert len(listed) == PAGE_SIZE
    assert "Sayfa 1 / 2" in first_page.text
    assert repeated.text == first_page.text


def test_missing_information_marker(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Only the candidate with unknown years, languages and location is marked."""
    complete = rank_candidate(employer, candidate_app, job_id, build_person_cv(person_number(1), "a", 60))
    pdf = build_person_cv(person_number(2), "b", 70)
    incomplete = rank_candidate(employer, candidate_app, job_id, pdf, **INCOMPLETE_PROFILE)

    page = employer.get(job_path(job_id)).text

    assert page.count("eksik bilgi") == 1
    assert re.search(rf"Aday #{incomplete} <span[^>]*>\(eksik bilgi\)", page)
    assert not re.search(rf"Aday #{complete} <span", page)


def test_evidence_and_claim_markers_and_matched_text(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Experience is evidence, skills are a claim; the matched section text is shown."""
    evidence = rank_candidate(employer, candidate_app, job_id, build_person_cv(person_number(1), "a", 60))
    claim_pdf = build_person_cv(person_number(2), "b", 60, in_skills=True)
    claim = rank_candidate(employer, candidate_app, job_id, claim_pdf)

    evidence_page = employer.get(application_path(evidence)).text
    claim_page = employer.get(application_path(claim)).text

    assert "<td>Kanıt</td>" in evidence_page and "<td>Beyan</td>" not in evidence_page
    assert "<td>Beyan</td>" in claim_page and "<td>Kanıt</td>" not in claim_page
    assert "EXPERIENCETEXTMARKER" in evidence_page
    assert "Python SQL Docker" in claim_page
    assert "Building REST APIs" in evidence_page and "<td>0.6</td>" in evidence_page


def test_detail_shows_the_summary_or_a_note_and_lists_the_candidate_header(
    employer: TestClient, candidate_app: FastAPI, job_id: int, fake_llm: FakeLlm
) -> None:
    """The background summary is shown; without one a Turkish note appears."""
    (ranked,) = add_candidates(employer, candidate_app, job_id, [60])
    page = employer.get(application_path(ranked)).text
    assert "Aday gereksinimleri karsiliyor." in page

    with candidate_app.state.session_factory() as session:
        session.execute(update(Application).values(summary=None))
        session.commit()
    assert "Özet henüz yok" in employer.get(application_path(ranked)).text


def test_rendering_never_calls_the_llm(
    employer: TestClient, candidate_app: FastAPI, job_id: int, fake_llm: FakeLlm
) -> None:
    """Viewing the list and the detail, even without a summary, makes no LLM call."""
    (ranked,) = add_candidates(employer, candidate_app, job_id, [60])
    with candidate_app.state.session_factory() as session:
        session.execute(update(Application).values(summary=None))
        session.commit()
    attempts_before = fake_llm.attempts

    employer.get(job_path(job_id))
    employer.get(application_path(ranked))
    employer.get(EMPLOYER_HOME_PATH)

    assert fake_llm.attempts == attempts_before


def test_contact_is_absent_until_shortlisted_and_present_afterwards(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """T-15 acceptance: name, e-mail and phone appear in no employer response before shortlisting."""
    pdf = build_person_cv(person_number(7), "kvkk", 80)
    ranked = rank_candidate(employer, candidate_app, job_id, pdf)
    other = rank_candidate(
        employer, candidate_app, job_id, build_person_cv(person_number(8), "other", 60)
    )
    secrets = stored_contact_values(candidate_app, ranked)
    assert all(secrets) and len(secrets) == 3
    markers = ["Kaplanoglu", "kaplanoglu", "987 65"]
    every_page = {
        "list": employer.get(job_path(job_id)).text,
        "detail": employer.get(application_path(ranked)).text,
        "postings": employer.get(EMPLOYER_HOME_PATH).text,
        "new form": employer.get(EMPLOYER_NEW_JOB_PATH).text,
        "unknown application": employer.get(application_path(9999)).text,
        "unknown posting": employer.get(job_path(9999)).text,
    }
    for name, text in every_page.items():
        for secret in (*secrets, *markers):
            assert secret not in text, f"{secret} leaked on {name}"

    response = employer.post(f"{application_path(ranked)}/shortlist")
    assert response.status_code == REDIRECT
    assert response.headers["location"] == application_path(ranked)

    detail = employer.get(application_path(ranked)).text
    for secret in secrets:
        assert secret in detail
    listing = employer.get(job_path(job_id)).text
    assert all(secret not in listing for secret in (*secrets, *markers))
    assert "Kısa listede" in listing
    assert "Kısa listeye al" not in detail
    other_detail = employer.get(application_path(other)).text
    assert all(secret not in other_detail for secret in secrets)


def test_shortlist_sets_status_and_decision_time(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Ranked becomes shortlisted with `decided_at` from the injected clock."""
    (ranked,) = add_candidates(employer, candidate_app, job_id, [60])
    assert "Kısa listeye al" in employer.get(application_path(ranked)).text

    employer.post(f"{application_path(ranked)}/shortlist")

    (stored,) = stored_applications(candidate_app)
    assert stored.status is ApplicationStatus.SHORTLISTED
    assert stored.decided_at == NOW


def test_shortlisting_twice_is_refused_with_409(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Only a ranked application can be shortlisted."""
    (ranked,) = add_candidates(employer, candidate_app, job_id, [60])
    employer.post(f"{application_path(ranked)}/shortlist")

    again = employer.post(f"{application_path(ranked)}/shortlist")

    assert again.status_code == CONFLICT
    assert "yalnızca sıralanmış" in again.text


@pytest.mark.parametrize("status", [JobStatus.CLOSED, JobStatus.FINALIZED])
def test_application_of_a_closed_posting_cannot_be_shortlisted(
    employer: TestClient, candidate_app: FastAPI, job_id: int, status: JobStatus
) -> None:
    """The posting must be open; the application stays ranked."""
    (ranked,) = add_candidates(employer, candidate_app, job_id, [60])
    with candidate_app.state.session_factory() as session:
        session.execute(update(JobPosting).values(status=status))
        session.commit()

    response = employer.post(f"{application_path(ranked)}/shortlist")

    assert response.status_code == CONFLICT
    assert "ilan artık açık değil" in response.text
    (stored,) = stored_applications(candidate_app)
    assert stored.status is ApplicationStatus.RANKED


def test_unknown_ids_and_drafts_give_404(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """Unknown posting or application, and a draft that is not ranked yet."""
    token = token_of(upload_cv(employer, job_id, build_person_cv(person_number(1), "d", 60)))
    (draft,) = stored_applications(candidate_app)
    assert token and draft.status is ApplicationStatus.DRAFT

    assert employer.get(job_path(9999)).status_code == NOT_FOUND
    assert employer.get(application_path(9999)).status_code == NOT_FOUND
    assert employer.get(application_path(draft.id)).status_code == NOT_FOUND
    assert employer.post(f"{application_path(9999)}/shortlist").status_code == NOT_FOUND


def test_all_three_routes_require_the_employer_cookie(
    candidate_app: FastAPI, candidate_client: TestClient, job_id: int
) -> None:
    """Without login each route redirects to the login form and changes nothing."""
    log_in(candidate_client)
    (ranked,) = add_candidates(candidate_client, candidate_app, job_id, [60])
    candidate_client.cookies.clear()

    responses = [
        candidate_client.get(job_path(job_id)),
        candidate_client.get(application_path(ranked)),
        candidate_client.post(f"{application_path(ranked)}/shortlist"),
    ]

    for response in responses:
        assert response.status_code == REDIRECT
        assert response.headers["location"] == EMPLOYER_LOGIN_PATH
    (stored,) = stored_applications(candidate_app)
    assert stored.status is ApplicationStatus.RANKED


def test_pages_respect_the_green_budget_and_escape_cv_text(
    employer: TestClient, candidate_app: FastAPI, job_id: int
) -> None:
    """A full page and a detail page stay under 50 KB gzipped, with no script and escaped text."""
    hostile = "<script>alert(1)</script> <b>bold</b> & more"
    qualities = [50 + number for number in range(PAGE_SIZE)]
    ids = add_candidates(employer, candidate_app, job_id, qualities[:-1])
    pdf = build_person_cv(person_number(99), "hostile", 90, experience_line=hostile)
    ids.append(rank_candidate(employer, candidate_app, job_id, pdf))
    css = employer.get(STYLESHEET_URL).content

    list_page = employer.get(job_path(job_id))
    detail_page = employer.get(application_path(ids[-1]))

    assert len(LABEL_PATTERN.findall(list_page.text)) == PAGE_SIZE
    for page in (list_page, detail_page):
        assert gzip_size(page.content) + gzip_size(css) < PAGE_BUDGET_BYTES
        assert "<script" not in page.text.lower()
        assert FORBIDDEN_ASSET_PATTERN.search(page.text) is None
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in detail_page.text
    assert "<b>bold</b>" not in detail_page.text
