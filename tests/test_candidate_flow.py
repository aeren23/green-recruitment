"""SPEC 3 steps 2 to 4 end to end through the candidate pages (T-13).

Fake embedder and fake LLM; no GPU, no Ollama. The summaries run in a
background task, which the test client completes before it returns.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from candidate_support import (
    confirm_form,
    create_open_posting,
    input_value_of,
    posting_status,
    section_texts_of,
    selected_types_of,
    stored_applications,
    stored_decisions,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import (
    CANDIDATE_EMAIL,
    CANDIDATE_NAME,
    CANDIDATE_PHONE,
    ControlledEmbedder,
    build_cv_pdf,
)
from sqlalchemy import select

from app.green.counters import read_counters
from app.models import CandidateProfile, JobPosting, Section
from app.pipeline import upload as upload_module
from app.privacy.reasons import StoredReasonCode
from app.privacy.retention import RetentionContext, close_posting
from app.web import messages
from app.web.dependencies import get_now

EDITED_EMAIL = "edited.person@example.org"
DECISION_RETENTION_DAYS = 30
SECTIONS_IN_CV = 4
DRAFT_TTL_HOURS = 24
BEYOND_TTL = timedelta(hours=DRAFT_TTL_HOURS + 1)
BEYOND_DECISION_RETENTION = timedelta(days=DECISION_RETENTION_DAYS + 1)
GONE = 410
CONFLICT = 409
UNPROCESSABLE = 422


@pytest.fixture
def backend_job(candidate_app: FastAPI, candidate_embedder: ControlledEmbedder) -> int:
    """An open posting without hard filters."""
    return create_open_posting(candidate_app, candidate_embedder, "Backend Developer")


@pytest.fixture
def accountant_job(candidate_app: FastAPI, candidate_embedder: ControlledEmbedder) -> int:
    """A second open posting without hard filters."""
    return create_open_posting(candidate_app, candidate_embedder, "Accountant")


def preview_form(client: TestClient, token: str, **fields: str) -> dict[str, object]:
    """Form data that re-posts the preview's sections unchanged, with field overrides."""
    page = client.get(f"/drafts/{token}").text
    types = selected_types_of(page)
    texts = section_texts_of(page)
    kept = [(row_type, text) for row_type, text in zip(types, texts, strict=True) if text]
    return confirm_form(kept, **fields)


def confirmed_application(client: TestClient, job_id: int, pdf: bytes, **fields: str) -> str:
    """Upload a CV, confirm its preview unchanged and return the token."""
    token = token_of(upload_cv(client, job_id, pdf))
    response = client.post(f"/drafts/{token}", data=preview_form(client, token, **fields))
    assert response.status_code == 303
    return token


def test_step_2_upload_preview_edit_confirm(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Upload, see the messages and fields, edit, confirm: the status says under review."""
    response = upload_cv(candidate_client, backend_job, build_cv_pdf("alpha"))
    assert response.status_code == 303
    token = token_of(response)
    preview = candidate_client.get(f"/drafts/{token}")
    assert preview.status_code == 200
    assert preview.headers["cache-control"] == "no-store"
    assert "Dosyanız saklanmadı" in preview.text
    assert "Önceki başvurunuzdan" not in preview.text
    assert input_value_of(preview.text, "contact_email") == CANDIDATE_EMAIL
    assert input_value_of(preview.text, "contact_name") == CANDIDATE_NAME
    assert input_value_of(preview.text, "contact_phone") == CANDIDATE_PHONE
    assert "Hakkınızda ne saklanacak" in preview.text
    assert f"{DRAFT_TTL_HOURS} saat" in preview.text
    assert len([text for text in section_texts_of(preview.text) if text]) == SECTIONS_IN_CV
    assert len(selected_types_of(preview.text)) == SECTIONS_IN_CV + 2


def test_edits_are_stored_scrubbed_and_blank_location_is_none(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """An e-mail typed into a section never reaches sections or profile_json."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    page = candidate_client.get(f"/drafts/{token}").text
    types = selected_types_of(page)[:SECTIONS_IN_CV]
    texts = section_texts_of(page)[:SECTIONS_IN_CV]
    texts[0] += f" Reach me at {EDITED_EMAIL} or {CANDIDATE_EMAIL}."
    form = confirm_form(
        list(zip(types, texts, strict=True)),
        years_experience="8,5",
        languages="İngilizce, Türkçe",
        location="  ",
        contact_name=CANDIDATE_NAME,
        contact_email=CANDIDATE_EMAIL,
        contact_phone=CANDIDATE_PHONE,
    )
    response = candidate_client.post(f"/drafts/{token}", data=form)
    assert response.status_code == 303
    assert response.headers["location"] == f"/applications/{token}"
    assert "Değerlendirmede" in candidate_client.get(response.headers["location"]).text
    with candidate_app.state.session_factory() as session:
        profile = session.scalars(select(CandidateProfile)).one()
        stored_text = " ".join(item.text for item in session.scalars(select(Section)))
        assert EDITED_EMAIL not in stored_text and CANDIDATE_EMAIL not in stored_text
        assert EDITED_EMAIL not in json.dumps(profile.profile_json)
        assert CANDIDATE_EMAIL not in json.dumps(profile.profile_json)
        assert profile.location is None
        assert profile.languages == ["english", "turkish"]
        assert profile.years_experience == 8.5
        assert profile.contact.email == CANDIDATE_EMAIL
        assert read_counters(session).pdf_bytes_not_stored > 0


def test_emptied_section_is_removed_and_a_blank_row_adds_one(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Emptying a text deletes the section; filling a blank row creates one."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    page = candidate_client.get(f"/drafts/{token}").text
    types = selected_types_of(page)
    texts = section_texts_of(page)
    texts[SECTIONS_IN_CV - 1] = ""
    texts[SECTIONS_IN_CV] = "Certified cloud practitioner QSCORE60"
    types[SECTIONS_IN_CV] = "certification"
    candidate_client.post(f"/drafts/{token}", data=confirm_form(list(zip(types, texts, strict=True))))
    with candidate_app.state.session_factory() as session:
        stored = [(item.type.value, item.text) for item in session.scalars(select(Section))]
    assert len(stored) == SECTIONS_IN_CV
    assert ("certification", "Certified cloud practitioner QSCORE60") in stored


def test_step_3_same_cv_to_a_second_posting_is_taken_from_the_cache(
    candidate_client: TestClient,
    candidate_app: FastAPI,
    backend_job: int,
    accountant_job: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache message shows, parsing is skipped and the preview is pre-filled."""
    pdf = build_cv_pdf("alpha")
    first = confirmed_application(candidate_client, backend_job, pdf, years_experience="7")

    def parsing_must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("the parse cache was bypassed")

    monkeypatch.setattr(upload_module, "parse_into_new_profile", parsing_must_not_run)
    response = upload_cv(candidate_client, accountant_job, pdf)
    assert response.status_code == 303
    assert "cached=1" in response.headers["location"]
    second = token_of(response)
    preview = candidate_client.get(response.headers["location"])
    assert "Önceki başvurunuzdan alındı, yeniden işlenmedi" in preview.text
    assert "Dosyanız saklanmadı" in preview.text
    assert input_value_of(preview.text, "years_experience") == "7"
    assert first != second
    with candidate_app.state.session_factory() as session:
        assert read_counters(session).parse_cache_hits == 1


def test_step_4_same_cv_to_the_same_posting_is_blocked(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """A confirmed application blocks a repeat; no new application is created."""
    pdf = build_cv_pdf("alpha")
    confirmed_application(candidate_client, backend_job, pdf)
    before = len(stored_applications(candidate_app))
    response = upload_cv(candidate_client, backend_job, pdf)
    assert response.status_code == CONFLICT
    assert messages.DUPLICATE_APPLICATION in response.text
    assert len(stored_applications(candidate_app)) == before


def test_a_live_draft_is_redirected_to_instead_of_duplicated(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """SC-07: uploading the same CV again while a draft lives returns that draft."""
    pdf = build_cv_pdf("alpha")
    first = upload_cv(candidate_client, backend_job, pdf)
    second = upload_cv(candidate_client, backend_job, pdf)
    assert second.status_code == 303
    assert token_of(second) == token_of(first)
    assert len(stored_applications(candidate_app)) == 1


def test_confirming_twice_goes_to_the_status_page(
    candidate_client: TestClient, backend_job: int
) -> None:
    """A second confirmation of the same draft is not an error."""
    token = confirmed_application(candidate_client, backend_job, build_cv_pdf("alpha"))
    again = candidate_client.post(f"/drafts/{token}", data=confirm_form([]))
    assert again.status_code == 303
    assert again.headers["location"] == f"/applications/{token}"
    assert candidate_client.get(f"/drafts/{token}").headers["location"] == f"/applications/{token}"


def test_invalid_years_show_a_turkish_error_and_keep_the_input(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """The form is shown again with the typed values; the draft stays a draft."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    form = preview_form(candidate_client, token, years_experience="çok")
    response = candidate_client.post(f"/drafts/{token}", data=form)
    assert response.status_code == UNPROCESSABLE
    assert "Deneyim yılı 0 ile 60 arasında" in response.text
    assert 'value="çok"' in response.text
    assert [item.status.value for item in stored_applications(candidate_app)] == ["draft"]


@pytest.mark.parametrize(
    ("filters", "fields", "reason"),
    [
        ({"required_languages": ["german"]}, {"languages": "İngilizce"}, StoredReasonCode.LANG_MISSING),
        (
            {"location": "Denizli", "remote_ok": False},
            {"location": "İzmir"},
            StoredReasonCode.LOCATION_MISMATCH,
        ),
        ({"min_years": 10.0}, {"years_experience": "2"}, StoredReasonCode.MIN_YEARS),
    ],
)
def test_elimination_status_shows_the_reason_and_the_objection_note(
    candidate_client: TestClient,
    candidate_app: FastAPI,
    candidate_embedder: ControlledEmbedder,
    filters: dict[str, object],
    fields: dict[str, str],
    reason: StoredReasonCode,
) -> None:
    """A hard-filter elimination keeps only a decision record and says why."""
    job_id = create_open_posting(candidate_app, candidate_embedder, "Strict", **filters)
    token = confirmed_application(candidate_client, job_id, build_cv_pdf("alpha"), **fields)
    page = candidate_client.get(f"/applications/{token}")
    assert page.status_code == 200
    assert "Elendiniz" in page.text
    assert messages.ELIMINATION_REASONS[reason].replace("'", "&#39;") in page.text
    assert "KVKK madde 11" in page.text
    assert [item.reason_code for item in stored_decisions(candidate_app)] == [reason.value]
    assert stored_applications(candidate_app) == []


def test_status_after_the_posting_is_closed_is_not_selected(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Closing through retention turns ranked applications into NOT_SELECTED."""
    token = confirmed_application(candidate_client, backend_job, build_cv_pdf("alpha"))
    with candidate_app.state.session_factory() as session:
        now = datetime.now(UTC)
        close_posting(RetentionContext(session, now, DECISION_RETENTION_DAYS), backend_job)
        session.commit()
    page = candidate_client.get(f"/applications/{token}")
    assert "Bu ilan için seçilmediniz" in page.text
    assert posting_status(candidate_app, backend_job) == "closed"


def test_unknown_token_and_expired_decision_show_the_deleted_message(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """Neither an application nor a live decision: the generic deleted message."""
    unknown = candidate_client.get("/applications/no-such-token")
    assert unknown.status_code == 200
    assert "kaydı saklama süresi dolduğu için silindi" in unknown.text
    token = confirmed_application(
        candidate_client, backend_job, build_cv_pdf("alpha"), years_experience="1"
    )
    with candidate_app.state.session_factory() as session:
        close_posting(RetentionContext(session, datetime.now(UTC), DECISION_RETENTION_DAYS), backend_job)
        session.commit()
    candidate_app.dependency_overrides[get_now] = lambda: datetime.now(UTC) + BEYOND_DECISION_RETENTION
    assert "silindi" in candidate_client.get(f"/applications/{token}").text


def test_expired_draft_shows_the_message_and_is_removed_without_a_record(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """T13-2: expiry is a response only; retention deletes the draft, no record is written."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    candidate_app.dependency_overrides[get_now] = lambda: datetime.now(UTC) + BEYOND_TTL
    response = candidate_client.get(f"/drafts/{token}")
    assert response.status_code == GONE
    assert "Taslağınızın süresi doldu" in response.text
    assert f'href="/jobs/{backend_job}"' in response.text
    assert stored_applications(candidate_app) == []
    assert stored_decisions(candidate_app) == []
    with candidate_app.state.session_factory() as session:
        assert session.scalars(select(CandidateProfile)).all() == []
    assert candidate_client.get(f"/drafts/{token}").headers["location"] == f"/applications/{token}"


def test_confirming_an_expired_draft_is_refused(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """A late POST gets the expired page, not a ranking."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    candidate_app.dependency_overrides[get_now] = lambda: datetime.now(UTC) + BEYOND_TTL
    response = candidate_client.post(f"/drafts/{token}", data=confirm_form([]))
    assert response.status_code == GONE
    assert stored_applications(candidate_app) == []


def test_closed_posting_shows_no_form_and_refuses_uploads(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """The detail page of a closed posting has no upload form."""
    with candidate_app.state.session_factory() as session:
        close_posting(RetentionContext(session, datetime.now(UTC), DECISION_RETENTION_DAYS), backend_job)
        session.commit()
    page = candidate_client.get(f"/jobs/{backend_job}")
    assert page.status_code == 200
    assert "enctype" not in page.text
    assert "Bu ilan artık başvuru almıyor" in page.text
    refused = upload_cv(candidate_client, backend_job, build_cv_pdf("alpha"))
    assert refused.status_code == CONFLICT
    assert messages.POSTING_CLOSED in refused.text
    assert stored_applications(candidate_app) == []


def test_job_page_shows_filters_and_the_disclosure_text(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """Hard filters in Turkish, the KVKK disclosure, retention periods and the upload form."""
    job_id = create_open_posting(
        candidate_app,
        candidate_embedder,
        "Filtered",
        required_languages=["english", "turkish"],
        location="Denizli",
        remote_ok=False,
        min_years=2.0,
    )
    page = candidate_client.get(f"/jobs/{job_id}")
    assert page.status_code == 200
    for expected in ("İngilizce, Türkçe", "denizli", "2.0 yıl", "Aydınlatma metni (KVKK)",
                     "30 gün", "24 saat", "enctype=\"multipart/form-data\"", "en fazla 1 MB"):
        assert expected in page.text


def test_unreadable_pdf_and_unknown_posting_give_turkish_errors(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """A corrupt file is a 422 on the job page; a missing posting is a 404."""
    bad = upload_cv(candidate_client, backend_job, b"this is not a pdf")
    assert bad.status_code == UNPROCESSABLE
    assert messages.UPLOAD_UNREADABLE in bad.text
    assert stored_applications(candidate_app) == []
    assert candidate_client.get("/jobs/9999").status_code == 404
    assert upload_cv(candidate_client, 9999, build_cv_pdf("alpha")).status_code == 404


def test_the_home_page_lists_open_postings_twenty_per_page(
    candidate_client: TestClient, candidate_app: FastAPI, candidate_embedder: ControlledEmbedder
) -> None:
    """Pagination of the open postings; closed ones are not listed."""
    ids = [create_open_posting(candidate_app, candidate_embedder, f"Role {n:02d}") for n in range(25)]
    with candidate_app.state.session_factory() as session:
        close_posting(RetentionContext(session, datetime.now(UTC), DECISION_RETENTION_DAYS), ids[0])
        session.commit()
        assert session.get(JobPosting, ids[0]).status.value == "closed"
    first = candidate_client.get("/")
    second = candidate_client.get("/", params={"page": 2})
    assert first.text.count('href="/jobs/') == 20
    assert second.text.count('href="/jobs/') == 4
    assert "Role 00" not in first.text + second.text
    assert "Sayfa 1 / 2" in first.text


def snapshot_of_draft(application: FastAPI) -> tuple[object, ...]:
    """Everything a confirmation could change: sections, contact, profile_json, statuses."""
    with application.state.session_factory() as session:
        profile = session.scalars(select(CandidateProfile)).one()
        sections = [(item.type.value, item.text) for item in session.scalars(select(Section))]
        contact = (profile.contact.name, profile.contact.email, profile.contact.phone)
        statuses = [item.status.value for item in stored_applications(application)]
        return (sections, contact, json.dumps(profile.profile_json), statuses)


def test_an_empty_post_changes_nothing_and_shows_the_stored_draft(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """A post without any form field is malformed: 422, the draft is untouched."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    before = snapshot_of_draft(candidate_app)
    response = candidate_client.post(f"/drafts/{token}", content=b"")
    assert response.status_code == UNPROCESSABLE
    assert messages.FORM_INCOMPLETE in response.text
    assert CANDIDATE_EMAIL in response.text
    assert len([text for text in section_texts_of(response.text) if text]) == SECTIONS_IN_CV
    assert snapshot_of_draft(candidate_app) == before
    assert before[3] == ["draft"]
    partial = candidate_client.post(f"/drafts/{token}", data={"years_experience": "3"})
    assert partial.status_code == UNPROCESSABLE
    assert snapshot_of_draft(candidate_app) == before


def test_emptying_every_section_is_refused(
    candidate_client: TestClient, candidate_app: FastAPI, backend_job: int
) -> None:
    """A well-formed post with all texts blank needs at least one section."""
    token = token_of(upload_cv(candidate_client, backend_job, build_cv_pdf("alpha")))
    before = snapshot_of_draft(candidate_app)
    page = candidate_client.get(f"/drafts/{token}").text
    types = selected_types_of(page)
    response = candidate_client.post(
        f"/drafts/{token}", data=confirm_form([(row_type, "  ") for row_type in types])
    )
    assert response.status_code == UNPROCESSABLE
    assert messages.SECTION_REQUIRED in response.text
    assert snapshot_of_draft(candidate_app) == before
