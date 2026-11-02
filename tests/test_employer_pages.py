"""The employer postings list and the new-posting form (SC-04, SPEC 9.1)."""

from collections.abc import Iterator

import pytest
from conftest import FakeEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from web_support import NOW, add_postings, log_in, stored_postings, submit_job

from app.web.dependencies import get_now
from app.web.paths import EMPLOYER_HOME_PATH, EMPLOYER_NEW_JOB_PATH

UNPROCESSABLE = 422
REQUIREMENT_TEXTS = ("Building REST APIs", "Relational databases and SQL")


@pytest.fixture
def employer(web_app: FastAPI, web_client: TestClient) -> Iterator[TestClient]:
    """A logged-in client with a fixed clock."""
    web_app.dependency_overrides[get_now] = lambda: NOW
    log_in(web_client)
    yield web_client
    web_app.dependency_overrides.clear()


def test_form_page_is_turkish_with_six_requirement_rows(employer: TestClient) -> None:
    """The fixed number of rows, a weight select for each, the filter fields."""
    page = employer.get(EMPLOYER_NEW_JOB_PATH)

    assert page.status_code == 200
    assert page.text.count('name="requirement_text"') == 6
    assert page.text.count('name="requirement_weight"') == 6
    for field_name in ("title", "description", "languages", "location", "remote_ok", "min_years"):
        assert f'name="{field_name}"' in page.text
    assert "Yeni ilan" in page.text


def test_valid_form_creates_the_posting_embeds_once_and_redirects(
    employer: TestClient, web_app: FastAPI, fake_embedder: FakeEmbedder
) -> None:
    """One batch, one call; normalized filters; created_at from the injected clock."""
    response = submit_job(
        employer,
        languages="Türkçe, İngilizce",
        location="DENİZLİ",
        min_years="2,5",
        remote_ok="1",
    )

    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_HOME_PATH
    assert fake_embedder.query_calls == [list(REQUIREMENT_TEXTS)]
    (posting,) = stored_postings(web_app)
    assert posting.title == "Data Engineer"
    assert posting.required_languages == ["turkish", "english"]
    assert posting.location == "denizli"
    assert posting.remote_ok is True
    assert posting.min_years == 2.5
    assert posting.created_at == NOW
    assert [(item.text, item.weight) for item in posting.requirements] == [
        (REQUIREMENT_TEXTS[0], 3),
        (REQUIREMENT_TEXTS[1], 2),
    ]
    assert employer.get(EMPLOYER_HOME_PATH).text.count("Data Engineer") == 1


def test_remote_checkbox_unchecked_means_on_site_and_blank_years_means_none(
    employer: TestClient, web_app: FastAPI
) -> None:
    """Absent checkbox is false; a blank minimum is no minimum."""
    submit_job(employer)
    (posting,) = stored_postings(web_app)
    assert posting.remote_ok is False
    assert posting.min_years is None
    assert posting.required_languages == []
    assert posting.location is None


def test_blank_rows_are_ignored_wherever_they_sit(
    employer: TestClient, web_app: FastAPI, fake_embedder: FakeEmbedder
) -> None:
    """Rows 2, 3 and 5 are blank (one with a junk weight); the others keep their order."""
    rows = (("First", "1"), ("", "9"), ("  ", "2"), ("Fourth", "3"), ("", "x"), ("Sixth", "2"))
    response = submit_job(employer, requirements=rows)

    assert response.status_code == 303
    (posting,) = stored_postings(web_app)
    assert [(item.text, item.weight) for item in posting.requirements] == [
        ("First", 1),
        ("Fourth", 3),
        ("Sixth", 2),
    ]
    assert fake_embedder.query_calls == [["First", "Fourth", "Sixth"]]


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"title": "   "}, "İlan başlığı boş bırakılamaz."),
        ({"title": "x" * 121}, "en fazla 120 karakter"),
        ({"requirements": ()}, "En az bir gereksinim girmelisiniz."),
        ({"requirements": (("", "2"), ("  ", "3"))}, "En az bir gereksinim girmelisiniz."),
        ({"requirements": (("SQL", "0"),)}, "Gereksinim 1: ağırlık 1 ile 3 arasında olmalıdır."),
        ({"requirements": (("a", "2"), ("SQL", "4"))}, "Gereksinim 2: ağırlık 1 ile 3"),
        ({"requirements": (("SQL", "abc"),)}, "Gereksinim 1: ağırlık 1 ile 3"),
        ({"requirements": (("x" * 201, "2"),)}, "Gereksinim 1: metin en fazla 200"),
        ({"min_years": "abc"}, "Asgari deneyim 0 ile 60"),
        ({"min_years": "-1"}, "Asgari deneyim 0 ile 60"),
        ({"min_years": "61"}, "Asgari deneyim 0 ile 60"),
        ({"min_years": "nan"}, "Asgari deneyim 0 ile 60"),
        ({"location": "x" * 81}, "Konum en fazla 80"),
    ],
)
def test_invalid_forms_are_rejected_with_turkish_errors_and_store_nothing(
    employer: TestClient,
    web_app: FastAPI,
    fake_embedder: FakeEmbedder,
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    """422, the message on the page, no embedding call, no posting."""
    response = submit_job(employer, **overrides)

    assert response.status_code == UNPROCESSABLE
    assert expected_message in response.text
    assert fake_embedder.query_calls == []
    assert stored_postings(web_app) == []


def test_errors_re_render_the_form_with_the_entered_values(employer: TestClient) -> None:
    """Every field keeps what the employer typed, including the chosen weights."""
    rows = (("Python programming", "3"), ("SQL", "1"), ("bad weight", "7"))
    response = submit_job(
        employer,
        requirements=rows,
        title="Veri Mühendisi",
        description="Açıklama metni",
        languages="Türkçe",
        location="İzmir",
        min_years="3",
        remote_ok="1",
    )

    assert response.status_code == UNPROCESSABLE
    text = response.text
    for typed in ("Veri Mühendisi", "Açıklama metni", "Türkçe", "İzmir", "Python programming"):
        assert typed in text
    assert 'value="3"' in text
    assert " checked" in text
    assert text.count("<option value=\"1\" selected>") == 1


@pytest.mark.parametrize("weight", ["1", "3"])
def test_weight_bounds_are_accepted_in_the_form(
    employer: TestClient, web_app: FastAPI, weight: str
) -> None:
    """Inclusive bounds 1 and 3."""
    response = submit_job(employer, requirements=(("SQL", weight),))
    assert response.status_code == 303
    assert stored_postings(web_app)[0].requirements[0].weight == int(weight)


def test_extra_posted_rows_beyond_the_fixed_count_are_ignored(
    employer: TestClient, web_app: FastAPI
) -> None:
    """A hand-made request with seven rows only uses the first six."""
    rows = tuple((f"Requirement {number}", "2") for number in range(1, 8))
    response = submit_job(employer, requirements=rows)
    assert response.status_code == 303
    assert len(stored_postings(web_app)[0].requirements) == 6


def test_script_in_a_title_is_escaped_in_list_and_form(
    employer: TestClient, web_app: FastAPI
) -> None:
    """Jinja2 autoescaping on: no raw tag reaches the HTML."""
    payload = "<script>alert(1)</script>"
    assert submit_job(employer, title=payload).status_code == 303

    listing = employer.get(EMPLOYER_HOME_PATH).text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in listing
    assert payload not in listing

    broken = submit_job(employer, title=payload, requirements=())
    assert broken.status_code == UNPROCESSABLE
    assert payload not in broken.text
    assert "&lt;script&gt;" in broken.text


def test_quote_in_a_field_cannot_break_out_of_the_attribute(employer: TestClient) -> None:
    """An attribute-value injection attempt stays inside `value="..."`."""
    response = submit_job(employer, title='"><b onmouseover=x>', requirements=())
    assert '"><b' not in response.text
    assert "&#34;&gt;&lt;b" in response.text


def test_list_shows_status_in_turkish_and_links_to_the_detail_path(
    employer: TestClient, web_app: FastAPI, fake_embedder: FakeEmbedder
) -> None:
    """Status label, creation date and the (T-15) detail link."""
    add_postings(web_app, fake_embedder, 1)
    page = employer.get(EMPLOYER_HOME_PATH)

    assert "Açık" in page.text
    assert "/employer/jobs/1" in page.text
    assert "01.03.2026" in page.text
    assert "Yeni ilan oluştur" in page.text


def test_empty_list_says_so(employer: TestClient) -> None:
    """No postings: a Turkish empty state, no table."""
    page = employer.get(EMPLOYER_HOME_PATH)
    assert "Henüz ilan yok." in page.text
    assert "<table" not in page.text


def test_form_defaults_are_empty_and_weight_two_is_preselected(employer: TestClient) -> None:
    """Default weight is 2 (medium) in every row."""
    page = employer.get(EMPLOYER_NEW_JOB_PATH).text
    assert page.count('<option value="2" selected>') == 6
