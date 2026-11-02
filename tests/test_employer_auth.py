"""Employer login, cookie and the guard on `/employer/*` (SPEC 2, 9.1; SC-04)."""

import secrets

import pytest
from conftest import DUMMY_EMPLOYER_TOKEN, FakeEmbedder
from fastapi import FastAPI
from fastapi.testclient import TestClient
from web_support import form_data, log_in, stored_postings

import app.web.auth as auth_module
from app.config import Settings
from app.green.tracker import NoOpStageTracker
from app.web.auth import (
    EMPLOYER_COOKIE_NAME,
    derive_session_value,
    has_valid_session,
    tokens_match,
)
from app.web.paths import (
    EMPLOYER_HOME_PATH,
    EMPLOYER_LOGIN_PATH,
    EMPLOYER_LOGOUT_PATH,
    EMPLOYER_NEW_JOB_PATH,
)

LOGIN_ERROR_TEXT = "Erişim anahtarı yanlış"
PROTECTED_GET_PATHS = (EMPLOYER_HOME_PATH, EMPLOYER_NEW_JOB_PATH)


def test_correct_token_sets_a_hardened_cookie_that_is_not_the_token(
    web_client: TestClient,
) -> None:
    """HttpOnly, SameSite=Lax, path `/`; the value is derived, never the raw token."""
    response = web_client.post(EMPLOYER_LOGIN_PATH, data={"token": DUMMY_EMPLOYER_TOKEN})

    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_HOME_PATH
    set_cookie = response.headers["set-cookie"]
    lowered = set_cookie.lower()
    assert "httponly" in lowered
    assert "samesite=lax" in lowered
    assert "path=/;" in lowered or lowered.endswith("path=/")
    cookie_value = response.cookies[EMPLOYER_COOKIE_NAME]
    assert cookie_value != DUMMY_EMPLOYER_TOKEN
    assert DUMMY_EMPLOYER_TOKEN not in set_cookie
    assert cookie_value == derive_session_value(DUMMY_EMPLOYER_TOKEN)


@pytest.mark.parametrize("wrong_token", ["wrong", "", "şifre-ğüöçı", DUMMY_EMPLOYER_TOKEN + "x"])
def test_wrong_token_shows_a_turkish_error_and_sets_no_cookie(
    web_client: TestClient, wrong_token: str
) -> None:
    """Including non-ASCII input, which must not crash the constant-time compare."""
    response = web_client.post(EMPLOYER_LOGIN_PATH, data={"token": wrong_token})

    assert response.status_code == 401
    assert LOGIN_ERROR_TEXT in response.text
    assert "set-cookie" not in response.headers
    assert web_client.get(EMPLOYER_HOME_PATH).status_code == 303


def test_login_form_is_public_and_asks_for_a_password_field(web_client: TestClient) -> None:
    """GET the form without a cookie: 200, Turkish, a password input."""
    response = web_client.get(EMPLOYER_LOGIN_PATH)
    assert response.status_code == 200
    assert 'type="password"' in response.text
    assert "İşveren girişi" in response.text


@pytest.mark.parametrize("path", PROTECTED_GET_PATHS)
def test_protected_pages_redirect_to_login_without_a_cookie(
    web_client: TestClient, path: str
) -> None:
    """The guard redirects, it does not render."""
    response = web_client.get(path)
    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH


def test_protected_post_without_cookie_redirects_and_creates_nothing(
    web_client: TestClient, web_app: FastAPI, fake_embedder: FakeEmbedder
) -> None:
    """An unauthenticated form submission neither embeds nor stores anything."""
    response = web_client.post(EMPLOYER_NEW_JOB_PATH, data=form_data())

    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH
    assert fake_embedder.query_calls == []
    assert stored_postings(web_app) == []


@pytest.mark.parametrize(
    "forged_value",
    ["tampered", "", DUMMY_EMPLOYER_TOKEN, "a" * 64, derive_session_value("another-token")],
)
def test_tampered_or_raw_token_cookie_is_rejected(
    web_client: TestClient, forged_value: str
) -> None:
    """Only the derived value passes; the raw token as a cookie does not."""
    log_in(web_client)
    web_client.cookies.clear()
    web_client.cookies.set(EMPLOYER_COOKIE_NAME, forged_value)

    response = web_client.get(EMPLOYER_HOME_PATH)

    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH


def test_valid_cookie_lets_the_employer_in_and_login_form_redirects_home(
    web_client: TestClient,
) -> None:
    """After login the list is served and the login form is not shown again."""
    log_in(web_client)
    assert web_client.get(EMPLOYER_HOME_PATH).status_code == 200
    again = web_client.get(EMPLOYER_LOGIN_PATH)
    assert again.status_code == 303
    assert again.headers["location"] == EMPLOYER_HOME_PATH


def test_logout_expires_the_cookie_and_closes_the_pages(web_client: TestClient) -> None:
    """POST logout clears the cookie; the guard then redirects again."""
    log_in(web_client)

    response = web_client.post(EMPLOYER_LOGOUT_PATH)

    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH
    assert "max-age=0" in response.headers["set-cookie"].lower()
    assert web_client.get(EMPLOYER_HOME_PATH).status_code == 303


def test_logout_without_login_redirects_to_login(web_client: TestClient) -> None:
    """The logout route is behind the guard like every other employer route."""
    response = web_client.post(EMPLOYER_LOGOUT_PATH)
    assert response.status_code == 303
    assert response.headers["location"] == EMPLOYER_LOGIN_PATH


def test_login_and_cookie_checks_use_constant_time_comparison(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both the token check and the cookie check go through `secrets.compare_digest`."""
    calls: list[tuple[bytes, bytes]] = []
    real_compare = secrets.compare_digest

    def spy(first: bytes, second: bytes) -> bool:
        calls.append((first, second))
        return real_compare(first, second)

    monkeypatch.setattr(auth_module.secrets, "compare_digest", spy)
    log_in(web_client)
    assert len(calls) == 1
    web_client.get(EMPLOYER_HOME_PATH)
    assert len(calls) == 2
    assert all(isinstance(first, bytes) and isinstance(second, bytes) for first, second in calls)


def test_derived_value_depends_on_the_token_and_hides_it() -> None:
    """Different tokens give different cookies; the value is not the token."""
    first = derive_session_value("token-one")
    assert first != derive_session_value("token-two")
    assert "token-one" not in first


def test_token_comparison_handles_non_ascii_and_mismatch() -> None:
    """`compare_digest` on encoded bytes: equal strings match, others do not, no crash."""
    assert tokens_match("anahtar-ş", "anahtar-ş")
    assert not tokens_match("anahtar-ş", "anahtar-s")
    assert not tokens_match("", "anahtar")


@pytest.mark.parametrize("cookie_value", [None, "", "ğüşiöç", "a" * 64])
def test_malformed_cookie_values_are_invalid_without_raising(
    test_settings: Settings, cookie_value: str | None
) -> None:
    """The HTTP client cannot send a non-ASCII cookie, so the check is called directly."""
    assert has_valid_session(cookie_value, test_settings) is False
    assert has_valid_session(derive_session_value(DUMMY_EMPLOYER_TOKEN), test_settings)


def test_web_layer_always_uses_the_no_op_tracker(web_app: FastAPI) -> None:
    """Decision D3, even though `GREEN_TRACKING` defaults to true."""
    with TestClient(web_app) as client:
        assert client.get("/health").status_code == 200
        assert web_app.state.settings.green_tracking is True
        assert isinstance(web_app.state.tracker, NoOpStageTracker)
