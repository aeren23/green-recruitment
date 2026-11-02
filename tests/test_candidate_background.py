"""T13-1: top-k summaries are written after the response, from the committed state."""


import pytest
from candidate_support import (
    AsgiRecorder,
    application_summary,
    call_asgi,
    confirm_form,
    create_open_posting,
    selected_types_of,
    section_texts_of,
    stored_applications,
    token_of,
    upload_cv,
    urlencoded_form,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline_support import SUMMARY_REPLY, ControlledEmbedder, FakeLlm, build_cv_pdf
from sqlalchemy import select

from app.config import Settings
from app.green.counters import read_counters
from app.main import create_app
from app.models import Application

RESPONSE_SENT = "response sent"
LLM_CALLED = "llm called"


class OrderRecordingLlm(FakeLlm):
    """A fake LLM that notes when it is called and what the database looked like then."""

    def __init__(self, events: list[str], application: FastAPI | None = None) -> None:
        super().__init__()
        self.events = events
        self.application = application
        self.status_when_called: list[str] = []

    def generate(self, prompt: str, expect_json: bool = False):
        """Record the call, and the stored status of every application at that moment."""
        self.events.append(LLM_CALLED)
        with self.application.state.session_factory() as session:
            self.status_when_called = [
                row.status.value for row in session.scalars(select(Application))
            ]
        return super().generate(prompt, expect_json)


class SentMarker(AsgiRecorder):
    """An ASGI recorder that notes when the response body is complete."""

    def __init__(self, events: list[str]) -> None:
        super().__init__([urlencoded_form(confirm_form([]))[0]])
        self.events = events

    async def send(self, message: dict) -> None:
        """Record the message; the last body chunk means the client has its response."""
        await super().send(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            self.events.append(RESPONSE_SENT)


def draft_with_sections(client: TestClient, job_id: int) -> tuple[str, dict]:
    """Upload a CV and return its token and a form that re-posts its sections."""
    token = token_of(upload_cv(client, job_id, build_cv_pdf("alpha")))
    page = client.get(f"/drafts/{token}").text
    rows = [(kind, text) for kind, text in zip(selected_types_of(page), section_texts_of(page), strict=True) if text]
    return token, confirm_form(rows)


def test_the_response_is_complete_before_the_llm_is_called(test_settings: Settings) -> None:
    """The order of events proves the request never waits for the LLM."""
    events: list[str] = []
    embedder = ControlledEmbedder()
    llm = OrderRecordingLlm(events)
    application = create_app(test_settings, embedder, llm)
    llm.application = application
    with TestClient(application, follow_redirects=False) as client:
        job_id = create_open_posting(application, embedder, "Backend Developer")
        token, form = draft_with_sections(client, job_id)
        body, headers = urlencoded_form(form)
        recorder = SentMarker(events)
        recorder._chunks = [body]
        headers["content-length"] = str(len(body))
        call_asgi(application, f"/drafts/{token}", headers, recorder)
    assert recorder.status == 303
    assert events == [RESPONSE_SENT, LLM_CALLED]
    assert llm.status_when_called == ["ranked"]
    assert application_summary(application, token) == SUMMARY_REPLY


def test_after_the_background_task_the_top_k_summary_exists(
    candidate_client: TestClient, candidate_app: FastAPI, fake_llm: FakeLlm
) -> None:
    """The confirmation commits, then the summary of the top-k candidate is written."""
    job_id = create_open_posting(candidate_app, candidate_app.state.embedder, "Backend Developer")
    token, form = draft_with_sections(candidate_client, job_id)
    response = candidate_client.post(f"/drafts/{token}", data=form)
    assert response.status_code == 303
    assert application_summary(candidate_app, token) == SUMMARY_REPLY
    assert len(fake_llm.prompts) == 1
    with candidate_app.state.session_factory() as session:
        assert read_counters(session).llm_calls == 1


def test_an_unreachable_llm_leaves_the_summary_empty_and_the_flow_working(
    test_settings: Settings,
) -> None:
    """Ollama down: no summary, but the candidate is ranked and sees the status."""
    embedder = ControlledEmbedder()
    llm = FakeLlm(unavailable=True)
    application = create_app(test_settings, embedder, llm)
    with TestClient(application, follow_redirects=False) as client:
        job_id = create_open_posting(application, embedder, "Backend Developer")
        token, form = draft_with_sections(client, job_id)
        response = client.post(f"/drafts/{token}", data=form)
        assert response.status_code == 303
        assert "Değerlendirmede" in client.get(response.headers["location"]).text
        assert application_summary(application, token) is None
        assert [row.status.value for row in stored_applications(application)] == ["ranked"]
    assert llm.attempts == 1


def test_a_failing_background_job_is_logged_without_the_error_text(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected error after the response is logged by type only, and the candidate is unaffected."""
    import app.web.background as background

    def explode(*args: object, **kwargs: object) -> int:
        raise RuntimeError("CV TEXT THAT MUST NOT BE LOGGED")

    monkeypatch.setattr(background, "fill_top_k_summaries", explode)
    embedder = ControlledEmbedder()
    application = create_app(test_settings, embedder, FakeLlm())
    with TestClient(application, follow_redirects=False) as client, caplog.at_level("INFO"):
        job_id = create_open_posting(application, embedder, "Backend Developer")
        token, form = draft_with_sections(client, job_id)
        assert client.post(f"/drafts/{token}", data=form).status_code == 303
    assert "RuntimeError" in caplog.text
    assert "MUST NOT BE LOGGED" not in caplog.text
