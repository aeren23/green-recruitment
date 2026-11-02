"""OllamaClient: request shape, token counts, errors, logging hygiene."""

import logging
import subprocess
import sys

import httpx
import pytest
from llm_support import (
    TEST_BASE_URL,
    TEST_MODEL,
    TEST_TIMEOUT_SECONDS,
    ScriptedOllama,
    ollama_reply,
)

from app.llm.ollama import LlmClient, LlmUnavailableError, WarmableLlm

PROMPT_MARKER = "PROMPT-SECRET-MARKER-7731"
RESPONSE_MARKER = "RESPONSE-SECRET-MARKER-5519"
OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 120}
GENERATE_URL = f"{TEST_BASE_URL}/api/generate"
IMPORT_PROBE = (
    "import socket\n"
    "def refuse(*args, **kwargs):\n"
    "    raise AssertionError('network used at import')\n"
    "socket.socket.connect = refuse\n"
    "socket.create_connection = refuse\n"
    "import app.llm, app.llm.ollama, app.llm.prompts, app.llm.tasks, app.llm.usage\n"
)


def test_generate_sends_the_specified_request() -> None:
    """Model, stream false, the three options and the endpoint path are sent."""
    ollama = ScriptedOllama(ollama_reply("ok"))
    ollama.client().generate("hello")
    request = ollama.requests[0]
    assert request.method == "POST"
    assert str(request.url) == GENERATE_URL
    body = ollama.bodies()[0]
    assert body["model"] == TEST_MODEL
    assert body["stream"] is False
    assert body["prompt"] == "hello"
    assert body["options"] == OPTIONS


def test_format_json_is_sent_only_when_requested() -> None:
    """`format: json` is present for the baseline and absent for summaries."""
    ollama = ScriptedOllama(ollama_reply("{}"), ollama_reply("text"))
    client = ollama.client()
    client.generate("p", expect_json=True)
    client.generate("p")
    with_format, without_format = ollama.bodies()
    assert with_format["format"] == "json"
    assert "format" not in without_format


def test_base_url_trailing_slash_is_tolerated() -> None:
    """A trailing slash in the base URL does not double the path separator."""
    ollama = ScriptedOllama(ollama_reply("ok"))
    ollama.client(base_url=f"{TEST_BASE_URL}/").generate("p")
    assert str(ollama.requests[0].url) == GENERATE_URL


def test_response_carries_text_tokens_and_duration() -> None:
    """Token counts and text come from the Ollama payload."""
    ollama = ScriptedOllama(ollama_reply("answer", prompt_tokens=42, completion_tokens=9))
    response = ollama.client().generate("p")
    assert response.text == "answer"
    assert response.prompt_eval_count == 42
    assert response.eval_count == 9
    assert response.duration_seconds >= 0


def test_missing_token_counts_default_to_zero() -> None:
    """Ollama omits `prompt_eval_count` on prompt-cache hits."""
    reply = httpx.Response(200, json={"response": "x", "done": True})
    response = ScriptedOllama(reply).client().generate("p")
    assert (response.prompt_eval_count, response.eval_count) == (0, 0)


def test_the_configured_timeout_is_applied_to_the_request() -> None:
    """The per-request timeout equals the configured one."""
    ollama = ScriptedOllama(ollama_reply("ok"))
    ollama.client().generate("p")
    assert ollama.requests[0].extensions["timeout"]["read"] == TEST_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.ConnectError("refused"),
    ],
)
def test_transport_failures_raise_unavailable(failure: Exception) -> None:
    """Timeouts and connection errors become `LlmUnavailableError`."""
    with pytest.raises(LlmUnavailableError):
        ScriptedOllama(failure).client().generate("p")


@pytest.mark.parametrize("status", [404, 500, 503, 301])
def test_non_success_status_raises_unavailable(status: int) -> None:
    """Any non-2xx status is an error, with the status in the message."""
    reply = httpx.Response(status, json={"error": RESPONSE_MARKER})
    with pytest.raises(LlmUnavailableError, match=str(status)) as caught:
        ScriptedOllama(reply).client().generate("p")
    assert RESPONSE_MARKER not in str(caught.value)


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["list"]),
        httpx.Response(200, json={"no_response_key": 1}),
        httpx.Response(200, json={"response": 5}),
    ],
)
def test_invalid_payload_raises_unavailable(reply: httpx.Response) -> None:
    """A 2xx body that is not a generate payload is an error."""
    with pytest.raises(LlmUnavailableError):
        ScriptedOllama(reply).client().generate("p")


def test_errors_carry_no_prompt_text_and_no_chained_exception() -> None:
    """The prompt is not in the message or in the exception chain."""
    ollama = ScriptedOllama(httpx.ConnectError(f"refused {PROMPT_MARKER}"))
    with pytest.raises(LlmUnavailableError) as caught:
        ollama.client().generate(PROMPT_MARKER)
    assert PROMPT_MARKER not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_logs_hold_no_prompt_or_response_text(caplog: pytest.LogCaptureFixture) -> None:
    """Success and failure paths log counts only (KVKK)."""
    ollama = ScriptedOllama(
        ollama_reply(RESPONSE_MARKER), httpx.ReadTimeout(PROMPT_MARKER)
    )
    client = ollama.client()
    with caplog.at_level(logging.DEBUG):
        client.generate(PROMPT_MARKER)
        with pytest.raises(LlmUnavailableError):
            client.generate(PROMPT_MARKER)
    assert caplog.records
    assert PROMPT_MARKER not in caplog.text
    assert RESPONSE_MARKER not in caplog.text


def test_warm_up_sends_a_promptless_request_with_the_run_options() -> None:
    """Warm-up loads the model with the same options, without a prompt."""
    ollama = ScriptedOllama(ollama_reply(""))
    ollama.client().warm_up()
    assert len(ollama.requests) == 1
    body = ollama.bodies()[0]
    assert body["model"] == TEST_MODEL
    assert body["stream"] is False
    assert body["options"] == OPTIONS
    assert "prompt" not in body


def test_warm_up_failure_raises_unavailable() -> None:
    """A warm-up that cannot reach Ollama is reported, not swallowed."""
    with pytest.raises(LlmUnavailableError):
        ScriptedOllama(httpx.ConnectError("refused")).client().warm_up()


def test_client_satisfies_the_protocols() -> None:
    """`OllamaClient` can stand wherever the protocols are expected."""
    client = ScriptedOllama().client()
    typed_client: LlmClient = client
    typed_warm: WarmableLlm = client
    assert typed_client is typed_warm


def test_importing_the_llm_package_opens_no_connection() -> None:
    """Import has no network side effects: connects are blocked in a fresh process."""
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
