"""Baseline scoring and green summary: retry rule, parsing, usage, error policy."""

import logging

import httpx
import pytest
from llm_support import ScriptedOllama, ollama_reply, score_reply

from app.llm.ollama import LlmUnavailableError
from app.llm.prompts import BaselinePromptInput, CvSection, SummaryPromptInput
from app.llm.tasks import parse_baseline_response, score_cv_with_llm, summarize_candidate
from app.llm.usage import LlmUsage

REQUIREMENTS = ("Building REST APIs",)
CV_MARKER = "CV-SECRET-MARKER-2210"
SUMMARY_MARKER = "SUMMARY-SECRET-MARKER-8842"
UNSELECTED_MARKER = "UNSELECTED-EDUCATION-MARKER"
MALFORMED_TEXT = "this is not json"
OVERLONG_CV_CHARS = 10_000
TRUNCATION_LIMIT = 100
SELECTED_ID = 1
UNSELECTED_ID = 2
BASELINE_INPUT = BaselinePromptInput(REQUIREMENTS, f"Experience. {CV_MARKER}")
SUMMARY_INPUT = SummaryPromptInput(
    REQUIREMENTS,
    [
        CvSection(SELECTED_ID, "experience", f"Built APIs. {SUMMARY_MARKER}"),
        CvSection(UNSELECTED_ID, "education", f"BSc. {UNSELECTED_MARKER}"),
    ],
    {SELECTED_ID},
)


def malformed_reply(prompt_tokens: int = 20, completion_tokens: int = 8) -> httpx.Response:
    """A reply whose text is not JSON."""
    return ollama_reply(MALFORMED_TEXT, prompt_tokens, completion_tokens)


def test_valid_score_is_returned_with_reason() -> None:
    """A well-formed answer needs one call."""
    ollama = ScriptedOllama(score_reply(87))
    usage = LlmUsage()
    result = score_cv_with_llm(ollama.client(), BASELINE_INPUT, usage)
    assert result.score == 87
    assert result.reason == "Short reason."
    assert len(ollama.requests) == 1
    assert usage.failures == 0


def test_baseline_request_asks_for_json_format() -> None:
    """The baseline path sets `format: json`."""
    ollama = ScriptedOllama(score_reply(50))
    score_cv_with_llm(ollama.client(), BASELINE_INPUT, LlmUsage())
    assert ollama.bodies()[0]["format"] == "json"


@pytest.mark.parametrize("boundary", [0, 100])
def test_boundary_scores_are_valid(boundary: int) -> None:
    """0 and 100 are accepted."""
    result = score_cv_with_llm(
        ScriptedOllama(score_reply(boundary)).client(), BASELINE_INPUT, LlmUsage()
    )
    assert result.score == boundary


def test_malformed_answer_is_retried_once_and_can_recover() -> None:
    """First answer malformed, second valid: score kept, two calls, no failure."""
    ollama = ScriptedOllama(malformed_reply(), score_reply(64))
    usage = LlmUsage()
    result = score_cv_with_llm(ollama.client(), BASELINE_INPUT, usage)
    assert result.score == 64
    assert len(ollama.requests) == 2
    assert usage.calls == 2
    assert usage.failures == 0


def test_twice_malformed_gives_none_and_is_counted() -> None:
    """Two malformed answers: no third attempt, `score=None`, one failure."""
    ollama = ScriptedOllama(malformed_reply(), malformed_reply())
    usage = LlmUsage()
    result = score_cv_with_llm(ollama.client(), BASELINE_INPUT, usage)
    assert result.score is None
    assert len(ollama.requests) == 2
    assert usage.failures == 1


def test_tokens_of_malformed_attempts_are_counted() -> None:
    """Both attempts add their prompt and completion tokens."""
    ollama = ScriptedOllama(malformed_reply(20, 8), score_reply(70, 21, 9))
    usage = LlmUsage()
    score_cv_with_llm(ollama.client(), BASELINE_INPUT, usage)
    assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (2, 41, 17)


@pytest.mark.parametrize(
    "text",
    [
        MALFORMED_TEXT,
        "",
        "[1, 2]",
        '"85"',
        "85",
        '{"reason": "no score"}',
        '{"score": null}',
        '{"score": "85"}',
        '{"score": 85.5}',
        '{"score": 85.0}',
        '{"score": true}',
        '{"score": -1}',
        '{"score": 101}',
        '{"score": [85]}',
    ],
)
def test_strict_parsing_rejects_malformed_scores(text: str) -> None:
    """Missing key, wrong type or out-of-range score is malformed."""
    assert parse_baseline_response(text) is None


def test_non_string_reason_is_dropped_but_score_kept() -> None:
    """A usable score is not discarded because the reason has the wrong type."""
    result = parse_baseline_response('{"score": 40, "reason": 12}')
    assert result is not None
    assert (result.score, result.reason) == (40, None)


@pytest.mark.parametrize("bad_score", [101, -5, 3.5, "high"])
def test_bad_score_in_a_full_run_is_retried_then_none(bad_score: object) -> None:
    """Invalid scores go through the same retry-once path."""
    ollama = ScriptedOllama(score_reply(bad_score), score_reply(bad_score))
    usage = LlmUsage()
    result = score_cv_with_llm(ollama.client(), BASELINE_INPUT, usage)
    assert result.score is None
    assert usage.failures == 1
    assert len(ollama.requests) == 2


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")]
)
def test_baseline_raises_when_ollama_is_unavailable(failure: Exception) -> None:
    """A baseline run without its LLM is invalid, so the error propagates."""
    with pytest.raises(LlmUnavailableError):
        score_cv_with_llm(ScriptedOllama(failure).client(), BASELINE_INPUT, LlmUsage())


def test_baseline_raises_on_non_success_status() -> None:
    """A 500 from Ollama propagates for the baseline."""
    ollama = ScriptedOllama(httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(LlmUnavailableError, match="500"):
        score_cv_with_llm(ollama.client(), BASELINE_INPUT, LlmUsage())


def test_baseline_unavailable_during_retry_propagates() -> None:
    """Ollama failing on the retry is still an unavailable error."""
    ollama = ScriptedOllama(malformed_reply(), httpx.ConnectError("gone"))
    with pytest.raises(LlmUnavailableError):
        score_cv_with_llm(ollama.client(), BASELINE_INPUT, LlmUsage())


def test_baseline_prompt_sent_to_ollama_is_truncated() -> None:
    """The prompt on the wire holds at most the configured number of CV characters."""
    ollama = ScriptedOllama(score_reply(10))
    prompt_input = BaselinePromptInput(
        REQUIREMENTS, "x" * OVERLONG_CV_CHARS, max_cv_chars=TRUNCATION_LIMIT
    )
    score_cv_with_llm(ollama.client(), prompt_input, LlmUsage())
    assert ollama.bodies()[0]["prompt"].count("x") == TRUNCATION_LIMIT


def test_summary_returns_text_and_counts_tokens() -> None:
    """A summary is stripped, returned, and its tokens are accumulated."""
    ollama = ScriptedOllama(ollama_reply("  Aday uygun.  ", 30, 12))
    usage = LlmUsage()
    summary = summarize_candidate(ollama.client(), SUMMARY_INPUT, usage)
    assert summary == "Aday uygun."
    assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (1, 30, 12)
    assert "format" not in ollama.bodies()[0]


def test_summary_prompt_on_the_wire_has_selected_sections_only() -> None:
    """The unselected section marker never reaches Ollama."""
    ollama = ScriptedOllama(ollama_reply("ok"))
    summarize_candidate(ollama.client(), SUMMARY_INPUT, LlmUsage())
    prompt = ollama.bodies()[0]["prompt"]
    assert SUMMARY_MARKER in prompt
    assert UNSELECTED_MARKER not in prompt


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")]
)
def test_summary_returns_none_when_ollama_is_unavailable(failure: Exception) -> None:
    """The pipeline keeps running: no exception, `None`, one failure, no call."""
    usage = LlmUsage()
    client = ScriptedOllama(failure).client()
    assert summarize_candidate(client, SUMMARY_INPUT, usage) is None
    assert (usage.failures, usage.calls) == (1, 0)


def test_summary_returns_none_on_non_success_status() -> None:
    """A 503 degrades the summary to `None`."""
    ollama = ScriptedOllama(httpx.Response(503))
    assert summarize_candidate(ollama.client(), SUMMARY_INPUT, LlmUsage()) is None


def test_empty_summary_text_is_none_and_counted() -> None:
    """Whitespace-only output is treated as no summary."""
    usage = LlmUsage()
    client = ScriptedOllama(ollama_reply("   ", 9, 1)).client()
    assert summarize_candidate(client, SUMMARY_INPUT, usage) is None
    assert (usage.calls, usage.failures) == (1, 1)


def test_usage_accumulates_across_tasks() -> None:
    """One `LlmUsage` can be shared by several calls."""
    ollama = ScriptedOllama(score_reply(50, 10, 4), ollama_reply("Ozet.", 6, 3))
    usage = LlmUsage()
    client = ollama.client()
    score_cv_with_llm(client, BASELINE_INPUT, usage)
    summarize_candidate(client, SUMMARY_INPUT, usage)
    assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (2, 16, 7)


def test_no_cv_or_response_text_in_logs_or_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Across malformed, failed and unavailable paths nothing sensitive leaks."""
    leaky = f"{MALFORMED_TEXT} {SUMMARY_MARKER}"
    ollama = ScriptedOllama(
        ollama_reply(leaky),
        ollama_reply(leaky),
        httpx.ConnectError(CV_MARKER),
        httpx.ConnectError(CV_MARKER),
    )
    client = ollama.client()
    with caplog.at_level(logging.DEBUG):
        score_cv_with_llm(client, BASELINE_INPUT, LlmUsage())
        summarize_candidate(client, SUMMARY_INPUT, LlmUsage())
        with pytest.raises(LlmUnavailableError) as caught:
            score_cv_with_llm(client, BASELINE_INPUT, LlmUsage())
    assert caplog.records
    for marker in (CV_MARKER, SUMMARY_MARKER, MALFORMED_TEXT):
        assert marker not in caplog.text
        assert marker not in str(caught.value)
