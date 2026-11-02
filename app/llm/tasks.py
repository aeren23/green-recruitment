"""The two LLM tasks: baseline CV scoring and green candidate summary.

Decision (baseline): an unreachable Ollama raises `LlmUnavailableError` and is
not turned into `score=None`, because a baseline run without its LLM is
invalid and must not produce a silently wrong energy comparison.

Decision (green): an unreachable Ollama makes `summarize_candidate` return
`None` and the pipeline keeps running (SPEC 6.6).

KVKK: prompts and responses are never logged; only counts and attempt numbers are.
"""

import json
import logging
from dataclasses import dataclass

from app.llm.ollama import LlmClient, LlmUnavailableError
from app.llm.prompts import (
    BaselinePromptInput,
    SummaryPromptInput,
    build_baseline_prompt,
    build_summary_prompt,
)
from app.llm.usage import LlmUsage

logger = logging.getLogger(__name__)

SCORE_KEY = "score"
REASON_KEY = "reason"
MIN_SCORE = 0
MAX_SCORE = 100
MALFORMED_RETRIES = 1

MALFORMED_LOG_MESSAGE = "baseline response malformed attempt=%d of %d"
EXHAUSTED_LOG_MESSAGE = "baseline score unavailable after %d attempts"
SUMMARY_UNAVAILABLE_LOG_MESSAGE = "summary skipped: LLM unavailable"
SUMMARY_EMPTY_LOG_MESSAGE = "summary skipped: LLM returned no text"


@dataclass(frozen=True)
class BaselineScore:
    """Result of the baseline scoring task.

    Attributes:
        score: Integer 0 to 100, or `None` when the answer stayed malformed
            after the single retry.
        reason: The model's one-sentence reason, when it gave a string.
    """

    score: int | None
    reason: str | None = None


def _is_valid_score(value: object) -> bool:
    """True for an integer 0 to 100; booleans and floats are rejected."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and MIN_SCORE <= value <= MAX_SCORE
    )


def parse_baseline_response(text: str) -> BaselineScore | None:
    """Parse a baseline answer strictly.

    Args:
        text: The raw model output.

    Returns:
        The score, or `None` when the answer is not a JSON object whose
        `score` is an integer from 0 to 100.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not _is_valid_score(data.get(SCORE_KEY)):
        return None
    reason = data.get(REASON_KEY)
    return BaselineScore(
        score=data[SCORE_KEY], reason=reason if isinstance(reason, str) else None
    )


def score_cv_with_llm(
    client: LlmClient, prompt_input: BaselinePromptInput, usage: LlmUsage
) -> BaselineScore:
    """Score one CV against the requirements (baseline `llm_score` stage).

    A malformed answer is retried exactly once; if it is malformed again the
    result is `score=None` and `usage.failures` is incremented. Every
    completed attempt adds its tokens to `usage`.

    Raises:
        LlmUnavailableError: If Ollama is unreachable or errors; the baseline
            run is then invalid.
        ValueError: If the prompt input has no requirements.
    """
    prompt = build_baseline_prompt(prompt_input)
    attempts = MALFORMED_RETRIES + 1
    for attempt in range(1, attempts + 1):
        response = client.generate(prompt, expect_json=True)
        usage.record_response(response)
        parsed = parse_baseline_response(response.text)
        if parsed is not None:
            return parsed
        logger.warning(MALFORMED_LOG_MESSAGE, attempt, attempts)
    logger.warning(EXHAUSTED_LOG_MESSAGE, attempts)
    usage.record_failure()
    return BaselineScore(score=None)


def summarize_candidate(
    client: LlmClient, prompt_input: SummaryPromptInput, usage: LlmUsage
) -> str | None:
    """Summarize a top-k candidate from the matrix-selected sections.

    Returns:
        The summary text (at most 2 Turkish sentences by prompt), or `None`
        when Ollama is unreachable or returned nothing; the pipeline then
        keeps running with an empty summary.

    Raises:
        ValueError: If the prompt input has no requirements or no sections.
    """
    prompt = build_summary_prompt(prompt_input)
    try:
        response = client.generate(prompt)
    except LlmUnavailableError:
        logger.warning(SUMMARY_UNAVAILABLE_LOG_MESSAGE)
        usage.record_failure()
        return None
    usage.record_response(response)
    summary = response.text.strip()
    if not summary:
        logger.warning(SUMMARY_EMPTY_LOG_MESSAGE)
        usage.record_failure()
        return None
    return summary
