"""Ollama client over `httpx` (SPEC sections 6.6 and 7.4).

The client is a transport only: it sends one prompt, returns one completion and
reports token counts. Counting calls and deciding what a malformed answer
means belong to `app/llm/tasks.py`.

KVKK: prompts and responses contain CV text. Neither is ever put into a log
record or an exception message; only status, durations and token counts are.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

# TODO(config): these defaults should move to `app/config.py` / `.env` later.
DEFAULT_TEMPERATURE = 0
DEFAULT_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 120
DEFAULT_TIMEOUT_SECONDS = 120.0

GENERATE_PATH = "/api/generate"
JSON_FORMAT = "json"
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX = 299

REQUEST_MODEL = "model"
REQUEST_PROMPT = "prompt"
REQUEST_STREAM = "stream"
REQUEST_OPTIONS = "options"
REQUEST_FORMAT = "format"
RESPONSE_TEXT = "response"
RESPONSE_PROMPT_TOKENS = "prompt_eval_count"
RESPONSE_COMPLETION_TOKENS = "eval_count"

REASON_TIMEOUT = "request timed out"
REASON_CONNECTION = "connection failed"
REASON_STATUS = "unexpected HTTP status {status}"
REASON_BODY = "response body is not a valid generate payload"
UNAVAILABLE_MESSAGE = "Ollama is unavailable: {reason}."
CALL_LOG_MESSAGE = (
    "ollama generate status=%d duration_s=%.3f prompt_tokens=%d completion_tokens=%d"
)
FAILURE_LOG_MESSAGE = "ollama request failed reason=%s"


class LlmUnavailableError(RuntimeError):
    """Ollama cannot be reached or answered with an error.

    The message never contains prompt or response text.
    """


@dataclass(frozen=True)
class LlmResponse:
    """One completed generation.

    Attributes:
        text: The model output. It may contain CV-derived text.
        prompt_eval_count: Prompt tokens evaluated (0 when Ollama omits it,
            which happens when the prompt is served from its cache).
        eval_count: Completion tokens generated.
        duration_seconds: Wall-clock time of the HTTP call.
    """

    text: str
    prompt_eval_count: int
    eval_count: int
    duration_seconds: float


class LlmClient(Protocol):
    """What the pipelines need from a language model: one completion."""

    def generate(self, prompt: str, expect_json: bool = False) -> LlmResponse:
        """Generate one completion.

        Args:
            prompt: The full prompt text.
            expect_json: Ask the model to constrain its output to valid JSON.

        Raises:
            LlmUnavailableError: If the model cannot be reached or fails.
        """
        ...


class WarmableLlm(Protocol):
    """A model that can be loaded into memory ahead of the measured run."""

    def warm_up(self) -> None:
        """Load the model into memory without generating text."""
        ...


@dataclass(frozen=True)
class GenerationOptions:
    """Ollama sampling options; field names are the Ollama option keys.

    Both pipelines use the same values so the comparison is fair (SPEC 7.4).
    """

    temperature: int = DEFAULT_TEMPERATURE
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT


@dataclass(frozen=True)
class OllamaConfig:
    """Where and how to talk to Ollama."""

    base_url: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    options: GenerationOptions = field(default_factory=GenerationOptions)


def _read_payload(http_response: httpx.Response) -> tuple[str | None, dict[str, Any]]:
    """Validate a response; return `(failure_reason, payload)`."""
    status = http_response.status_code
    if not HTTP_SUCCESS_MIN <= status <= HTTP_SUCCESS_MAX:
        return REASON_STATUS.format(status=status), {}
    try:
        payload = http_response.json()
    except ValueError:
        return REASON_BODY, {}
    if not isinstance(payload, dict) or not isinstance(payload.get(RESPONSE_TEXT), str):
        return REASON_BODY, {}
    return None, payload


def _token_count(payload: dict[str, Any], key: str) -> int:
    """Read a token counter, treating an absent or invalid value as zero."""
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class OllamaClient:
    """`LlmClient` and `WarmableLlm` backed by a local Ollama server.

    Args:
        http_client: Injected `httpx.Client`; the caller owns its lifetime.
        config: Endpoint, model, timeout and generation options.
    """

    def __init__(self, http_client: httpx.Client, config: OllamaConfig) -> None:
        self._http_client = http_client
        self._config = config
        self._generate_url = f"{config.base_url.rstrip('/')}{GENERATE_PATH}"

    def generate(self, prompt: str, expect_json: bool = False) -> LlmResponse:
        """Generate one completion with `stream: false`.

        Args:
            prompt: The full prompt text (never logged).
            expect_json: Send `format: "json"`; used by the baseline only.

        Raises:
            LlmUnavailableError: On timeout, connection error, non-2xx status
                or a body that is not a generate payload.
        """
        body = self._build_body()
        body[REQUEST_PROMPT] = prompt
        if expect_json:
            body[REQUEST_FORMAT] = JSON_FORMAT
        started = time.perf_counter()
        status, payload = self._post(body)
        response = LlmResponse(
            text=payload[RESPONSE_TEXT],
            prompt_eval_count=_token_count(payload, RESPONSE_PROMPT_TOKENS),
            eval_count=_token_count(payload, RESPONSE_COMPLETION_TOKENS),
            duration_seconds=time.perf_counter() - started,
        )
        logger.info(
            CALL_LOG_MESSAGE,
            status,
            response.duration_seconds,
            response.prompt_eval_count,
            response.eval_count,
        )
        return response

    def warm_up(self) -> None:
        """Load the model into memory (SPEC 7.4) with the run's `num_ctx`.

        A request without a prompt only loads the model. It carries the same
        options as real calls, because Ollama reloads a model when `num_ctx`
        differs.

        Raises:
            LlmUnavailableError: If Ollama cannot load the model.
        """
        self._post(self._build_body())

    def _build_body(self) -> dict[str, Any]:
        """The request fields shared by warm-up and real calls."""
        return {
            REQUEST_MODEL: self._config.model,
            REQUEST_STREAM: False,
            REQUEST_OPTIONS: asdict(self._config.options),
        }

    def _post(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST to `/api/generate`; return the HTTP status and validated payload."""
        failure_reason: str | None = None
        status = 0
        payload: dict[str, Any] = {}
        try:
            http_response = self._http_client.post(
                self._generate_url, json=body, timeout=self._config.timeout_seconds
            )
        except httpx.TimeoutException:
            failure_reason = REASON_TIMEOUT
        except httpx.HTTPError:
            failure_reason = REASON_CONNECTION
        else:
            status = http_response.status_code
            failure_reason, payload = _read_payload(http_response)
        # Raised outside the `except` blocks so no transport exception is chained.
        if failure_reason is not None:
            logger.warning(FAILURE_LOG_MESSAGE, failure_reason)
            raise LlmUnavailableError(UNAVAILABLE_MESSAGE.format(reason=failure_reason))
        return status, payload
