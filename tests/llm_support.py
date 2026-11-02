"""Helpers for the LLM tests: a scripted `httpx.MockTransport`, no network."""

import json
from typing import Any

import httpx

from app.llm.ollama import OllamaClient, OllamaConfig

TEST_BASE_URL = "http://ollama.test"
TEST_MODEL = "test-model:1b"
TEST_TIMEOUT_SECONDS = 7.0
DEFAULT_PROMPT_TOKENS = 11
DEFAULT_COMPLETION_TOKENS = 5


def ollama_reply(
    text: str,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
) -> httpx.Response:
    """A successful `/api/generate` reply."""
    return httpx.Response(
        200,
        json={
            "response": text,
            "done": True,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        },
    )


def score_reply(
    score: object,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
) -> httpx.Response:
    """A reply whose text is a baseline JSON answer."""
    text = json.dumps({"score": score, "reason": "Short reason."})
    return ollama_reply(text, prompt_tokens, completion_tokens)


class ScriptedOllama:
    """Replays one response (or raised error) per request and records requests."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self._script = list(script)
        self.requests: list[httpx.Request] = []
        self.http_client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def bodies(self) -> list[dict[str, Any]]:
        """The decoded JSON body of every recorded request."""
        return [json.loads(request.content) for request in self.requests]

    def client(self, base_url: str = TEST_BASE_URL) -> OllamaClient:
        """An `OllamaClient` wired to this script."""
        config = OllamaConfig(
            base_url=base_url, model=TEST_MODEL, timeout_seconds=TEST_TIMEOUT_SECONDS
        )
        return OllamaClient(self.http_client, config)
