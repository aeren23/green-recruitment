"""Records the wall time of every LLM call, for the per-call timing statistics."""

from typing import Protocol

from app.llm.ollama import LlmClient, LlmResponse, WarmableLlm


class WarmLlmClient(LlmClient, WarmableLlm, Protocol):
    """An LLM client that can also be loaded into memory ahead of the run."""


class TimedLlm:
    """Wraps an LLM client and keeps the duration of each generation.

    Args:
        inner: The client doing the work.
    """

    def __init__(self, inner: WarmLlmClient) -> None:
        self._inner = inner
        self.call_seconds: list[float] = []

    def generate(self, prompt: str, expect_json: bool = False) -> LlmResponse:
        """Delegate and record `duration_seconds` of the answer."""
        response = self._inner.generate(prompt, expect_json)
        self.call_seconds.append(response.duration_seconds)
        return response

    def warm_up(self) -> None:
        """Delegate the warm-up; it is not counted as a call."""
        self._inner.warm_up()
