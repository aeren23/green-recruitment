"""Accumulator for LLM counters (`llm_calls`, `llm_prompt_tokens`, `llm_completion_tokens`)."""

from dataclasses import dataclass

from app.llm.ollama import LlmResponse


@dataclass
class LlmUsage:
    """Counts what a pipeline asked of the LLM.

    Attributes:
        calls: Completed generations, malformed answers and retries included.
        prompt_tokens: Sum of `prompt_eval_count`.
        completion_tokens: Sum of `eval_count`.
        failures: Tasks that ended without a usable result (a baseline score
            that stayed malformed after the retry, or an unreachable Ollama
            during a summary).
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    def record_response(self, response: LlmResponse) -> None:
        """Add one completed generation, whether or not its content was usable."""
        self.calls += 1
        self.prompt_tokens += response.prompt_eval_count
        self.completion_tokens += response.eval_count

    def record_failure(self) -> None:
        """Count one task that produced no usable result."""
        self.failures += 1
