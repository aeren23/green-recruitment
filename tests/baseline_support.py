"""Test double for the baseline runner tests: an LLM that scores CVs from markers.

The synthetic CVs of `pipeline_support` carry a `QSCORE<percent>` marker. The
double answers every scoring prompt with that percentage as the score, so the
expected ranking is known without a model. A CV whose tag is listed in
`malformed_tags` gets an answer that is not JSON, which the scoring task turns
into a missing score after its single retry.
"""

import json
import re
from collections.abc import Collection

from bench_support import WarmFakeLlm
from pipeline_support import COMPLETION_TOKENS, LLM_SECONDS, PROMPT_TOKENS

from app.llm.ollama import LlmResponse, LlmUnavailableError

SCORE_PATTERN = re.compile(r"QSCORE(\d+)")
REASON_MARKER = "REASONMARKERTEXT"
MALFORMED_ANSWER = "this is not json"
CV_TAG_TEMPLATE = "Candidate profile {tag} "


class ScoringLlm(WarmFakeLlm):
    """A `WarmFakeLlm` that answers baseline scoring prompts with JSON.

    Args:
        malformed_tags: CV tags whose prompts get a non-JSON answer.
        fail_after: Raise `LlmUnavailableError` from the call after this many
            attempts, to simulate Ollama going away mid-run.
        unavailable: Fail the warm-up and every call.
    """

    def __init__(
        self,
        malformed_tags: Collection[str] = (),
        fail_after: int | None = None,
        unavailable: bool = False,
    ) -> None:
        super().__init__(unavailable=unavailable)
        self.malformed_tags = tuple(malformed_tags)
        self.fail_after = fail_after
        self.json_requests = 0

    def generate(self, prompt: str, expect_json: bool = False) -> LlmResponse:
        """Record the prompt, then score it from its marker or fail."""
        self.attempts += 1
        if self.unavailable or self._past_failure_point():
            raise LlmUnavailableError("Ollama is unavailable: test.")
        self.prompts.append(prompt)
        self.json_requests += int(expect_json)
        return LlmResponse(self._answer(prompt), PROMPT_TOKENS, COMPLETION_TOKENS, LLM_SECONDS)

    def _past_failure_point(self) -> bool:
        """True once more attempts were made than `fail_after` allows."""
        return self.fail_after is not None and self.attempts > self.fail_after

    def _answer(self, prompt: str) -> str:
        """JSON with the marker's percentage, or a malformed answer."""
        if any(CV_TAG_TEMPLATE.format(tag=tag) in prompt for tag in self.malformed_tags):
            return MALFORMED_ANSWER
        scores = [int(found) for found in SCORE_PATTERN.findall(prompt)]
        return json.dumps({"score": max(scores, default=0), "reason": REASON_MARKER})
