"""Test doubles and builders for the pipeline tests: no GPU, no Ollama, no model.

* `ControlledEmbedder` lets a test choose how well a CV matches every
  requirement by writing `QSCORE<percent>` into a section text.
* `RecordingTracker` records stage names in order and fails on nesting.
* `FakeLlm` records the prompts it receives and can be switched to "down".
* `build_cv_pdf` builds a synthetic CV as a real PDF with contact details.
"""

import re
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

import numpy as np
from pdf_fixtures import build_pdf

from app.green.tracker import NestedStageError, StageTotals, require_known_stage
from app.llm.ollama import LlmResponse, LlmUnavailableError
from app.scoring.embed import FloatMatrix

EMBEDDING_DIMENSION = 8
CONTROLLED_MODEL_VERSION = "controlled-embedder-v1"
QUALITY_PATTERN = re.compile(r"QSCORE(\d+)")
PERCENT = 100.0
UNMATCHED_QUALITY = 0.0

CANDIDATE_NAME = "Jane Doe"
CANDIDATE_EMAIL = "jane.doe@example.com"
CANDIDATE_PHONE = "+90 532 123 45 67"
EXPERIENCE_MARKER = "EXPERIENCETEXTMARKER"
UNSELECTED_MARKER = "UNSELECTEDEDUCATIONMARKER"
DEFAULT_QUALITY = 50
DEFAULT_YEARS = (2016, 2024)
SUMMARY_REPLY = "Aday gereksinimleri karsiliyor."
PROMPT_TOKENS = 10
COMPLETION_TOKENS = 5
LLM_SECONDS = 0.01


class ControlledEmbedder:
    """`EmbeddingProvider` whose similarities come from `QSCORE<percent>` markers.

    Every query maps to the first unit vector. A passage with a marker maps to
    a unit vector whose cosine similarity to every query is the marker's
    percentage; a passage without a marker is orthogonal to every query.
    """

    model_version = CONTROLLED_MODEL_VERSION

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    @property
    def passage_texts_embedded(self) -> int:
        """Total number of passages embedded over all calls."""
        return sum(len(call) for call in self.passage_calls)

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Record the call; every query is the first axis."""
        self.query_calls.append(list(texts))
        rows = [self._unit_vector(1.0) for _ in texts]
        return np.stack(rows) if rows else np.empty((0, EMBEDDING_DIMENSION), np.float32)

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Record the call; each passage gets the similarity its marker names."""
        self.passage_calls.append(list(texts))
        rows = [self._unit_vector(self._quality_of(text)) for text in texts]
        return np.stack(rows) if rows else np.empty((0, EMBEDDING_DIMENSION), np.float32)

    @staticmethod
    def _quality_of(text: str) -> float:
        """The similarity named by the highest marker in the text, else 0."""
        percentages = [int(found) for found in QUALITY_PATTERN.findall(text)]
        return max(percentages) / PERCENT if percentages else UNMATCHED_QUALITY

    @staticmethod
    def _unit_vector(similarity: float) -> FloatMatrix:
        """A unit vector whose inner product with the first axis is `similarity`."""
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        vector[0] = similarity
        vector[1] = float(np.sqrt(max(0.0, 1.0 - similarity**2)))
        return vector


class RecordingTracker:
    """A `StageTracker` that records stage names and rejects nested stages."""

    def __init__(self) -> None:
        self.stage_names: list[str] = []
        self._active: str | None = None

    @property
    def active_stage(self) -> str | None:
        """The name of the stage that is running right now, if any."""
        return self._active

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Validate the name, then record it when the stage is entered."""
        require_known_stage(name)
        return self._recorded(name)

    @contextmanager
    def _recorded(self, name: str) -> Iterator[None]:
        """Enter one stage; raise `NestedStageError` if another is running."""
        if self._active is not None:
            raise NestedStageError(f"{name} opened inside {self._active}")
        self._active = name
        self.stage_names.append(name)
        try:
            yield
        finally:
            self._active = None

    def totals(self) -> dict[str, StageTotals]:
        """Nothing is measured."""
        return {}

    def measurement_notes(self) -> str:
        """Nothing is measured."""
        return "recording tracker"

    def clear(self) -> None:
        """Forget the recorded names."""
        self.stage_names.clear()


class FakeLlm:
    """A scripted `LlmClient` that records every prompt it receives.

    Args:
        reply: The text every successful call returns.
        unavailable: When true, every call raises `LlmUnavailableError`.
    """

    def __init__(self, reply: str = SUMMARY_REPLY, unavailable: bool = False) -> None:
        self.reply = reply
        self.unavailable = unavailable
        self.prompts: list[str] = []
        self.attempts = 0

    def generate(self, prompt: str, expect_json: bool = False) -> LlmResponse:
        """Record the prompt and answer, or raise when switched to unavailable."""
        self.attempts += 1
        if self.unavailable:
            raise LlmUnavailableError("Ollama is unavailable: test.")
        self.prompts.append(prompt)
        return LlmResponse(self.reply, PROMPT_TOKENS, COMPLETION_TOKENS, LLM_SECONDS)


@dataclass(frozen=True)
class CvSpec:
    """What a synthetic CV looks like.

    Attributes:
        tag: Any word; it appears in the summary and makes the hash unique.
        quality: Percentage written into the experience (or skills) section.
        marker_in_skills: Put the quality marker in skills instead of experience.
        years: Start and end year of the single experience entry.
    """

    tag: str
    quality: int = DEFAULT_QUALITY
    marker_in_skills: bool = False
    years: tuple[int, int] = DEFAULT_YEARS


def build_cv_pdf(
    tag: str, quality: int = DEFAULT_QUALITY, years: tuple[int, int] = DEFAULT_YEARS
) -> bytes:
    """A synthetic CV whose experience section matches every requirement at `quality`."""
    return build_cv_from_spec(CvSpec(tag, quality, years=years))


def build_claim_cv_pdf(tag: str, quality: int = DEFAULT_QUALITY) -> bytes:
    """A synthetic CV whose match sits in the skills section (a claim, not evidence)."""
    return build_cv_from_spec(CvSpec(tag, quality, marker_in_skills=True))


def build_cv_from_spec(spec: CvSpec) -> bytes:
    """Build the CV described by the spec as PDF bytes."""
    tag, quality = spec.tag, spec.quality
    start, end = spec.years
    experience_marker = "" if spec.marker_in_skills else f"QSCORE{quality}"
    skills_marker = f"QSCORE{quality}" if spec.marker_in_skills else ""
    lines = [
        CANDIDATE_NAME,
        f"{CANDIDATE_EMAIL} {CANDIDATE_PHONE}",
        "SUMMARY",
        f"Candidate profile {tag} seeking backend engineering roles.",
        "EXPERIENCE",
        f"Software Engineer, Example Corp, {start} - {end}",
        f"Built services and databases {EXPERIENCE_MARKER} {experience_marker}",
        "SKILLS",
        f"Python SQL Docker {skills_marker}",
        "EDUCATION",
        f"BSc Computer Engineering, Example University, 2010 - 2014 {UNSELECTED_MARKER}",
    ]
    return build_pdf(lines)
