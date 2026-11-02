"""Prompt templates and builders for the two LLM tasks (SPEC 6.6 and 7.4).

The baseline prompt carries the CV text (truncated); the green summary prompt
carries only the sections selected in the matrix, never the whole CV.
Templates are named constants; inputs are dataclasses.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

# TODO(config): should come from `Settings.max_cv_chars` (SPEC 10) later.
MAX_CV_CHARS = 6000

REQUIREMENT_LINE = "- {requirement}"
SECTION_BLOCK = "[{section_type}]\n{text}"
LINE_SEPARATOR = "\n"
BLOCK_SEPARATOR = "\n\n"

BASELINE_PROMPT_TEMPLATE = (
    "You are screening a job application. Rate how well the CV below matches "
    "the job requirements.\n"
    'Answer with one JSON object only: {{"score": <integer from 0 to 100>, '
    '"reason": "<one sentence>"}}.\n\n'
    "Job requirements:\n{requirements}\n\n"
    "CV:\n{cv_text}\n"
)
SUMMARY_PROMPT_TEMPLATE = (
    "You are screening a job application. Using only the CV excerpts below, "
    "write at most 2 sentences in Turkish that tell how well the candidate "
    "meets the job requirements. Do not invent facts. Do not mention names "
    "or contact details.\n\n"
    "Job requirements:\n{requirements}\n\n"
    "CV excerpts:\n{sections}\n"
)

EMPTY_REQUIREMENTS_MESSAGE = "A prompt needs at least one requirement."
EMPTY_SECTIONS_MESSAGE = "None of the candidate's sections is selected for the summary."


@dataclass(frozen=True)
class BaselinePromptInput:
    """Inputs of the baseline scoring prompt.

    Attributes:
        requirements: Requirement texts of the job posting.
        cv_text: The full CV text; it is cut to `max_cv_chars` characters.
        max_cv_chars: Truncation limit of the CV text.
    """

    requirements: Sequence[str]
    cv_text: str
    max_cv_chars: int = MAX_CV_CHARS


@dataclass(frozen=True)
class CvSection:
    """One CV section of a candidate, as stored in the `sections` table."""

    section_id: int
    section_type: str
    text: str


@dataclass(frozen=True)
class SummaryPromptInput:
    """Inputs of the green summary prompt.

    Attributes:
        requirements: Requirement texts of the job posting.
        sections: The candidate's sections.
        selected_section_ids: Ids of the sections the matrix selected as
            evidence (SPEC 6.6). Only these reach the prompt; the builder
            enforces it, so a caller cannot leak the rest of the CV.
    """

    requirements: Sequence[str]
    sections: Sequence[CvSection]
    selected_section_ids: Collection[int]


def _format_requirements(requirements: Sequence[str]) -> str:
    """Render requirements as a bullet list; fail fast when there are none."""
    if not requirements:
        raise ValueError(EMPTY_REQUIREMENTS_MESSAGE)
    return LINE_SEPARATOR.join(
        REQUIREMENT_LINE.format(requirement=requirement) for requirement in requirements
    )


def build_baseline_prompt(prompt_input: BaselinePromptInput) -> str:
    """Build the baseline scoring prompt.

    Args:
        prompt_input: Requirements and the CV text.

    Returns:
        The prompt, with the CV text cut to `max_cv_chars` characters.

    Raises:
        ValueError: If there are no requirements.
    """
    return BASELINE_PROMPT_TEMPLATE.format(
        requirements=_format_requirements(prompt_input.requirements),
        cv_text=prompt_input.cv_text[: prompt_input.max_cv_chars],
    )


def build_summary_prompt(prompt_input: SummaryPromptInput) -> str:
    """Build the green summary prompt from the selected sections only.

    Args:
        prompt_input: Requirements, the candidate's sections and the ids
            of the matrix-selected ones.

    Returns:
        The prompt; it asks for at most 2 sentences in Turkish.

    Raises:
        ValueError: If there are no requirements or none of the sections is selected.
    """
    requirements = _format_requirements(prompt_input.requirements)
    selected = [
        section
        for section in prompt_input.sections
        if section.section_id in prompt_input.selected_section_ids
    ]
    if not selected:
        raise ValueError(EMPTY_SECTIONS_MESSAGE)
    sections = BLOCK_SEPARATOR.join(
        SECTION_BLOCK.format(section_type=section.section_type, text=section.text)
        for section in selected
    )
    return SUMMARY_PROMPT_TEMPLATE.format(requirements=requirements, sections=sections)
