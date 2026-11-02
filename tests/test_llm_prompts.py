"""Prompt builders: truncation, selected-sections-only, Turkish summary request."""

import pytest

from app.llm.prompts import (
    MAX_CV_CHARS,
    BaselinePromptInput,
    CvSection,
    SummaryPromptInput,
    build_baseline_prompt,
    build_summary_prompt,
)

REQUIREMENTS = ("Building REST APIs", "Relational databases and SQL")
SELECTED_MARKER = "SELECTED-EVIDENCE-MARKER"
UNSELECTED_MARKER = "UNSELECTED-EDUCATION-MARKER"
OVERLONG_EXTRA = 50
EXPERIENCE_ID = 1
EDUCATION_ID = 2
PROJECT_ID = 3

CANDIDATE_SECTIONS = (
    CvSection(EXPERIENCE_ID, "experience", f"Built APIs. {SELECTED_MARKER}"),
    CvSection(EDUCATION_ID, "education", f"BSc. {UNSELECTED_MARKER}"),
    CvSection(PROJECT_ID, "project", "Designed a database."),
)


def summary_input(selected_ids: set[int]) -> SummaryPromptInput:
    """A summary input over the full candidate with the given selection."""
    return SummaryPromptInput(REQUIREMENTS, CANDIDATE_SECTIONS, selected_ids)


def test_baseline_prompt_lists_requirements_and_asks_for_json() -> None:
    """Requirements appear, and the JSON contract is spelled out."""
    prompt = build_baseline_prompt(BaselinePromptInput(REQUIREMENTS, "cv body"))
    for requirement in REQUIREMENTS:
        assert requirement in prompt
    assert "cv body" in prompt
    assert '"score"' in prompt
    assert '"reason"' in prompt
    assert "0 to 100" in prompt


def test_baseline_cv_is_cut_at_max_cv_chars() -> None:
    """Exactly the first MAX_CV_CHARS characters of the CV are included."""
    kept = "a" * MAX_CV_CHARS
    cut = "b" * OVERLONG_EXTRA
    prompt = build_baseline_prompt(BaselinePromptInput(REQUIREMENTS, kept + cut))
    assert kept in prompt
    assert "b" not in prompt.split("CV:")[1]


def test_baseline_cv_at_the_limit_is_kept_whole() -> None:
    """A CV of exactly MAX_CV_CHARS characters is not shortened."""
    cv_text = "c" * MAX_CV_CHARS
    prompt = build_baseline_prompt(BaselinePromptInput(REQUIREMENTS, cv_text))
    assert cv_text in prompt


def test_baseline_truncation_limit_is_injectable() -> None:
    """A smaller limit (for example from settings) is honoured."""
    prompt_input = BaselinePromptInput(REQUIREMENTS, "0123456789", max_cv_chars=4)
    prompt = build_baseline_prompt(prompt_input)
    assert "0123" in prompt
    assert "01234" not in prompt


def test_cv_braces_do_not_break_the_template() -> None:
    """CV text containing format braces is inserted verbatim."""
    prompt = build_baseline_prompt(BaselinePromptInput(REQUIREMENTS, "{x} {0} {{y}}"))
    assert "{x} {0} {{y}}" in prompt


def test_summary_prompt_contains_selected_sections_only() -> None:
    """The unselected marker never reaches the prompt; the selected one does."""
    prompt = build_summary_prompt(summary_input({EXPERIENCE_ID, PROJECT_ID}))
    assert SELECTED_MARKER in prompt
    assert "Designed a database." in prompt
    assert "[experience]" in prompt
    assert UNSELECTED_MARKER not in prompt
    assert "[education]" not in prompt


def test_summary_prompt_ignores_selected_ids_that_match_no_section() -> None:
    """Unknown ids are skipped; known selected sections are still used."""
    prompt = build_summary_prompt(summary_input({PROJECT_ID, 999}))
    assert "Designed a database." in prompt
    assert SELECTED_MARKER not in prompt


def test_summary_prompt_asks_for_two_turkish_sentences() -> None:
    """The output language and length limit are part of the prompt."""
    prompt = build_summary_prompt(summary_input({PROJECT_ID}))
    assert "Turkish" in prompt
    assert "at most 2 sentences" in prompt


@pytest.mark.parametrize("requirements", [(), []])
def test_prompts_reject_empty_requirements(requirements: list[str]) -> None:
    """Both builders fail fast without requirements."""
    with pytest.raises(ValueError):
        build_baseline_prompt(BaselinePromptInput(requirements, "cv"))
    with pytest.raises(ValueError):
        build_summary_prompt(
            SummaryPromptInput(requirements, CANDIDATE_SECTIONS, {PROJECT_ID})
        )


@pytest.mark.parametrize("selected_ids", [set(), {999}])
def test_summary_prompt_rejects_an_empty_selection(selected_ids: set[int]) -> None:
    """A summary without selected evidence text is a caller bug."""
    with pytest.raises(ValueError):
        build_summary_prompt(summary_input(selected_ids))
