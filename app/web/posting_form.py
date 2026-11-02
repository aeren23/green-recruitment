"""The employer's "new posting" form: its values and server-side validation.

`PostingFormValues` keeps the raw strings exactly as typed, so a form with
errors can be shown again with the entered values. `build_posting_input` turns
valid values into a `PostingInput` for `app.postings`, or returns Turkish
error messages. Blank requirement rows are ignored.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.models import REQUIREMENT_WEIGHT_MAX, REQUIREMENT_WEIGHT_MIN
from app.postings import PostingInput, RequirementInput
from app.web import messages

REQUIREMENT_ROW_COUNT = 6
DEFAULT_WEIGHT = "2"
TITLE_MAX_LENGTH = 120
DESCRIPTION_MAX_LENGTH = 4000
LANGUAGES_MAX_LENGTH = 200
LOCATION_MAX_LENGTH = 80
REQUIREMENT_TEXT_MAX_LENGTH = 200
MIN_YEARS_LIMIT = 60.0
LANGUAGE_SEPARATOR = ","
DECIMAL_COMMA = ","
DECIMAL_POINT = "."


@dataclass(frozen=True)
class RequirementRow:
    """One requirement row as typed: text and the weight as a string."""

    text: str = ""
    weight: str = DEFAULT_WEIGHT


@dataclass(frozen=True)
class PostingFormValues:
    """All form fields as typed (raw strings), padded to a fixed row count."""

    title: str = ""
    description: str = ""
    languages: str = ""
    location: str = ""
    remote_ok: bool = False
    min_years: str = ""
    rows: tuple[RequirementRow, ...] = ()

    def with_padded_rows(self) -> "PostingFormValues":
        """The same values with exactly `REQUIREMENT_ROW_COUNT` rows."""
        kept = self.rows[:REQUIREMENT_ROW_COUNT]
        blanks = (RequirementRow(),) * (REQUIREMENT_ROW_COUNT - len(kept))
        return PostingFormValues(
            self.title,
            self.description,
            self.languages,
            self.location,
            self.remote_ok,
            self.min_years,
            kept + blanks,
        )


@dataclass(frozen=True)
class FormResult:
    """Either a posting to create or the Turkish errors to show."""

    posting_input: PostingInput | None
    errors: tuple[str, ...]


def pair_rows(texts: Sequence[str], weights: Sequence[str]) -> tuple[RequirementRow, ...]:
    """Pair the repeated `text` and `weight` form fields row by row."""
    padded_weights = list(weights) + [DEFAULT_WEIGHT] * (len(texts) - len(weights))
    return tuple(
        RequirementRow(text, weight)
        for text, weight in zip(texts, padded_weights[: len(texts)], strict=True)
    )


def build_posting_input(values: PostingFormValues, now: datetime) -> FormResult:
    """Validate the form and build the posting to create.

    Args:
        values: The raw form values.
        now: The injected clock, stored as the posting's creation time.

    Returns:
        A result holding either the `PostingInput` or the error messages.
    """
    errors = _check_text_fields(values)
    min_years, years_errors = _parse_min_years(values.min_years)
    requirements, requirement_errors = _parse_requirements(values.rows)
    errors = errors + years_errors + requirement_errors
    if errors:
        return FormResult(None, tuple(errors))
    return FormResult(_assemble(values, requirements, min_years, now), ())


def _assemble(
    values: PostingFormValues,
    requirements: list[RequirementInput],
    min_years: float | None,
    now: datetime,
) -> PostingInput:
    """The `PostingInput` for values that passed every check."""
    languages = [part.strip() for part in values.languages.split(LANGUAGE_SEPARATOR)]
    return PostingInput(
        title=values.title,
        description=values.description,
        requirements=requirements,
        created_at=now,
        required_languages=[language for language in languages if language],
        location=values.location,
        remote_ok=values.remote_ok,
        min_years=min_years,
    )


def _check_text_fields(values: PostingFormValues) -> list[str]:
    """Errors of the title, description, languages and location fields."""
    errors: list[str] = []
    if not values.title.strip():
        errors.append(messages.TITLE_REQUIRED)
    errors += _too_long(values.title, TITLE_MAX_LENGTH, messages.TITLE_TOO_LONG)
    errors += _too_long(
        values.description, DESCRIPTION_MAX_LENGTH, messages.DESCRIPTION_TOO_LONG
    )
    errors += _too_long(values.languages, LANGUAGES_MAX_LENGTH, messages.LANGUAGES_TOO_LONG)
    errors += _too_long(values.location, LOCATION_MAX_LENGTH, messages.LOCATION_TOO_LONG)
    return errors


def _too_long(text: str, limit: int, message: str) -> list[str]:
    """The message (with its limit filled in) when the text exceeds the limit."""
    return [message.format(limit=limit)] if len(text) > limit else []


def _parse_min_years(raw_value: str) -> tuple[float | None, list[str]]:
    """Blank means "no minimum"; otherwise a finite number from 0 to the limit."""
    stripped = raw_value.strip()
    if not stripped:
        return None, []
    try:
        years = float(stripped.replace(DECIMAL_COMMA, DECIMAL_POINT))
    except ValueError:
        years = math.nan
    if math.isfinite(years) and 0 <= years <= MIN_YEARS_LIMIT:
        return years, []
    return None, [messages.MIN_YEARS_INVALID.format(limit=int(MIN_YEARS_LIMIT))]


def _parse_requirements(
    rows: Sequence[RequirementRow],
) -> tuple[list[RequirementInput], list[str]]:
    """Requirements of the non-blank rows, and the errors of all rows."""
    requirements: list[RequirementInput] = []
    errors: list[str] = []
    for row_number, row in enumerate(rows[:REQUIREMENT_ROW_COUNT], start=1):
        if not row.text.strip():
            continue
        row_errors = _check_row(row, row_number)
        errors += row_errors
        if not row_errors:
            requirements.append(RequirementInput(row.text, int(row.weight)))
    if not requirements and not errors:
        errors.append(messages.REQUIREMENTS_REQUIRED)
    return requirements, errors


def _check_row(row: RequirementRow, row_number: int) -> list[str]:
    """Errors of one non-blank requirement row (length and weight)."""
    errors: list[str] = []
    if len(row.text) > REQUIREMENT_TEXT_MAX_LENGTH:
        errors.append(
            messages.REQUIREMENT_TOO_LONG.format(
                row=row_number, limit=REQUIREMENT_TEXT_MAX_LENGTH
            )
        )
    if not _weight_in_range(row.weight):
        errors.append(
            messages.REQUIREMENT_WEIGHT_INVALID.format(
                row=row_number, low=REQUIREMENT_WEIGHT_MIN, high=REQUIREMENT_WEIGHT_MAX
            )
        )
    return errors


def _weight_in_range(raw_weight: str) -> bool:
    """True when the weight is an integer from 1 to 3."""
    try:
        weight = int(raw_weight)
    except ValueError:
        return False
    return REQUIREMENT_WEIGHT_MIN <= weight <= REQUIREMENT_WEIGHT_MAX
