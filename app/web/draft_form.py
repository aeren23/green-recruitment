"""The candidate's preview form: its raw values and server-side validation.

`DraftFormValues` keeps every field as typed, so a form with errors can be
shown again unchanged. `build_edits` turns valid values into the pipeline's
`ProfileEdits`. Nothing here decides what a section is worth or scrubs contact
data; the pipeline does that when the edits are applied. CV-derived values are
hidden from `repr` (KVKK).
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.models import SectionType
from app.pipeline import ContactEdit, DraftPreview, ProfileEdits, SectionEdit
from app.web import messages

SECTION_TYPE_FIELD = "section_type"
SECTION_TEXT_FIELD = "section_text"
PROFILE_AND_CONTACT_FIELDS = (
    "years_experience",
    "languages",
    "location",
    "contact_name",
    "contact_email",
    "contact_phone",
)
# Every field the preview page always renders; a post lacking one is malformed.
STRUCTURAL_FIELDS = (SECTION_TYPE_FIELD, SECTION_TEXT_FIELD, *PROFILE_AND_CONTACT_FIELDS)
DEFAULT_SECTION_TYPE = SectionType.EXPERIENCE
BLANK_SECTION_ROWS = 2
MAX_SECTION_ROWS = 40
SECTION_TEXT_MAX_LENGTH = 20000
SHORT_FIELD_MAX_LENGTH = 200
YEARS_LIMIT = 60.0
LIST_SEPARATOR = ","
DECIMAL_COMMA = ","
DECIMAL_POINT = "."
JOIN_SEPARATOR = ", "


@dataclass(frozen=True)
class SectionRow:
    """One section row as typed: its type value and its text."""

    type: str = DEFAULT_SECTION_TYPE.value
    text: str = field(default="", repr=False)


@dataclass(frozen=True)
class DraftFormValues:
    """All preview-form fields as typed (raw strings)."""

    rows: tuple[SectionRow, ...] = field(default=(), repr=False)
    years: str = field(default="", repr=False)
    languages: str = field(default="", repr=False)
    location: str = field(default="", repr=False)
    contact_name: str = field(default="", repr=False)
    contact_email: str = field(default="", repr=False)
    contact_phone: str = field(default="", repr=False)

    def with_blank_rows(self) -> "DraftFormValues":
        """The same values followed by empty rows for adding sections."""
        blanks = (SectionRow(),) * BLANK_SECTION_ROWS
        return DraftFormValues(
            self.rows + blanks,
            self.years,
            self.languages,
            self.location,
            self.contact_name,
            self.contact_email,
            self.contact_phone,
        )


@dataclass(frozen=True)
class EditsResult:
    """Either the edits to apply or the Turkish errors to show."""

    edits: ProfileEdits | None
    errors: tuple[str, ...]


def values_from_preview(preview: DraftPreview) -> DraftFormValues:
    """The form values that show a draft's parsed content."""
    languages = (messages.LANGUAGE_LABELS.get(name, name) for name in preview.languages)
    contact = preview.contact
    return DraftFormValues(
        rows=tuple(SectionRow(item.type.value, item.text) for item in preview.sections),
        years=format_years(preview.years_experience),
        languages=JOIN_SEPARATOR.join(languages),
        location=preview.location or "",
        contact_name=contact.name or "",
        contact_email=contact.email or "",
        contact_phone=contact.phone or "",
    )


def format_years(years: float | None) -> str:
    """Years as shown in the form: a Turkish decimal comma, blank when unknown."""
    if years is None:
        return ""
    return f"{years:g}".replace(DECIMAL_POINT, DECIMAL_COMMA)


def pair_section_rows(
    types: Sequence[str], texts: Sequence[str]
) -> tuple[SectionRow, ...]:
    """Pair the repeated type and text fields row by row."""
    padded_types = list(types) + [DEFAULT_SECTION_TYPE.value] * (len(texts) - len(types))
    return tuple(
        SectionRow(row_type, text)
        for row_type, text in zip(padded_types[: len(texts)], texts, strict=True)
    )


def is_well_formed(field_names: Iterable[str]) -> bool:
    """Whether a posted form carries every field the preview always renders.

    A truncated or empty post lacks them; applying it would silently wipe the
    candidate's profile, so the caller must refuse it and change nothing.
    """
    present = set(field_names)
    return all(name in present for name in STRUCTURAL_FIELDS)


def build_edits(values: DraftFormValues) -> EditsResult:
    """Validate the form and build the candidate's corrections.

    Args:
        values: The raw form values.

    Returns:
        The `ProfileEdits`, or the error messages when a field is invalid.
    """
    sections, section_errors = parse_sections(values.rows)
    years, years_errors = parse_years(values.years)
    errors = section_errors + years_errors + check_short_fields(values)
    if errors:
        return EditsResult(None, tuple(errors))
    edits = ProfileEdits(
        sections=sections,
        years_experience=years,
        languages=split_list(values.languages),
        location=blank_to_none(values.location),
        contact=ContactEdit(
            blank_to_none(values.contact_name),
            blank_to_none(values.contact_email),
            blank_to_none(values.contact_phone),
        ),
    )
    return EditsResult(edits, ())


def parse_sections(
    rows: Sequence[SectionRow],
) -> tuple[list[SectionEdit], list[str]]:
    """Section edits from the rows, and the errors of all rows."""
    if len(rows) > MAX_SECTION_ROWS:
        return [], [messages.TOO_MANY_SECTIONS.format(limit=MAX_SECTION_ROWS)]
    sections: list[SectionEdit] = []
    errors: list[str] = []
    if not any(row.text.strip() for row in rows):
        return [], [messages.SECTION_REQUIRED]
    for number, row in enumerate(rows, start=1):
        row_errors = check_row(row, number)
        errors += row_errors
        if not row_errors:
            sections.append(SectionEdit(SectionType(row.type), row.text))
    return sections, errors


def check_row(row: SectionRow, number: int) -> list[str]:
    """Errors of one section row: an unknown type or a text that is too long."""
    errors: list[str] = []
    if row.type not in {item.value for item in SectionType}:
        errors.append(messages.SECTION_TYPE_INVALID.format(row=number))
    if len(row.text) > SECTION_TEXT_MAX_LENGTH:
        errors.append(
            messages.SECTION_TOO_LONG.format(row=number, limit=SECTION_TEXT_MAX_LENGTH)
        )
    return errors


def parse_years(raw_value: str) -> tuple[float | None, list[str]]:
    """Blank means unknown; otherwise a finite number from 0 to the limit."""
    stripped = raw_value.strip()
    if not stripped:
        return None, []
    try:
        years = float(stripped.replace(DECIMAL_COMMA, DECIMAL_POINT))
    except ValueError:
        years = math.nan
    if math.isfinite(years) and 0 <= years <= YEARS_LIMIT:
        return years, []
    return None, [messages.YEARS_INVALID.format(limit=int(YEARS_LIMIT))]


def check_short_fields(values: DraftFormValues) -> list[str]:
    """Length errors of the single-line fields."""
    labelled = (
        (messages.FIELD_LANGUAGES, values.languages),
        (messages.FIELD_LOCATION, values.location),
        (messages.FIELD_CONTACT_NAME, values.contact_name),
        (messages.FIELD_CONTACT_EMAIL, values.contact_email),
        (messages.FIELD_CONTACT_PHONE, values.contact_phone),
    )
    return [
        messages.FIELD_TOO_LONG.format(field=label, limit=SHORT_FIELD_MAX_LENGTH)
        for label, text in labelled
        if len(text) > SHORT_FIELD_MAX_LENGTH
    ]


def split_list(raw_value: str) -> list[str]:
    """Comma-separated entries, stripped, without blanks."""
    parts = (part.strip() for part in raw_value.split(LIST_SEPARATOR))
    return [part for part in parts if part]


def blank_to_none(raw_value: str) -> str | None:
    """A stripped value, `None` when nothing is left (never an empty string)."""
    return raw_value.strip() or None
