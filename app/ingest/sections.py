"""Rule-based section splitting and profile extraction (SPEC 6.3, no LLM).

The input is the contact-scrubbed text of `extract_text`; it is not normalized,
because heading and entry detection depend on the line structure. Nothing here
logs or raises with CV text (KVKK).
"""

from dataclasses import dataclass, field
from datetime import date

from app.ingest.dates import find_date_ranges, total_years
from app.ingest.entries import chunk_words, split_entries
from app.ingest.headings import HeadingTarget, classify_heading, fold_heading
from app.ingest.languages import extract_languages
from app.ingest.text_lines import BLANK_LINE, LINE_SEPARATOR, clean_lines, is_noise_line
from app.models import SectionType

__all__ = ["ParsedCv", "ParsedSection", "fold_heading", "parse_cv"]

PARSER_VERSION = "1"
ENTRY_TARGETS = frozenset({HeadingTarget.EXPERIENCE, HeadingTarget.PROJECT})
NOT_A_TEXT_MESSAGE = "parse_cv expects the extracted text as a str."


@dataclass(frozen=True)
class ParsedSection:
    """One embeddable piece of a CV; its text is hidden from `repr` (KVKK)."""

    type: SectionType
    ordinal: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class ParsedCv:
    """The result of parsing one CV.

    Attributes:
        sections: Embeddable sections in reading order.
        years_experience: Merged experience in years, None without a range.
        languages: Spoken languages, in English.
        no_headings_found: True when the text had no heading at all and was
            kept as a single summary; the candidate sees a warning (SPEC 6.3).
        profile_json: The `profile_json` structure of SPEC 6.3.
    """

    sections: list[ParsedSection]
    years_experience: float | None
    languages: list[str]
    no_headings_found: bool
    profile_json: dict = field(repr=False)


@dataclass
class Segment:
    """The lines under one heading (or before the first one)."""

    target: HeadingTarget
    lines: list[str] = field(default_factory=list)


def segment_lines(lines: list[str]) -> tuple[list[Segment], bool]:
    """Group lines under their headings.

    Returns:
        The segments (the first holds any text before the first heading, as
        summary) and whether at least one heading was found.
    """
    segments = [Segment(HeadingTarget.SUMMARY)]
    headings_found = False
    for line in lines:
        target = classify_heading(line)
        if target is None:
            segments[-1].lines.append(line)
        else:
            headings_found = True
            segments.append(Segment(target))
    return segments, headings_found


def entries_of(segment: Segment) -> list[str]:
    """Return the entry texts of a segment (one text unless it is an entry type)."""
    if segment.target in ENTRY_TARGETS:
        return split_entries(LINE_SEPARATOR.join(segment.lines))
    kept = (line for line in segment.lines if line != BLANK_LINE and not is_noise_line(line))
    text = LINE_SEPARATOR.join(kept)
    return [text] if text.strip() else []


def years_from_entries(entries: list[str], reference_date: date) -> float | None:
    """Total experience over the date ranges of experience entries only."""
    ranges = [found for entry in entries for found in find_date_ranges(entry)]
    return total_years(ranges, reference_date)


def build_sections(typed_entries: list[tuple[SectionType, str]]) -> list[ParsedSection]:
    """Chunk entries to the word limit and number the pieces in order."""
    pieces = [
        (section_type, chunk)
        for section_type, entry in typed_entries
        for chunk in chunk_words(entry)
    ]
    return [
        ParsedSection(type=section_type, ordinal=ordinal, text=chunk)
        for ordinal, (section_type, chunk) in enumerate(pieces)
    ]


def build_profile_json(
    sections: list[ParsedSection], years: float | None, languages: list[str]
) -> dict:
    """Assemble the `profile_json` dictionary (location is never parsed)."""
    return {
        "parser_version": PARSER_VERSION,
        "sections": [
            {"type": section.type.value, "ordinal": section.ordinal, "text": section.text}
            for section in sections
        ],
        "years_experience": years,
        "languages": languages,
        "location": None,
    }


@dataclass
class SegmentContent:
    """What the segments of a CV yield before chunking."""

    typed_entries: list[tuple[SectionType, str]] = field(default_factory=list)
    experience_entries: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)


def collect_content(segments: list[Segment]) -> SegmentContent:
    """Turn segments into entries and languages; ignored headings drop their text."""
    content = SegmentContent()
    for segment in segments:
        if segment.target is HeadingTarget.IGNORED:
            continue
        if segment.target is HeadingTarget.LANGUAGES:
            known = extract_languages(segment.lines)
            content.languages += [name for name in known if name not in content.languages]
            continue
        entries = entries_of(segment)
        if segment.target is HeadingTarget.EXPERIENCE:
            content.experience_entries += entries
        section_type = SectionType(segment.target.value)
        content.typed_entries += [(section_type, entry) for entry in entries]
    return content


def parse_cv(text: str, reference_date: date) -> ParsedCv:
    """Split a contact-scrubbed CV into sections and profile fields.

    Args:
        text: Output of `extract_text` with contact details already removed.
        reference_date: Date that `Current` ends and future dates are clipped
            to; the web layer passes today, the benchmark a fixed date.

    Returns:
        The parsed CV. Never raises on odd content; a CV without headings
        becomes one summary with `no_headings_found` set.

    Raises:
        TypeError: If `text` is not a string (the message carries no content).
    """
    if not isinstance(text, str):
        raise TypeError(NOT_A_TEXT_MESSAGE)
    segments, headings_found = segment_lines(clean_lines(text))
    content = collect_content(segments)
    sections = build_sections(content.typed_entries)
    years = years_from_entries(content.experience_entries, reference_date)
    return ParsedCv(
        sections=sections,
        years_experience=years,
        languages=content.languages,
        no_headings_found=not headings_found,
        profile_json=build_profile_json(sections, years, content.languages),
    )
