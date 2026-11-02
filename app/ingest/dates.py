"""Date range detection and experience arithmetic (SPEC 6.3).

Ranges are found across line breaks because pypdf puts the start date, the
`to` separator and the end date on separate lines. All arithmetic is in whole
months counted from year 0, so it never depends on the current clock: the
caller passes the `reference_date`.
"""

import re
from dataclasses import dataclass
from datetime import date

MONTHS_PER_YEAR = 12
YEARS_DECIMAL_PLACES = 1

MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
MONTH_ABBREVIATION_LENGTH = 3
MONTH_NAME_ALTERNATIVES = "|".join(
    sorted(
        {name for name in MONTH_NAMES}
        | {name[:MONTH_ABBREVIATION_LENGTH] for name in MONTH_NAMES}
        | {"sept"},
        key=len,
        reverse=True,
    )
)
OPEN_END_WORDS = ("present", "current", "now", "günümüz")
OPEN_END_ALTERNATIVES = "|".join(OPEN_END_WORDS)
RANGE_SEPARATORS = r"[-\u2013\u2014]|\b(?:to|through)\b"


def date_pattern(group_prefix: str) -> str:
    """Return a `YYYY`, `MM/YYYY` or `Month YYYY` pattern with named groups."""
    return (
        r"(?<![\d/])(?:"
        rf"\b(?P<{group_prefix}_month_name>{MONTH_NAME_ALTERNATIVES})\.?[ \t]+"
        rf"|(?P<{group_prefix}_month_number>0?[1-9]|1[0-2])[ \t]*/[ \t]*"
        r")?"
        rf"(?P<{group_prefix}_year>(?:19|20)\d{{2}})(?!\d)"
    )


DATE_RANGE_PATTERN = re.compile(
    rf"{date_pattern('start')}\s*(?:{RANGE_SEPARATORS})\s*"
    rf"(?:{date_pattern('end')}|\b(?P<open_end>{OPEN_END_ALTERNATIVES})\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DateRange:
    """A date range found in text.

    Attributes:
        start_offset: Index of the first character of the range in the text.
        end_offset: Index just after the last character of the range.
        start_month: Absolute month number (`year * 12 + month - 1`).
        end_month: Absolute month number, or None for an open end.
    """

    start_offset: int
    end_offset: int
    start_month: int
    end_month: int | None


def absolute_month(year: int, month: int) -> int:
    """Return the absolute month number for a 1-based calendar month."""
    return year * MONTHS_PER_YEAR + month - 1


def month_from_groups(match: re.Match[str], group_prefix: str) -> int:
    """Read one side of a range match; year-only dates mean January."""
    year = int(match.group(f"{group_prefix}_year"))
    name = match.group(f"{group_prefix}_month_name")
    number = match.group(f"{group_prefix}_month_number")
    if name:
        month = MONTH_NAME_INDEX[name.lower()[:MONTH_ABBREVIATION_LENGTH]]
    elif number:
        month = int(number)
    else:
        month = 1
    return absolute_month(year, month)


MONTH_NAME_INDEX = {
    name[:MONTH_ABBREVIATION_LENGTH]: index
    for index, name in enumerate(MONTH_NAMES, start=1)
}


def find_date_ranges(text: str) -> list[DateRange]:
    """Find every non-overlapping date range in the text, in order."""
    ranges: list[DateRange] = []
    for match in DATE_RANGE_PATTERN.finditer(text):
        is_open = match.group("open_end") is not None
        ranges.append(
            DateRange(
                start_offset=match.start(),
                end_offset=match.end(),
                start_month=month_from_groups(match, "start"),
                end_month=None if is_open else month_from_groups(match, "end"),
            )
        )
    return ranges


def resolve_interval(date_range: DateRange, reference_month: int) -> tuple[int, int] | None:
    """Clip a range to the reference month; None when nothing valid remains.

    An open end runs to the reference month. A range that ends before it
    starts, or that starts after the reference month, is ignored.
    """
    end_month = reference_month if date_range.end_month is None else date_range.end_month
    end_month = min(end_month, reference_month)
    if end_month <= date_range.start_month:
        return None
    return date_range.start_month, end_month


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching month intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_years(ranges: list[DateRange], reference_date: date) -> float | None:
    """Total experience in years from date ranges, overlaps counted once.

    Args:
        ranges: Ranges found in experience entries.
        reference_date: The date that open ends and the future are clipped to.

    Returns:
        Years rounded to one decimal, or None when no valid range exists.
    """
    reference_month = absolute_month(reference_date.year, reference_date.month)
    intervals = [
        interval
        for date_range in ranges
        if (interval := resolve_interval(date_range, reference_month)) is not None
    ]
    if not intervals:
        return None
    total_months = sum(end - start for start, end in merge_intervals(intervals))
    return round(total_months / MONTHS_PER_YEAR, YEARS_DECIMAL_PLACES)
