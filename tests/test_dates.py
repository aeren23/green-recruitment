"""Tests for date range detection and experience arithmetic."""

from datetime import date

import pytest

from app.ingest.dates import absolute_month, find_date_ranges, total_years

REFERENCE = date(2018, 1, 1)


def only_range(text: str):
    """Return the single range found in the text."""
    ranges = find_date_ranges(text)
    assert len(ranges) == 1
    return ranges[0]


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("2015 - 2019", (2015, 1), (2019, 1)),
        ("2015-2019", (2015, 1), (2019, 1)),
        ("2015 \u2013 2019", (2015, 1), (2019, 1)),
        ("2015 to 2019", (2015, 1), (2019, 1)),
        ("2015 through 2019", (2015, 1), (2019, 1)),
        ("2015 TO 2019", (2015, 1), (2019, 1)),
        ("03/2015 - 11/2016", (2015, 3), (2016, 11)),
        ("3/2015 - 11/2016", (2015, 3), (2016, 11)),
        ("March 2015 to November 2016", (2015, 3), (2016, 11)),
        ("Mar 2015 - Nov 2016", (2015, 3), (2016, 11)),
        ("Sept 2015 - Oct. 2016", (2015, 9), (2016, 10)),
        ("March 2015 - 2016", (2015, 3), (2016, 1)),
        ("2015 - 06/2016", (2015, 1), (2016, 6)),
    ],
)
def test_range_forms_resolve_to_months(
    text: str, start: tuple[int, int], end: tuple[int, int]
) -> None:
    """Each separator and date format yields the right start and end month."""
    found = only_range(text)
    assert found.start_month == absolute_month(*start)
    assert found.end_month == absolute_month(*end)


@pytest.mark.parametrize("word", ["Present", "Current", "Now", "Günümüz", "GÜNÜMÜZ", "current"])
def test_open_end_words_leave_the_end_open(word: str) -> None:
    """Present, Current, Now and Günümüz mean the range has no end."""
    found = only_range(f"06/2015 - {word}")
    assert found.end_month is None


def test_range_spans_lines_and_whitespace_only_lines() -> None:
    """The dataset layout puts the date, `to` and the date on separate lines."""
    text = "Clerk\n03/2015\n \nto\n \nMarch 2016\nCompany Name"
    found = only_range(text)
    assert found.start_month == absolute_month(2015, 3)
    assert found.end_month == absolute_month(2016, 3)


@pytest.mark.parametrize(
    "text",
    [
        "Graduated 2015",
        "Version 2.0 released 2015",
        "Python 2015 and 2019 editions",
        "2015 2019",
        "12345 - 2019",
        "2015 - 20190",
        "Total 2015to2019",
        "from 2015 until 2019",
        "2015 - tomorrow",
    ],
)
def test_text_that_is_not_a_range_is_ignored(text: str) -> None:
    """Single years, glued words and unknown end words are not ranges."""
    assert find_date_ranges(text) == []


def test_several_ranges_are_found_in_order() -> None:
    """Each range gets its own offsets, in text order."""
    text = "2010 - 2012\nSome text\n2013 - 2015"
    first, second = find_date_ranges(text)
    assert first.end_offset <= second.start_offset


def test_total_years_of_a_plain_range() -> None:
    """A year-only range is counted from January to January."""
    assert total_years(find_date_ranges("2010 - 2014"), REFERENCE) == 4.0


def test_overlapping_ranges_are_merged() -> None:
    """Two jobs held at once count once."""
    ranges = find_date_ranges("2010 - 2014\n2012 - 2016")
    assert total_years(ranges, REFERENCE) == 6.0


def test_touching_and_nested_ranges_are_merged() -> None:
    """Back-to-back and contained ranges add nothing twice."""
    ranges = find_date_ranges("2010 - 2012\n2012 - 2014\n2011 - 2013")
    assert total_years(ranges, REFERENCE) == 4.0


def test_disjoint_ranges_add_up() -> None:
    """A gap between jobs is not counted."""
    ranges = find_date_ranges("2008 - 2010\n2012 - 2013")
    assert total_years(ranges, REFERENCE) == 3.0


def test_open_end_runs_to_the_reference_month() -> None:
    """`Current` means the reference month, so the date decides the result."""
    ranges = find_date_ranges("01/2015 - Current")
    assert total_years(ranges, date(2018, 1, 1)) == 3.0
    assert total_years(ranges, date(2020, 7, 15)) == 5.5


def test_future_end_is_clipped_to_the_reference_date() -> None:
    """A range reaching past the reference date stops at it."""
    assert total_years(find_date_ranges("2016 - 2025"), REFERENCE) == 2.0


def test_range_starting_after_the_reference_date_is_dropped() -> None:
    """Nothing before the reference date remains of a future range."""
    assert total_years(find_date_ranges("2019 - 2021"), REFERENCE) is None


def test_end_before_start_is_ignored() -> None:
    """A reversed range is bad data and does not subtract or add time."""
    ranges = find_date_ranges("2014 - 2010\n2015 - 2016")
    assert total_years(ranges, REFERENCE) == 1.0


def test_no_range_gives_none() -> None:
    """Unknown experience is None, never 0, so the MIN_YEARS filter keeps the CV."""
    assert total_years([], REFERENCE) is None


def test_only_reversed_ranges_give_none() -> None:
    """Without one valid range, experience is unknown."""
    assert total_years(find_date_ranges("2014 - 2010"), REFERENCE) is None


def test_months_are_used_and_the_total_is_rounded_to_one_decimal() -> None:
    """Fourteen months is 1.2 years."""
    ranges = find_date_ranges("01/2015 - 03/2016")
    assert total_years(ranges, REFERENCE) == 1.2
