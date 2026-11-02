"""Entry splitting and word chunking for section text (SPEC 6.3).

An experience or project entry starts where its date range starts. In the
dataset's common layout the job title sits on the line directly above the start
date, so a short title-like line there moves into the new entry.
"""

import re

from app.ingest.dates import DateRange, find_date_ranges
from app.ingest.text_lines import LINE_SEPARATOR, is_noise_line

MAX_CHUNK_WORDS = 200
MAX_TITLE_WORDS = 8
TITLE_TERMINAL_PUNCTUATION = ".!?;"
BLANK_LINE_SEPARATOR = LINE_SEPARATOR * 2
WORD_PATTERN = re.compile(r"\S+")


def is_title_like(line: str) -> bool:
    """True for a short line with text that does not end like a sentence."""
    words = line.split()
    if not 1 <= len(words) <= MAX_TITLE_WORDS or is_noise_line(line):
        return False
    return not line.rstrip().endswith(tuple(TITLE_TERMINAL_PUNCTUATION))


def start_of_line(text: str, offset: int) -> int:
    """Return the offset where the line containing `offset` starts."""
    return text.rfind(LINE_SEPARATOR, 0, offset) + 1


def title_line_start_above(text: str, line_start: int) -> int | None:
    """Return the start of the title-like line directly above, if any."""
    if line_start == 0:
        return None
    previous_line_end = line_start - 1
    previous_line_start = start_of_line(text, previous_line_end)
    if is_title_like(text[previous_line_start:previous_line_end]):
        return previous_line_start
    return None


def entry_start(text: str, range_start: int, floor: int) -> int:
    """Return where the entry owning a date range begins.

    Args:
        text: The section text.
        range_start: Offset of the date range.
        floor: End of the previous range; an entry never reaches back past it.
    """
    line_start = start_of_line(text, range_start)
    if not text[line_start:range_start].strip():
        title_start = title_line_start_above(text, line_start)
        if title_start is not None and title_start >= floor:
            line_start = title_start
    return max(line_start, floor)


def split_at_date_ranges(text: str, ranges: list[DateRange]) -> list[str]:
    """Cut the text into one entry per date range.

    Text before the first entry joins the first entry, so nothing is lost.
    """
    starts: list[int] = []
    floor = 0
    for date_range in ranges:
        starts.append(entry_start(text, date_range.start_offset, floor))
        floor = date_range.end_offset
    boundaries = [0, *starts[1:], len(text)]
    pieces = (text[a:b].strip() for a, b in zip(boundaries, boundaries[1:]))
    return [piece for piece in pieces if piece]


def split_entries(text: str) -> list[str]:
    """Split experience or project text into entries.

    Date ranges separate entries. Only when the text holds no date at all, a
    truly empty line does (whitespace-only lines were removed earlier).

    Args:
        text: Section text, lines joined by newlines.

    Returns:
        Non-empty entry texts in order.
    """
    ranges = find_date_ranges(text)
    if ranges:
        return split_at_date_ranges(text, ranges)
    pieces = (piece.strip() for piece in text.split(BLANK_LINE_SEPARATOR))
    return [piece for piece in pieces if piece]


def chunk_words(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Cut text into pieces of at most `max_words` words, keeping line breaks.

    Args:
        text: Entry text.
        max_words: Word limit per piece (the e5-small limit is 512 tokens).

    Returns:
        The text itself when short enough, otherwise consecutive pieces.
    """
    words = list(WORD_PATTERN.finditer(text))
    if not words:
        return []
    return [
        text[words[first].start() : words[min(first + max_words, len(words)) - 1].end()]
        for first in range(0, len(words), max_words)
    ]
