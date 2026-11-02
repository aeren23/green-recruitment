"""Line cleaning shared by contact extraction and section splitting.

pypdf puts each text fragment on its own line and pads fragments with lines
that hold only a space, so a "blank line" has two meanings here: a whitespace
only line is padding and is dropped, while a truly empty line is kept as one
`BLANK_LINE` marker (collapsed when repeated) because it may separate entries.
"""

import re

LINE_SEPARATOR = "\n"
BLANK_LINE = ""
LINE_BREAK_PATTERN = re.compile(r"\r\n?")

# Zero-width space, non-joiner, joiner, word joiner and byte order mark. pypdf
# emits U+200B as a bullet stand-in, so it never carries meaning.
ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")

# Latin-1 letters and fractions that UTF-8 bullets and dashes turn into when
# they are decoded with the wrong code page (for example "ï¼" or "â\u20ac\u201c"). A
# line made only of these and non-alphanumeric characters is residue, not text.
MOJIBAKE_RESIDUE_CHARACTERS = frozenset(
    "ÂÃâïðœ¼½¾"
)

# pypdf puts the `-` of a date range such as `05/2003 - 09/2006` on a line of
# its own, so these lines are range separators and must survive cleaning.
LONE_SEPARATOR_LINES = frozenset({"-", "\u2013", "\u2014"})


def is_noise_line(line: str) -> bool:
    """True when a line has no letter or digit, or only mojibake residue."""
    return all(
        not character.isalnum() or character in MOJIBAKE_RESIDUE_CHARACTERS
        for character in line
    )


def clean_lines(text: str) -> list[str]:
    """Split text into trimmed content lines and `BLANK_LINE` markers.

    Whitespace-only lines, zero-width characters and noise lines are dropped,
    except a lone dash, which may be the separator of a date range. A truly
    empty line becomes one `BLANK_LINE` marker, never two in a row.

    Args:
        text: Extracted text, line structure intact.

    Returns:
        Content lines in order, with blank markers between them.
    """
    lines: list[str] = []
    for raw_line in LINE_BREAK_PATTERN.sub(LINE_SEPARATOR, text).split(LINE_SEPARATOR):
        cleaned = ZERO_WIDTH_PATTERN.sub("", raw_line).strip()
        if raw_line == BLANK_LINE:
            if lines and lines[-1] != BLANK_LINE:
                lines.append(BLANK_LINE)
        elif cleaned in LONE_SEPARATOR_LINES or (cleaned and not is_noise_line(cleaned)):
            lines.append(cleaned)
    return lines
