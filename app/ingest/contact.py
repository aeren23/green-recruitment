"""Contact extraction and blind-screening removal (SPEC 6.3, KVKK).

Email and phone come from regular expressions, the name from a first-line
heuristic. The name rule is deliberately inclusive: a missed name is a KVKK
leak, while a removed job title (the dataset's first line is a title, not a
name) is only a small quality loss. Fields are hidden from `repr` so a stray
log line cannot carry them.
"""

import re
from dataclasses import dataclass, field

from app.ingest.headings import classify_heading
from app.ingest.text_lines import clean_lines, is_noise_line

EMAIL_PATTERN = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")

# Phone patterns are deliberately inclusive: a leaked number is a KVKK breach,
# while a removed digit group is a small loss. Separators are spaces, dots and
# hyphens, never a newline, so numbers on neighbouring lines are not joined.
SEPARATOR = r"[ \t.-]*"
LOCAL_SEPARATOR = r"(?:[ \t]?[.-][ \t]?|[ \t])"
# `+90 5xx xxx xx xx`, `0 (5xx) xxx xx xx`, `05xxxxxxxxx`; the area code
# starts with 2 to 5.
TURKISH_PHONE = (
    rf"(?:(?:\+|00)?90{SEPARATOR})?(?:0{SEPARATOR})?"
    rf"\(?[2-5]\d{{2}}\)?{SEPARATOR}\d{{3}}{SEPARATOR}\d{{2}}{SEPARATOR}\d{{2}}"
)
# `ddd-ddd-dddd`, `(ddd) ddd-dddd`, also with an unbalanced or missing
# parenthesis (`ddd) ddd-dddd`) and an optional country code 1.
AREA_CODE_PHONE = (
    rf"(?:\+?1{SEPARATOR})?\(?\d{{3}}\)?{SEPARATOR}"
    rf"\d{{3}}{SEPARATOR}\d{{4}}"
)
FULL_PHONE_PATTERN = re.compile(rf"(?<!\d)(?:{TURKISH_PHONE}|{AREA_CODE_PHONE})(?!\d)")
# International numbers: a `+` or `00` prefix, then groups of digits, for
# example `+44 20 7946 0958`.
INTERNATIONAL_PHONE = (
    rf"\(?(?:\+|(?<!\d)00{SEPARATOR})\d{{1,3}}\)?(?:{SEPARATOR}\(?\d{{1,5}}\)?){{2,5}}(?!\d)"
)
# A parenthesised area code with an optional prefix: `(ddd) ddd-ddd`,
# `dd (dddd) ddd-ddd`.
PARENTHESISED_PHONE = (
    rf"(?<!\d)(?:\d{{1,4}}{SEPARATOR})?\(\d{{2,5}}\){SEPARATOR}"
    rf"\d{{3}}(?:{SEPARATOR}\d{{3,4}})?(?!\d)"
)
# Digit groups that no other pattern covers: `ddd-ddd-ddd`, a trunk or area
# prefix followed by a 6 to 8 digit subscriber number (`0212 1234567`,
# `01632 960123`), and `dddd ddd ddd` that does not start like a year.
GROUPED_PHONE = (
    rf"(?<![\w/])(?:\d{{3}}[ \t.-]\d{{3}}[ \t.-]\d{{3}}"
    rf"|\d{{2,5}}[ \t.-]\d{{6,8}}"
    rf"|(?!(?:19|20)\d{{2}})\d{{4}}[ \t.-]\d{{3}}[ \t.-]\d{{3,4}})(?!\d)"
)
# Seven-digit local numbers: `ddd-dddd`, `ddd.dddd`, `ddd dddd`. They must not
# sit inside a word or after a slash, so `AB123-4567` and `01/2015` are safe.
LOCAL_PHONE = rf"(?<![\w/])\d{{3}}{LOCAL_SEPARATOR}\d{{4}}(?!\d)"
PHONE_PATTERN = re.compile(
    "|".join(
        (
            FULL_PHONE_PATTERN.pattern,
            INTERNATIONAL_PHONE,
            PARENTHESISED_PHONE,
            GROUPED_PHONE,
            LOCAL_PHONE,
        )
    )
)

MIN_NAME_WORDS = 2
MAX_NAME_WORDS = 4
NAME_WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['\u2019.-][^\W\d_]+)*\.?")
# Letters that have a case partner the regex flag does not pair (Turkish I).
TURKISH_I_LETTERS = "i\u0131\u0130I"
TURKISH_I_CLASS = f"[{TURKISH_I_LETTERS}]"
NAME_WORD_GAP = r"\s+"
WORD_EDGE_BEFORE = r"(?<!\w)"
WORD_EDGE_AFTER = r"(?!\w)"
REMOVAL_REPLACEMENT = ""


@dataclass(frozen=True)
class ExtractedContact:
    """Contact details found in a CV; every field is optional.

    The fields are excluded from `repr` on purpose (KVKK).
    """

    name: str | None = field(default=None, repr=False)
    email: str | None = field(default=None, repr=False)
    phone: str | None = field(default=None, repr=False)


def is_plausible_name(line: str) -> bool:
    """True for 2 to 4 words made of letters only, and not a section heading."""
    words = line.split()
    if not MIN_NAME_WORDS <= len(words) <= MAX_NAME_WORDS:
        return False
    if classify_heading(line) is not None:
        return False
    return all(NAME_WORD_PATTERN.fullmatch(word) for word in words)


def first_line_name(text: str) -> str | None:
    """Return the first content line when it looks like a name, else None."""
    content_lines = [line for line in clean_lines(text) if line and not is_noise_line(line)]
    if content_lines and is_plausible_name(content_lines[0]):
        return " ".join(content_lines[0].split())
    return None


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the first match of a pattern, trimmed, or None."""
    match = pattern.search(text)
    return match.group().strip() if match else None


def extract_contact(text: str) -> ExtractedContact:
    """Find the name, email and phone number in extracted CV text.

    Args:
        text: Output of `extract_text`.

    Returns:
        The contact details found; a field is None when nothing matched.
    """
    return ExtractedContact(
        name=first_line_name(text),
        email=first_match(EMAIL_PATTERN, text),
        phone=first_match(FULL_PHONE_PATTERN, text) or first_match(PHONE_PATTERN, text),
    )


def name_pattern(name: str) -> re.Pattern[str]:
    """Build a case-insensitive pattern for a name with flexible whitespace.

    Turkish dotted and dotless I are matched interchangeably so that
    `İbrahim Çelik` is also found as `IBRAHIM ÇELIK`.
    """
    words = [
        "".join(
            TURKISH_I_CLASS if character in TURKISH_I_LETTERS else re.escape(character)
            for character in word
        )
        for word in name.split()
    ]
    body = NAME_WORD_GAP.join(words)
    return re.compile(f"{WORD_EDGE_BEFORE}{body}{WORD_EDGE_AFTER}", re.IGNORECASE)


def remove_exact(text: str, value: str | None) -> str:
    """Remove every case-insensitive occurrence of a literal value."""
    if not value:
        return text
    return re.sub(re.escape(value), REMOVAL_REPLACEMENT, text, flags=re.IGNORECASE)


def remove_contact(text: str, contact: ExtractedContact) -> str:
    """Remove contact details from text (blind screening).

    Every occurrence of the found name, email and phone is removed. Every other
    email address and phone number the patterns recognise goes too, so a second
    number in the CV cannot leak into a section.

    Args:
        text: Extracted CV text.
        contact: The details found by `extract_contact`.

    Returns:
        The text without contact details.
    """
    scrubbed = remove_exact(text, contact.email)
    scrubbed = remove_exact(scrubbed, contact.phone)
    scrubbed = EMAIL_PATTERN.sub(REMOVAL_REPLACEMENT, scrubbed)
    scrubbed = PHONE_PATTERN.sub(REMOVAL_REPLACEMENT, scrubbed)
    if contact.name:
        scrubbed = name_pattern(contact.name).sub(REMOVAL_REPLACEMENT, scrubbed)
    return scrubbed
