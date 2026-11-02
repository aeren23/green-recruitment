"""Tests for contact extraction and removal. All people and numbers are made up."""

import logging

import pytest

from app.ingest.contact import (
    PHONE_PATTERN,
    ExtractedContact,
    extract_contact,
    remove_contact,
)

NAME_LINE = "Jane Roe"
EMAIL = "jane.roe@example.org"
PHONE_TURKISH = "+90 532 111 22 33"
PHONE_US = "555-010-4477"


def make_cv(*lines: str) -> str:
    """Join lines the way extracted PDF text is laid out."""
    return "\n".join(lines)


@pytest.mark.parametrize(
    "phone",
    [
        "+90 532 111 22 33",
        "+90 (532) 111 22 33",
        "+905321112233",
        "0 (532) 111 22 33",
        "0532 111 22 33",
        "05321112233",
        "0532-111-22-33",
        "905321112233",
        "555-010-4477",
        "(555) 010-4477",
        "555.010.4477",
        "+1 555 010 4477",
        "555) 010-4477",
        "(555 010-4477",
        "(555)010.4477",
        "555 010 4477",
        "555-4477",
        "555.4477",
        "555 4477",
        "+44 20 7946 0958",
        "+49 (30) 123456",
        "(+90) 532 111 22 33",
        "00 44 20 7946 0958",
        "(555) 010-447",
        "44 (2079) 460-958",
        "555-010-447",
        "0212 1234567",
        "01632 960123",
        "1000 123 4567",
    ],
)
def test_phone_pattern_matches_turkish_and_us_forms(phone: str) -> None:
    """Each documented phone form is found as one whole number."""
    match = PHONE_PATTERN.search(f"Call {phone} today")
    assert match is not None
    assert match.group().strip() == phone


@pytest.mark.parametrize(
    "text",
    [
        "2015 - 2019",
        "2015-2019",
        "2015 to 2019",
        "01/2015",
        "01/2015 - 03/2016",
        "12/2015 to 01/2016",
        "Jan 2015 - Dec 2016",
        "2012 2013 2014 2015",
        "2015 - Present",
        "Managed 120 staff and a 5000000 budget",
        "03/2018 - 12/2019",
        "01/2015 to 03/2018",
        "Jan 2015 - Dec 2018",
        "1999-2001",
        "GPA 3.85/4.00",
        "GPA 3.8 4.0",
        "Badge AB123-4567",
        "Serial ID5550104477890",
        "Account 12345678901234",
        "C++ 2015 - 2019",
        "Sales rose in 2015 2016 2017",
        "(2015) data",
        "12 2015 - 06 2016",
        "2000 - 2005",
        "$1,000,000 revenue",
    ],
)
def test_phone_pattern_ignores_dates_and_year_ranges(text: str) -> None:
    """Year ranges, month/year dates and ordinary numbers are not phone numbers."""
    assert PHONE_PATTERN.search(text) is None


def test_extracts_email_phone_and_name() -> None:
    """All three fields come out of a conventional CV header."""
    text = make_cv(NAME_LINE, EMAIL, PHONE_TURKISH, "SUMMARY", "Accountant.")
    contact = extract_contact(text)
    assert contact == ExtractedContact(name=NAME_LINE, email=EMAIL, phone=PHONE_TURKISH)


def test_missing_values_stay_none() -> None:
    """Nothing to find means None, not an empty string."""
    contact = extract_contact(make_cv("SUMMARY", "Worked on ledgers for nine years."))
    assert contact == ExtractedContact()


def test_name_skips_noise_lines_before_it() -> None:
    """Zero-width and symbol-only lines do not hide the first real line."""
    text = make_cv("\u200b", " ", "-", "ï¼", NAME_LINE, "SUMMARY", "Text.")
    assert extract_contact(text).name == NAME_LINE


@pytest.mark.parametrize(
    "first_line",
    [
        "Jane",
        "Jane Roe Smith Jones Brown",
        "Jane Roe 2nd",
        "Jane Roe 3000",
        "jane@example.org Roe",
        "SUMMARY",
        "Professional Summary",
        "Work History",
    ],
)
def test_first_line_that_is_not_a_name_is_rejected(first_line: str) -> None:
    """Too few or many words, digits, an at sign or a heading are not a name."""
    assert extract_contact(make_cv(first_line, "Text.")).name is None


def test_only_the_first_line_is_considered_for_the_name() -> None:
    """A name-like second line is not taken when the first line is not a name."""
    assert extract_contact(make_cv("Curriculum", NAME_LINE)).name is None


def test_job_title_on_the_first_line_is_accepted_by_design() -> None:
    """The heuristic is inclusive: a title is taken, because a missed name leaks."""
    assert extract_contact(make_cv("ACCOUNTANT ASSISTANT", "Text.")).name == (
        "ACCOUNTANT ASSISTANT"
    )


def test_hyphenated_apostrophe_and_dotted_names_are_accepted() -> None:
    """Common name punctuation does not make a name look like a non-name."""
    assert extract_contact("Mary-Anne O'Neil\nText.").name == "Mary-Anne O'Neil"
    assert extract_contact("John Q. Public\nText.").name == "John Q. Public"


def test_turkish_name_is_extracted() -> None:
    """Turkish letters are letters."""
    assert extract_contact("Ayşe Yılmaz\nText.").name == "Ayşe Yılmaz"


def test_remove_contact_removes_every_occurrence() -> None:
    """The found values disappear everywhere, not only from the header."""
    text = make_cv(
        NAME_LINE, EMAIL, PHONE_TURKISH, "SUMMARY",
        f"Reach {EMAIL} or {PHONE_TURKISH}.", f"References from {NAME_LINE}.",
    )
    scrubbed = remove_contact(text, extract_contact(text))
    for marker in (NAME_LINE, EMAIL, PHONE_TURKISH):
        assert marker not in scrubbed
    assert "SUMMARY" in scrubbed


def test_name_removal_is_case_insensitive() -> None:
    """An uppercase repetition of the name is removed too."""
    text = make_cv(NAME_LINE, "Text.", "Signed JANE ROE and jane   roe.")
    scrubbed = remove_contact(text, extract_contact(text))
    assert "roe" not in scrubbed.casefold()


def test_name_removal_handles_turkish_dotted_capital_i() -> None:
    """`İbrahim Çelik` is also removed when repeated as `IBRAHIM ÇELIK`."""
    text = make_cv("İbrahim Çelik", "Text.", "IBRAHIM ÇELIK signed.", "ibrahim çelik too.")
    scrubbed = remove_contact(text, extract_contact(text))
    assert "elik" not in scrubbed.casefold()
    assert "ELIK" not in scrubbed


def test_name_removal_does_not_cut_longer_words() -> None:
    """A word that merely contains the name is left alone."""
    text = make_cv(NAME_LINE, "Text.", "The Roeland Janerose group.")
    scrubbed = remove_contact(text, ExtractedContact(name="Jane Roe"))
    assert "Roeland Janerose" in scrubbed


def test_other_emails_and_phones_are_removed_as_well() -> None:
    """A second number or address, not returned by extract, cannot leak."""
    text = make_cv(
        NAME_LINE, PHONE_TURKISH, "Office: " + PHONE_US, "other.person@example.net"
    )
    scrubbed = remove_contact(text, extract_contact(text))
    for marker in (PHONE_US, "other.person@example.net", PHONE_TURKISH):
        assert marker not in scrubbed


def test_remove_contact_leaves_dates_alone() -> None:
    """Year ranges survive phone removal, otherwise experience would be lost."""
    text = make_cv(NAME_LINE, PHONE_US, "Clerk, 2015 - 2019", "01/2015 to 03/2016")
    scrubbed = remove_contact(text, extract_contact(text))
    assert "2015 - 2019" in scrubbed
    assert "01/2015 to 03/2016" in scrubbed


def test_remove_contact_with_nothing_found_returns_the_text() -> None:
    """No contact details means no change."""
    text = make_cv("SUMMARY", "Plain text without contact details.")
    assert remove_contact(text, ExtractedContact()) == text


def test_extracted_contact_repr_hides_values() -> None:
    """A stray `print(contact)` or log line cannot carry the details."""
    contact = ExtractedContact(name=NAME_LINE, email=EMAIL, phone=PHONE_US)
    rendered = repr(contact) + str(contact)
    for marker in (NAME_LINE, EMAIL, PHONE_US):
        assert marker not in rendered


def test_extraction_writes_nothing_to_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Extraction and removal emit no log record that could carry CV text."""
    text = make_cv(NAME_LINE, EMAIL, PHONE_US, "SUMMARY", "Text.")
    with caplog.at_level(logging.DEBUG):
        remove_contact(text, extract_contact(text))
    assert caplog.records == []


LEAK_SHAPES = (
    "555) 010-4477",
    "(555 010-4477",
    "555-4477",
    "555.4477",
    "555 4477",
    "+44 20 7946 0958",
    "(+90) 532 111 22 33",
    "44 (2079) 460-958",
    "555-010-447",
    "01632 960123",
)


@pytest.mark.parametrize("shape", LEAK_SHAPES)
def test_each_phone_shape_is_found_and_removed(shape: str) -> None:
    """Unbalanced parentheses and seven-digit local numbers are found and removed."""
    text = make_cv("SUMMARY", f"Reach me at {shape} any time.", "More text.")
    contact = extract_contact(text)
    assert contact.phone is not None and contact.phone.strip() in shape
    assert shape not in remove_contact(text, contact)


def test_local_number_next_to_a_split_area_code_line_is_removed() -> None:
    """A local number and an area-code line that lost its parenthesis both go."""
    text = make_cv(
        "SUMMARY", "Based in the city, office 555-4477, open daily.", "555) 010-4477", "End.",
    )
    scrubbed = remove_contact(text, extract_contact(text))
    assert "555-4477" not in scrubbed
    assert "010-4477" not in scrubbed
    assert "open daily" in scrubbed


def test_full_number_is_preferred_over_a_local_one_when_extracting() -> None:
    """The reported phone is the complete number even if a local one comes first."""
    text = make_cv("SUMMARY", "Desk 555-4477.", f"Mobile {PHONE_US}.")
    assert extract_contact(text).phone == PHONE_US
