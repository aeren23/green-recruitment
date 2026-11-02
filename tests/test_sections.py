"""Tests for heading matching, entry splitting, chunking and the parsed profile.

All CV text is synthetic.
"""

import json
import logging
from datetime import date

import pytest

from app.ingest.contact import extract_contact, remove_contact
from app.ingest.entries import MAX_CHUNK_WORDS, chunk_words
from app.ingest.headings import HeadingTarget, classify_heading
from app.ingest.sections import ParsedCv, fold_heading, parse_cv
from app.models import SectionType

REFERENCE = date(2018, 1, 1)
OTHER_REFERENCE = date(2020, 7, 15)
BLANK = ""
PADDING = " "


def make_cv(*lines: str) -> str:
    """Join lines the way extracted PDF text is laid out."""
    return "\n".join(lines)


def parse(*lines: str, reference: date = REFERENCE) -> ParsedCv:
    """Parse a CV made of the given lines."""
    return parse_cv(make_cv(*lines), reference)


def texts_of(parsed: ParsedCv, section_type: SectionType) -> list[str]:
    """Return the texts of all sections of one type."""
    return [s.text for s in parsed.sections if s.type == section_type]


def types_of(parsed: ParsedCv) -> list[SectionType]:
    """Return the section types in order."""
    return [s.type for s in parsed.sections]


# --- headings ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "target"),
    [
        ("Summary", HeadingTarget.SUMMARY),
        ("Professional Summary", HeadingTarget.SUMMARY),
        ("Executive Profile", HeadingTarget.SUMMARY),
        ("Career Overview", HeadingTarget.SUMMARY),
        ("Career Focus", HeadingTarget.SUMMARY),
        ("Professional Profile", HeadingTarget.SUMMARY),
        ("Accomplishments", HeadingTarget.SUMMARY),
        ("Additional Information", HeadingTarget.SUMMARY),
        ("Highlights", HeadingTarget.SKILLS),
        ("Core Qualifications", HeadingTarget.SKILLS),
        ("Skills", HeadingTarget.SKILLS),
        ("Work History", HeadingTarget.EXPERIENCE),
        ("Education and Training", HeadingTarget.EDUCATION),
        ("Certifications", HeadingTarget.CERTIFICATION),
        ("Personal Projects", HeadingTarget.PROJECT),
        ("Languages", HeadingTarget.LANGUAGES),
        ("Interests", HeadingTarget.IGNORED),
        ("Affiliations", HeadingTarget.IGNORED),
        ("Professional Affiliations", HeadingTarget.IGNORED),
        ("Personal Information", HeadingTarget.IGNORED),
    ],
)
def test_english_headings_map_to_their_targets(line: str, target: HeadingTarget) -> None:
    """The SPEC 6.3 table, including the SC-15 and SC-16 additions."""
    assert classify_heading(line) is target


@pytest.mark.parametrize(
    ("line", "target"),
    [
        ("DENEYİM", HeadingTarget.EXPERIENCE),
        ("İş Deneyimi", HeadingTarget.EXPERIENCE),
        ("İŞ DENEYİMİ", HeadingTarget.EXPERIENCE),
        ("EĞİTİM", HeadingTarget.EDUCATION),
        ("HAKKIMDA", HeadingTarget.SUMMARY),
        ("ÖZET", HeadingTarget.SUMMARY),
        ("YETENEKLER", HeadingTarget.SKILLS),
        ("Beceriler", HeadingTarget.SKILLS),
        ("PROJELER", HeadingTarget.PROJECT),
        ("SERTİFİKALAR", HeadingTarget.CERTIFICATION),
        ("DİLLER", HeadingTarget.LANGUAGES),
    ],
)
def test_uppercase_turkish_headings_match(line: str, target: HeadingTarget) -> None:
    """Turkish dotted and dotless I fold correctly in uppercase headings."""
    assert classify_heading(line) is target


def test_fold_heading_unifies_turkish_case_forms() -> None:
    """`İ`, `I`, `ı` and `i` all fold to the same letter."""
    assert fold_heading("DENEYİM") == fold_heading("deneyim") == fold_heading("Deneyim")
    assert fold_heading("HAKKIMDA") == fold_heading("hakkımda")
    assert fold_heading("I\u0307") == fold_heading("İ")


@pytest.mark.parametrize("line", ["Skills:", "SKILLS :", "  Experience  ", "Work   History"])
def test_trailing_colon_and_padding_are_allowed(line: str) -> None:
    """Colons and extra whitespace do not hide a heading."""
    assert classify_heading(line) is not None


@pytest.mark.parametrize(
    "line",
    [
        "Experience in Python",
        "Experience with ledgers",
        "Education and Training programs",
        "Skills: Python, SQL",
        "Summary of the project plan",
        "Languages spoken at work",
    ],
)
def test_a_heading_word_starting_a_sentence_is_not_a_heading(line: str) -> None:
    """Only whole lines are headings."""
    assert classify_heading(line) is None


def test_experience_in_sentence_stays_inside_its_section() -> None:
    """The line "Experience in Python" neither starts a section nor is lost."""
    parsed = parse(
        "EXPERIENCE",
        "Developer",
        "2015 - 2017",
        "Experience in Python and SQL",
        "SKILLS",
        "Python",
    )
    assert types_of(parsed) == [SectionType.EXPERIENCE, SectionType.SKILLS]
    assert "Experience in Python and SQL" in parsed.sections[0].text


def test_sections_follow_uppercase_turkish_headings() -> None:
    """A Turkish CV with uppercase headings splits into the right types."""
    parsed = parse(
        "ÖZET", "Deneyimli muhasebeci.",
        "DENEYİM", "Muhasebe Uzmanı", "2012 - 2016", "Defter tutma.",
        "EĞİTİM", "Üniversite, 2008 - 2012",
        "YETENEKLER", "Excel, Logo",
    )
    assert types_of(parsed) == [
        SectionType.SUMMARY, SectionType.EXPERIENCE, SectionType.EDUCATION, SectionType.SKILLS,
    ]
    assert parsed.years_experience == 4.0


def test_text_before_the_first_heading_becomes_summary() -> None:
    """A preamble is kept as a summary section, before the others."""
    parsed = parse("Careful analyst with ledger skills.", "SKILLS", "Excel")
    assert types_of(parsed) == [SectionType.SUMMARY, SectionType.SKILLS]
    assert texts_of(parsed, SectionType.SUMMARY) == ["Careful analyst with ledger skills."]
    assert not parsed.no_headings_found


def test_no_headings_gives_one_summary_and_a_warning_flag() -> None:
    """Text without any heading is a single summary and sets the flag."""
    parsed = parse("Just a paragraph about", "a person who audits books.")
    assert types_of(parsed) == [SectionType.SUMMARY]
    assert parsed.no_headings_found
    assert parsed.sections[0].text == "Just a paragraph about\na person who audits books."


def test_empty_text_has_no_sections_and_the_flag() -> None:
    """Empty input does not raise."""
    parsed = parse_cv("", REFERENCE)
    assert parsed.sections == []
    assert parsed.no_headings_found
    assert parsed.years_experience is None


def test_heading_without_text_creates_no_section() -> None:
    """An empty section is dropped."""
    parsed = parse("SKILLS", "EDUCATION", "BSc Accounting")
    assert types_of(parsed) == [SectionType.EDUCATION]


def test_ignored_heading_text_is_absent_from_every_output() -> None:
    """Interests text is neither stored nor embedded, and ends the section above."""
    marker = "zebra-keeping hobby"
    parsed = parse(
        "SKILLS", "Excel",
        "INTERESTS", marker, "Chess club",
        "EDUCATION", "BSc Accounting",
    )
    assert types_of(parsed) == [SectionType.SKILLS, SectionType.EDUCATION]
    everything = json.dumps(parsed.profile_json) + "".join(s.text for s in parsed.sections)
    assert marker not in everything
    assert "Chess club" not in everything
    assert texts_of(parsed, SectionType.SKILLS) == ["Excel"]


@pytest.mark.parametrize("heading", ["Affiliations", "Professional Affiliations", "Personal Information"])
def test_every_ignored_heading_drops_its_text(heading: str) -> None:
    """All four ignored headings behave the same."""
    parsed = parse("SKILLS", "Excel", heading, "dropped text here")
    assert "dropped" not in json.dumps(parsed.profile_json)


# --- noise and structure ----------------------------------------------------


def test_noise_lines_are_removed() -> None:
    """Zero-width, symbol-only and mojibake-only lines never reach a section."""
    parsed = parse(
        "SKILLS", "\u200b", "-", "ï¼", "â\u20ac\u201c", "Â", PADDING,
        "Excel", "\u200b\u200b", "...", "Word",
    )
    assert texts_of(parsed, SectionType.SKILLS) == ["Excel\nWord"]


def test_zero_width_characters_are_stripped_from_lines() -> None:
    """A zero-width space inside a line does not stop heading detection."""
    parsed = parse("\u200bSKILLS\u200b", "Excel")
    assert types_of(parsed) == [SectionType.SKILLS]


def test_skills_stay_one_section() -> None:
    """A skills list, blank lines and all, is not split into entries."""
    parsed = parse("SKILLS", "Excel", BLANK, "Word", BLANK, "Access")
    assert texts_of(parsed, SectionType.SKILLS) == ["Excel\nWord\nAccess"]


# --- entry splitting --------------------------------------------------------


def dataset_layout_entry(title: str, start: str, end: str, *body: str) -> list[str]:
    """One entry in the pypdf layout: title, start, padding, `to`, end, body."""
    return [title, start, PADDING, "to", end, "Company Name", *body]


def test_entries_split_at_each_date_range() -> None:
    """Each date range starts an entry; the title above it moves along."""
    parsed = parse(
        "EXPERIENCE",
        *dataset_layout_entry("Junior Accountant", "03/2012", "05/2014", "Posted journal entries."),
        *dataset_layout_entry("Senior Accountant", "06/2014", "Current", "Led month end close."),
    )
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first.startswith("Junior Accountant") and "journal entries" in first
    assert second.startswith("Senior Accountant") and "month end close" in second
    assert "Senior Accountant" not in first


def test_title_line_attaches_to_the_entry_below_it() -> None:
    """The short line directly above a date range is the new entry's title."""
    parsed = parse(
        "EXPERIENCE",
        "Clerk", "2010 - 2012", "Filed invoices.",
        "Supervisor", "2012 - 2014", "Ran the team.",
    )
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first == "Clerk\n2010 - 2012\nFiled invoices."
    assert second == "Supervisor\n2012 - 2014\nRan the team."


def test_sentence_above_a_date_is_not_taken_as_a_title() -> None:
    """A line ending like a sentence stays with the previous entry."""
    parsed = parse(
        "EXPERIENCE",
        "Clerk", "2010 - 2012", "Filed all the invoices every single day.",
        "2012 - 2014", "Ran the team.",
    )
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first.endswith("Filed all the invoices every single day.")
    assert second == "2012 - 2014\nRan the team."


def test_long_line_above_a_date_is_not_taken_as_a_title() -> None:
    """More than eight words is not a title."""
    long_line = "one two three four five six seven eight nine"
    parsed = parse("EXPERIENCE", "2010 - 2012", long_line, "2012 - 2014", "Ran the team.")
    first, _ = texts_of(parsed, SectionType.EXPERIENCE)
    assert long_line in first


def test_end_date_line_is_not_mistaken_for_a_title() -> None:
    """When ranges touch, the previous range's end line stays with its entry."""
    parsed = parse("EXPERIENCE", "01/2010", "to", "Current", "02/2011", "to", "Present")
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first.endswith("Current")
    assert second.startswith("02/2011")


def test_date_on_the_title_line_starts_the_entry_at_that_line() -> None:
    """A one-line header with company and dates keeps its whole line."""
    parsed = parse(
        "EXPERIENCE",
        "Clerk, Example Corp, 2010 - 2012", "Filed invoices.",
        "Lead, Example Corp, 2012 - 2014", "Ran the team.",
    )
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first.startswith("Clerk, Example Corp")
    assert second.startswith("Lead, Example Corp")


def test_text_before_the_first_date_stays_with_the_first_entry() -> None:
    """Nothing is lost when the section opens with undated text."""
    parsed = parse("EXPERIENCE", "Ten years in audit.", "Clerk", "2010 - 2012", "Filed.")
    assert texts_of(parsed, SectionType.EXPERIENCE) == [
        "Ten years in audit.\nClerk\n2010 - 2012\nFiled."
    ]


def test_whitespace_only_lines_never_split_entries() -> None:
    """pypdf padding lines between fragments do not cut an entry apart."""
    parsed = parse(
        "EXPERIENCE", "Clerk", "2010 - 2012",
        "Filed invoices.", PADDING, "   ", "Matched payments.", PADDING, "Closed books.",
    )
    entries = texts_of(parsed, SectionType.EXPERIENCE)
    assert len(entries) == 1
    assert "Filed invoices.\nMatched payments.\nClosed books." in entries[0]


def test_real_blank_lines_do_not_split_a_section_that_has_dates() -> None:
    """Dates decide the split when the section has any."""
    parsed = parse("EXPERIENCE", "Clerk", "2010 - 2012", "Filed.", BLANK, "Matched payments.")
    assert len(texts_of(parsed, SectionType.EXPERIENCE)) == 1


def test_real_blank_lines_split_a_section_without_dates() -> None:
    """With no dates in the section, a truly empty line separates entries."""
    parsed = parse(
        "PROJECTS", "Ledger tool", "Built a tool.", BLANK, "Audit tool", "Built another.",
    )
    assert texts_of(parsed, SectionType.PROJECT) == [
        "Ledger tool\nBuilt a tool.",
        "Audit tool\nBuilt another.",
    ]


def test_project_entries_split_by_date_like_experience() -> None:
    """Project sections use the same date based splitting."""
    parsed = parse("PROJECTS", "Tool A", "2014 - 2015", "Built.", "Tool B", "2016 - 2017", "Built.")
    assert len(texts_of(parsed, SectionType.PROJECT)) == 2


def test_education_is_not_split_into_entries() -> None:
    """Only experience and project sections are split."""
    parsed = parse("EDUCATION", "BSc Accounting", "2008 - 2012", "MSc Finance", "2012 - 2014")
    assert len(texts_of(parsed, SectionType.EDUCATION)) == 1


# --- chunking ---------------------------------------------------------------


def words(count: int) -> str:
    """A line of `count` distinct words."""
    return " ".join(f"w{index}" for index in range(count))


def test_chunk_words_keeps_short_text_whole() -> None:
    """Text within the limit is returned unchanged."""
    text = "a b\nc"
    assert chunk_words(text) == [text]


def test_chunk_words_cuts_at_the_word_limit() -> None:
    """450 words make pieces of 200, 200 and 50 words."""
    pieces = chunk_words(words(450))
    assert [len(piece.split()) for piece in pieces] == [MAX_CHUNK_WORDS, MAX_CHUNK_WORDS, 50]
    assert " ".join(pieces) == words(450)


def test_chunk_words_keeps_line_breaks_inside_a_piece() -> None:
    """Pieces are slices of the original text, newlines included."""
    text = "a b\nc d\ne f"
    assert chunk_words(text, max_words=3) == ["a b\nc", "d\ne f"]


def test_exactly_200_words_is_one_piece() -> None:
    """The limit is inclusive."""
    assert len(chunk_words(words(MAX_CHUNK_WORDS))) == 1
    assert len(chunk_words(words(MAX_CHUNK_WORDS + 1))) == 2


def test_chunk_words_of_blank_text_is_empty() -> None:
    """No words, no pieces."""
    assert chunk_words("  \n ") == []


def test_long_entry_becomes_consecutive_sections() -> None:
    """A 450 word entry yields three sections with consecutive ordinals."""
    parsed = parse("SUMMARY", "Intro.", "EXPERIENCE", "Clerk", "2010 - 2012", words(450))
    experience = [s for s in parsed.sections if s.type == SectionType.EXPERIENCE]
    assert [len(s.text.split()) for s in experience] == [200, 200, 54]
    assert [s.ordinal for s in parsed.sections] == list(range(len(parsed.sections)))


# --- years of experience ----------------------------------------------------


def years(*lines: str, reference: date = REFERENCE) -> float | None:
    """Parse an experience section and return its years."""
    return parse("EXPERIENCE", *lines, reference=reference).years_experience


def test_range_split_over_lines_with_to() -> None:
    """The dataset layout: start, padding, `to`, end on separate lines."""
    assert years("Clerk", "03/2012", PADDING, "to", PADDING, "03/2015") == 3.0


def test_month_year_names_and_month_numbers() -> None:
    """`MM/YYYY` and `Month YYYY` use month resolution."""
    assert years("January 2012", "to", "July 2013") == 1.5
    assert years("Jan 2012 - Jul 2013") == 1.5
    assert years("01/2012 - 07/2013") == 1.5


@pytest.mark.parametrize("end", ["Current", "Present", "Now", "Günümüz"])
def test_open_end_words_run_to_the_reference_date(end: str) -> None:
    """Each end word means "up to the reference date"."""
    assert years("Clerk", "06/2015", "to", end) == 2.6


def test_reference_date_changes_open_ended_years() -> None:
    """The same CV gives different experience for different reference dates."""
    lines = ("Clerk", "01/2015", "to", "Current")
    assert years(*lines, reference=REFERENCE) == 3.0
    assert years(*lines, reference=OTHER_REFERENCE) == 5.5


def test_overlapping_ranges_are_counted_once() -> None:
    """Two concurrent jobs do not double count."""
    assert years("A", "2010 - 2014", "B", "2012 - 2016") == 6.0


def test_education_dates_are_not_counted() -> None:
    """Only the experience section feeds `years_experience`."""
    parsed = parse(
        "EDUCATION", "BSc Accounting", "2000 - 2004",
        "PROJECTS", "Tool", "2005 - 2006",
        "EXPERIENCE", "Clerk", "2010 - 2012",
    )
    assert parsed.years_experience == 2.0


def test_no_dates_gives_none() -> None:
    """Without any range, years are unknown, not zero."""
    assert years("Clerk at Example Corp for a long time.") is None
    assert parse("EDUCATION", "BSc", "2000 - 2004").years_experience is None


def test_ranges_in_several_experience_sections_are_merged() -> None:
    """A split experience history still adds up once."""
    parsed = parse(
        "EXPERIENCE", "A", "2008 - 2010",
        "SKILLS", "Excel",
        "WORK HISTORY", "B", "2009 - 2012",
    )
    assert parsed.years_experience == 4.0


@pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014"])
def test_dash_on_its_own_line_separates_a_range(dash: str) -> None:
    """pypdf puts a lone `-` between the dates on a line of its own."""
    assert years("Clerk", "05/2012", PADDING, dash, PADDING, "05/2015") == 3.0
    assert years("Clerk", "05/2012", PADDING, dash, "Current") == 5.7


def test_dash_line_does_not_become_a_title_or_noise_in_other_sections() -> None:
    """A lone dash above a date is no title, and is dropped outside entries."""
    parsed = parse(
        "EXPERIENCE", "Clerk", "2010 - 2012", "Filed.", "-", "2012 - 2014", "Ran the team.",
        "SKILLS", "-", "Excel", "-", "Word",
    )
    first, second = texts_of(parsed, SectionType.EXPERIENCE)
    assert first.endswith("-") and second.startswith("2012")
    assert texts_of(parsed, SectionType.SKILLS) == ["Excel\nWord"]


def test_hyphen_ranges_still_work() -> None:
    """The SPEC's original `YYYY - YYYY` form is covered by the same rule."""
    assert years("Clerk, 2015 - 2017") == 2.0


# --- languages and profile --------------------------------------------------


def test_languages_are_mapped_to_english_and_not_embedded() -> None:
    """Turkish names map to English; the section has no Section row."""
    parsed = parse(
        "SKILLS", "Excel",
        "DİLLER", "İngilizce - İleri", "Almanca: orta", "Turkish (native)", "Fluent",
    )
    assert parsed.languages == ["english", "german", "turkish"]
    assert types_of(parsed) == [SectionType.SKILLS]
    assert "ngilizce" not in json.dumps(parsed.profile_json)


def test_unknown_language_words_are_dropped_and_duplicates_merged() -> None:
    """Proficiency words and unknown languages do not appear."""
    parsed = parse("LANGUAGES", "English, english", "Klingon", "Native speaker")
    assert parsed.languages == ["english"]


def test_no_languages_section_gives_an_empty_list() -> None:
    """Unknown languages are an empty list (the filter treats it as unknown)."""
    assert parse("SKILLS", "Excel").languages == []


def test_profile_json_follows_the_spec_schema() -> None:
    """Keys, parser version, ordinals and the null location."""
    parsed = parse("EXPERIENCE", "Clerk", "2010 - 2012", "LANGUAGES", "English")
    profile = parsed.profile_json
    assert set(profile) == {"parser_version", "sections", "years_experience", "languages", "location"}
    assert profile["parser_version"] == "1"
    assert profile["location"] is None
    assert profile["years_experience"] == 2.0
    assert profile["languages"] == ["english"]
    assert profile["sections"] == [
        {"type": "experience", "ordinal": 0, "text": "Clerk\n2010 - 2012"}
    ]
    json.dumps(profile)


# --- KVKK -------------------------------------------------------------------

MARKER_NAME = "Jane Roe"
MARKER_EMAIL = "jane.roe@example.org"
MARKER_PHONE = "+90 532 111 22 33"
MARKER_PHONE_US = "555-010-4477"


def scrubbed_parse(*lines: str) -> ParsedCv:
    """Run the real ingest chain after extraction: scrub contact, then parse."""
    text = make_cv(*lines)
    return parse_cv(remove_contact(text, extract_contact(text)), REFERENCE)


def test_contact_details_are_absent_from_sections_and_profile_json() -> None:
    """Name, email and phone appear in no section text and not in profile_json."""
    parsed = scrubbed_parse(
        MARKER_NAME, MARKER_EMAIL, MARKER_PHONE,
        "SUMMARY", f"Write to {MARKER_EMAIL}.",
        "EXPERIENCE", "Clerk", "2010 - 2012", f"Reference: {MARKER_NAME}, {MARKER_PHONE_US}.",
        "SKILLS", "Excel",
    )
    everything = json.dumps(parsed.profile_json) + "".join(s.text for s in parsed.sections)
    for marker in (MARKER_NAME, MARKER_EMAIL, MARKER_PHONE, MARKER_PHONE_US, "Roe"):
        assert marker not in everything
    assert parsed.years_experience == 2.0


@pytest.mark.parametrize(
    "shape",
    [
        "555) 010-4477", "(555 010-4477", "555-4477", "555.4477", "555 4477",
        "+44 20 7946 0958", "(+90) 532 111 22 33", "01632 960123",
    ],
)
def test_every_phone_shape_is_absent_from_sections_and_profile_json(shape: str) -> None:
    """Regression: unbalanced-parenthesis and local numbers once survived scrubbing."""
    parsed = scrubbed_parse(
        MARKER_NAME, "SUMMARY", f"Call {shape} after nine.",
        "EXPERIENCE", "Clerk", "2010 - 2012", f"Office line {shape}.",
    )
    everything = json.dumps(parsed.profile_json) + "".join(s.text for s in parsed.sections)
    assert shape not in everything
    assert shape.split()[-1] not in everything
    assert parsed.years_experience == 2.0


def test_scrubbing_keeps_experience_dates_intact() -> None:
    """Phone removal must not eat date ranges next to it."""
    parsed = scrubbed_parse(
        MARKER_NAME, MARKER_PHONE_US, "EXPERIENCE", "Clerk", "01/2012", "to", "07/2013",
    )
    assert parsed.years_experience == 1.5


def test_parsing_writes_no_log_records(caplog: pytest.LogCaptureFixture) -> None:
    """The parser emits nothing that could carry CV text."""
    with caplog.at_level(logging.DEBUG):
        scrubbed_parse(MARKER_NAME, "EXPERIENCE", "Clerk", "2010 - 2012", "INTERESTS", "chess")
    assert caplog.records == []


def test_parse_errors_and_reprs_carry_no_cv_text() -> None:
    """Failure messages and `repr` output never include section text."""
    marker = "zebra-keeping hobby"
    parsed = parse("SKILLS", marker)
    assert marker not in repr(parsed) + repr(parsed.sections[0])
    with pytest.raises(TypeError) as error:
        parse_cv(marker.encode(), REFERENCE)  # type: ignore[arg-type]
    assert marker not in str(error.value)
    assert error.value.__cause__ is None
