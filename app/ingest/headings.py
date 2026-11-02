"""Section heading vocabulary and whole-line heading matching (SPEC 6.3).

The Turkish headings are matched literally against Turkish CVs and must stay
in Turkish. Matching folds both sides with `fold_heading`, so `DENEYİM`,
`Deneyim` and `deneyim` are the same heading.
"""

import unicodedata
from enum import StrEnum

COLON = ":"
WORD_SEPARATOR = " "
COMBINING_DOT_ABOVE = "\u0307"
# Turkish has dotted and dotless I; plain casefolding maps "I" to "i" but
# leaves "İ" and "ı" alone, so uppercase Turkish headings would not match.
TURKISH_I_VARIANTS = str.maketrans({"\u0130": "i", "I": "i", "\u0131": "i"})
UNICODE_NORMAL_FORM = "NFKC"


class HeadingTarget(StrEnum):
    """What the text under a heading becomes."""

    SUMMARY = "summary"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    PROJECT = "project"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    LANGUAGES = "languages"
    IGNORED = "ignored"


HEADINGS_BY_TARGET: dict[HeadingTarget, tuple[str, ...]] = {
    HeadingTarget.SUMMARY: (
        "summary", "profile", "objective", "about", "professional summary",
        "executive profile", "career overview", "career focus",
        "professional profile", "accomplishments", "additional information",
        "özet", "hakkımda",
    ),
    HeadingTarget.SKILLS: (
        "skills", "technical skills", "core competencies", "qualifications",
        "skill highlights", "highlights", "core qualifications",
        "yetenekler", "beceriler", "yetkinlikler",
    ),
    HeadingTarget.EXPERIENCE: (
        "experience", "work experience", "work history",
        "professional experience", "employment", "deneyim", "iş deneyimi",
    ),
    HeadingTarget.PROJECT: ("projects", "personal projects", "projeler"),
    HeadingTarget.EDUCATION: ("education", "education and training", "eğitim"),
    HeadingTarget.CERTIFICATION: (
        "certifications", "certificates", "licenses", "sertifikalar",
    ),
    HeadingTarget.LANGUAGES: ("languages", "diller"),
    HeadingTarget.IGNORED: (
        "interests", "affiliations", "professional affiliations",
        "personal information",
    ),
}


def fold_heading(text: str) -> str:
    """Fold text for heading comparison, handling Turkish letter case.

    Args:
        text: A heading or a candidate line.

    Returns:
        NFKC-normalized, I/İ/ı-unified, casefolded text with single spaces.
    """
    normalized = unicodedata.normalize(UNICODE_NORMAL_FORM, text)
    unified = normalized.translate(TURKISH_I_VARIANTS).replace(COMBINING_DOT_ABOVE, "")
    return WORD_SEPARATOR.join(unified.casefold().split())


HEADING_LOOKUP: dict[str, HeadingTarget] = {
    fold_heading(heading): target
    for target, headings in HEADINGS_BY_TARGET.items()
    for heading in headings
}


def classify_heading(line: str) -> HeadingTarget | None:
    """Return the target of a heading line, or None for ordinary text.

    Only a whole-line match counts (a trailing colon is allowed), so a
    sentence such as "Experience in Python" is not a heading.
    """
    return HEADING_LOOKUP.get(fold_heading(line.strip().rstrip(COLON)))
