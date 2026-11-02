"""Spoken-language extraction from a `languages` section (SPEC 6.3).

Turkish names are mapped to English so they match `required_languages`.
Unknown words (proficiency levels, filler) are dropped.
"""

import re
from collections.abc import Iterable

from app.ingest.headings import fold_heading

WORD_PATTERN = re.compile(r"[^\W\d_]+")

ENGLISH_NAMES_BY_LANGUAGE_WORD = {
    "english": "english", "ingilizce": "english",
    "turkish": "turkish", "türkçe": "turkish", "turkce": "turkish",
    "german": "german", "almanca": "german",
    "french": "french", "fransızca": "french", "fransizca": "french",
    "spanish": "spanish", "ispanyolca": "spanish",
    "arabic": "arabic", "arapça": "arabic", "arapca": "arabic",
    "russian": "russian", "rusça": "russian", "rusca": "russian",
    "italian": "italian", "italyanca": "italian",
    "chinese": "chinese", "çince": "chinese", "cince": "chinese",
    "japanese": "japanese", "japonca": "japanese",
}
LANGUAGE_LOOKUP = {
    fold_heading(word): language
    for word, language in ENGLISH_NAMES_BY_LANGUAGE_WORD.items()
}


def extract_languages(lines: Iterable[str]) -> list[str]:
    """Return the known languages named in the lines, deduplicated in order."""
    found: list[str] = []
    for line in lines:
        for word in WORD_PATTERN.findall(line):
            language = LANGUAGE_LOOKUP.get(fold_heading(word))
            if language is not None and language not in found:
                found.append(language)
    return found
