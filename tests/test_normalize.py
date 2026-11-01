"""Tests for normalization and keyed hashing (T-02, SPEC section 6.2)."""

import hashlib
import hmac

import pytest

from app.ingest.normalize import (
    SHORT_HASH_LENGTH,
    compute_raw_hash,
    normalize,
    short_hash,
)
from app.ingest.pdf import extract_text
from pdf_fixtures import SAMPLE_CV_LINES, PdfLayout, build_pdf

FIRST_SECRET = "first-test-secret"
SECOND_SECRET = "second-test-secret"
SHA256_HEX_LENGTH = 64
EXPECTED_KEY_ENCODING = "utf-8"


def hash_text(text: str, secret: str = FIRST_SECRET) -> str:
    """Normalize and hash text the way the pipeline does."""
    return compute_raw_hash(normalize(text), secret)


def hash_pdf(pdf_bytes: bytes) -> str:
    """Hash the text extracted from a PDF."""
    return hash_text(extract_text(pdf_bytes))


def test_same_pdf_gives_same_hash_twice() -> None:
    """Hashing is deterministic."""
    pdf_bytes = build_pdf(SAMPLE_CV_LINES)
    assert hash_pdf(pdf_bytes) == hash_pdf(pdf_bytes)


def test_re_exported_pdf_gives_same_hash() -> None:
    """Different bytes with the same text hash identically."""
    original = build_pdf(SAMPLE_CV_LINES)
    re_exported = build_pdf(SAMPLE_CV_LINES, PdfLayout("Re-exported", True))
    assert original != re_exported
    assert hash_pdf(original) == hash_pdf(re_exported)


def test_different_text_gives_different_hash() -> None:
    """A real content change moves the hash, so the cache cannot over-match."""
    changed_lines = [*SAMPLE_CV_LINES, "Additional certification: Kubernetes"]
    assert hash_pdf(build_pdf(SAMPLE_CV_LINES)) != hash_pdf(build_pdf(changed_lines))


def test_case_differences_do_not_change_the_hash() -> None:
    """Upper and lower case hash the same."""
    assert hash_text("Senior Python Developer") == hash_text("SENIOR PYTHON DEVELOPER")


def test_whitespace_differences_do_not_change_the_hash() -> None:
    """Spaces, tabs and line breaks collapse before hashing."""
    spaced = "Senior   Python\tDeveloper\n\n Backend  "
    assert hash_text(spaced) == hash_text("senior python developer backend")


def test_cplusplus_and_csharp_are_preserved() -> None:
    """The documented example from SPEC section 6.2."""
    assert normalize("C++, C#!") == "c++ c#"


def test_c_cplusplus_and_csharp_hash_differently() -> None:
    """Keeping `+` and `#` is what makes the three languages distinguishable."""
    hashes = {hash_text("c"), hash_text("c++"), hash_text("c#")}
    assert len(hashes) == 3


def test_nfkc_expands_ligatures() -> None:
    """The `fi` ligature becomes two letters."""
    assert normalize("oﬃce ﬁle") == "office file"


def test_nfkc_folds_fullwidth_characters() -> None:
    """Fullwidth letters and plus signs become their ASCII forms."""
    assert normalize("Ｃ＋＋") == "c++"


def test_casefold_goes_beyond_lowercase() -> None:
    """Casefold maps the German sharp s to `ss`, which lower() would not."""
    assert normalize("Straße") == "strasse"


@pytest.mark.parametrize(
    "punctuation",
    [
        "—",  # em dash
        "•",  # bullet
        "…",  # horizontal ellipsis (NFKC turns it into three dots)
        "“",  # left double quotation mark
        "¿",  # inverted question mark
        "、",  # ideographic comma
        "،",  # Arabic comma
    ],
)
def test_non_ascii_punctuation_becomes_a_space(punctuation: str) -> None:
    """Unicode punctuation is removed like ASCII punctuation."""
    assert normalize(f"alpha{punctuation}beta") == "alpha beta"


def test_ascii_punctuation_becomes_a_space() -> None:
    """Common ASCII punctuation splits words."""
    assert normalize("a.b,c;d:e!f?g(h)i/j-k_l") == "a b c d e f g h i j k l"


@pytest.mark.parametrize("text", ["", " ", " \t\r\n ", "  ", "...", "!?"])
def test_empty_or_blank_input_gives_empty_string(text: str) -> None:
    """Nothing but whitespace or punctuation normalizes to an empty string."""
    assert normalize(text) == ""


def test_normalize_is_idempotent() -> None:
    """Normalizing twice changes nothing."""
    once = normalize("Hello,  World! C++ Ｃ# ﬁle")
    assert normalize(once) == once


def test_hmac_matches_stdlib_reference() -> None:
    """The digest equals an independent HMAC-SHA256 computation."""
    normalized = normalize("Senior Python Developer")
    expected = hmac.new(
        FIRST_SECRET.encode(EXPECTED_KEY_ENCODING),
        normalized.encode(EXPECTED_KEY_ENCODING),
        hashlib.sha256,
    ).hexdigest()
    assert compute_raw_hash(normalized, FIRST_SECRET) == expected
    assert len(expected) == SHA256_HEX_LENGTH


def test_hash_is_not_plain_sha256() -> None:
    """The secret is mixed in, so the digest differs from an unkeyed SHA-256."""
    normalized = normalize("Senior Python Developer")
    plain = hashlib.sha256(normalized.encode(EXPECTED_KEY_ENCODING)).hexdigest()
    assert compute_raw_hash(normalized, FIRST_SECRET) != plain


def test_different_secrets_give_different_hashes() -> None:
    """Without the secret the hash cannot be reproduced."""
    normalized = normalize("Senior Python Developer")
    assert compute_raw_hash(normalized, FIRST_SECRET) != compute_raw_hash(
        normalized, SECOND_SECRET
    )


def test_short_hash_is_the_8_character_prefix() -> None:
    """Logs may carry only the first 8 characters of the raw hash."""
    raw_hash = hash_text("Senior Python Developer")
    assert len(short_hash(raw_hash)) == SHORT_HASH_LENGTH == 8
    assert raw_hash.startswith(short_hash(raw_hash))
