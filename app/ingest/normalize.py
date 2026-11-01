"""Text normalization and keyed hashing (SPEC section 6.2).

`normalize` exists only to make the hash stable. Section splitting works on the
raw extracted text, never on the normalized one.
"""

import hashlib
import hmac
import unicodedata

UNICODE_NORMAL_FORM = "NFKC"
PUNCTUATION_CATEGORY_PREFIX = "P"
# `#` is category Po and would be removed without this entry. `+` is category
# Sm (a symbol, not punctuation) and is never removed; it is listed only
# defensively, so C++ keeps working if the punctuation rule is ever widened.
PRESERVED_PUNCTUATION = frozenset("+#")
WORD_SEPARATOR = " "
HASH_TEXT_ENCODING = "utf-8"
SHORT_HASH_LENGTH = 8


def is_replaceable_punctuation(character: str) -> bool:
    """True for punctuation that becomes a space.

    Every Unicode `P*` category counts, except `#`, so that `C#` stays
    distinguishable from `C`. `+` is a symbol (Sm), not punctuation, and is
    never replaced; it is in `PRESERVED_PUNCTUATION` only defensively.
    """
    if character in PRESERVED_PUNCTUATION:
        return False
    return unicodedata.category(character).startswith(PUNCTUATION_CATEGORY_PREFIX)


def normalize(text: str) -> str:
    """Normalize text for hashing.

    Applies NFKC, casefold, punctuation to spaces (keeping `+` and `#`), then
    collapses all whitespace runs to one space and strips the ends.

    Args:
        text: Raw extracted text.

    Returns:
        The normalized text. Empty for empty or whitespace-only input.
    """
    folded = unicodedata.normalize(UNICODE_NORMAL_FORM, text).casefold()
    spaced = "".join(
        WORD_SEPARATOR if is_replaceable_punctuation(character) else character
        for character in folded
    )
    return WORD_SEPARATOR.join(spaced.split())


def compute_raw_hash(normalized_text: str, secret: str) -> str:
    """Return the keyed HMAC-SHA256 hex digest of normalized text.

    Args:
        normalized_text: Output of `normalize`.
        secret: The hashing key (`HASH_SECRET`), injected by the caller.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hmac.new(
        secret.encode(HASH_TEXT_ENCODING),
        normalized_text.encode(HASH_TEXT_ENCODING),
        hashlib.sha256,
    ).hexdigest()


def short_hash(raw_hash: str) -> str:
    """Return the log-safe prefix of a raw hash (the only form logs may carry)."""
    return raw_hash[:SHORT_HASH_LENGTH]
