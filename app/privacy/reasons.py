"""Reason codes of SPEC section 8.2.

The two enums encode the privacy rule that only some codes may ever be written
to the database: a `DecisionRecord` stores a `StoredReasonCode`, while a
`ResponseOnlyReasonCode` is shown in an HTTP response and never persisted.
"""

from enum import StrEnum


class StoredReasonCode(StrEnum):
    """Reason codes that a `DecisionRecord` may hold (SPEC 8.2)."""

    LANG_MISSING = "LANG_MISSING"
    LOCATION_MISMATCH = "LOCATION_MISMATCH"
    MIN_YEARS = "MIN_YEARS"
    NOT_SELECTED = "NOT_SELECTED"


class ResponseOnlyReasonCode(StrEnum):
    """Reason codes that only appear in a response and are never stored (SPEC 8.2)."""

    DUPLICATE_APPLICATION = "DUPLICATE_APPLICATION"
    DRAFT_EXPIRED = "DRAFT_EXPIRED"


STORED_REASON_CODE_VALUES = frozenset(code.value for code in StoredReasonCode)


def is_stored_reason_code(value: str) -> bool:
    """Whether a raw string is a code that may be written to a `DecisionRecord`.

    Args:
        value: The reason code string to check.

    Returns:
        True only for the stored codes; response-only and unknown strings give False.
    """
    return value in STORED_REASON_CODE_VALUES
