"""Reason codes of SPEC 8.2: the stored and response-only sets must stay apart."""

import pytest

from app.privacy.reasons import (
    STORED_REASON_CODE_VALUES,
    ResponseOnlyReasonCode,
    StoredReasonCode,
    is_stored_reason_code,
)

SPEC_STORED_CODES = {"LANG_MISSING", "LOCATION_MISMATCH", "MIN_YEARS", "NOT_SELECTED"}
SPEC_RESPONSE_ONLY_CODES = {"DUPLICATE_APPLICATION", "DRAFT_EXPIRED"}


def test_stored_codes_match_the_spec() -> None:
    """Exactly the four codes of SPEC 8.2 may be stored."""
    assert {code.value for code in StoredReasonCode} == SPEC_STORED_CODES
    assert STORED_REASON_CODE_VALUES == SPEC_STORED_CODES


def test_response_only_codes_match_the_spec() -> None:
    """Exactly the two response-only codes of SPEC 8.2 exist."""
    assert {code.value for code in ResponseOnlyReasonCode} == SPEC_RESPONSE_ONLY_CODES


def test_stored_and_response_only_codes_do_not_overlap() -> None:
    """A response-only code can never be mistaken for a storable one."""
    assert STORED_REASON_CODE_VALUES.isdisjoint(SPEC_RESPONSE_ONLY_CODES)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MIN_YEARS", True),
        (StoredReasonCode.NOT_SELECTED, True),
        ("DUPLICATE_APPLICATION", False),
        (ResponseOnlyReasonCode.DRAFT_EXPIRED, False),
        ("SOMETHING_ELSE", False),
    ],
)
def test_is_stored_reason_code(value: str, expected: bool) -> None:
    """Only stored codes are accepted for a `DecisionRecord`."""
    assert is_stored_reason_code(value) is expected
