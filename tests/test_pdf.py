"""Tests for in-memory PDF text extraction (T-02)."""

import io
import logging
import random
import traceback
import warnings

import pytest
from pypdf import PdfReader, PdfWriter

from app.ingest.pdf import (
    MALFORMED_PDF_ERRORS,
    MIN_EXTRACTED_CHARS,
    PdfIngestError,
    TextTooShortError,
    UnreadablePdfError,
    extract_text,
)
from pdf_fixtures import SAMPLE_CV_LINES, PdfLayout, build_pdf

RANDOM_SEED = 20261101
RANDOM_BYTE_COUNT = 500
TRUNCATED_PREFIX_BYTES = 100
OWNER_PASSWORD = "owner-password"
NON_EMPTY_USER_PASSWORD = "user-password"
EMPTY_USER_PASSWORD = ""
RC4_ALGORITHM = "RC4-128"  # pure Python in pypdf, so no extra dependency
CV_TEXT_MARKER = "ZQXCONFIDENTIALMARKERZQX"
MARKER_LINE_COUNT = 8
FILLER_CHARACTER = "a"
FUZZ_ITERATIONS = 400
FUZZ_SEED = 42
MAX_BYTES_PER_MUTATION = 12
LOGGER_CAPTURE_LEVEL = logging.DEBUG
PYPDF_LOGGER_NAME = "pypdf"

def corrupt_one_byte(pdf_bytes: bytes, offset: int, replacement: int) -> bytes:
    """Return a copy of the PDF with a single byte replaced."""
    mutable = bytearray(pdf_bytes)
    mutable[offset] = replacement
    return bytes(mutable)


def encrypt_pdf(pdf_bytes: bytes, user_password: str) -> bytes:
    """Return an encrypted copy of the PDF built with pypdf's writer."""
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes)))
    writer.encrypt(user_password, OWNER_PASSWORD, algorithm=RC4_ALGORITHM)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def pdf_with_stripped_length(length: int) -> bytes:
    """Return a PDF whose extracted text is `length` characters once stripped."""
    return build_pdf([FILLER_CHARACTER * length])


def marker_cv_lines() -> list[str]:
    """Return long CV lines that all carry a recognisable marker."""
    return [
        f"{CV_TEXT_MARKER} line {number} of a confidential curriculum vitae"
        for number in range(MARKER_LINE_COUNT)
    ]


def test_sample_cv_is_long_enough_for_the_other_tests() -> None:
    """The shared sample must clear the minimum, or later tests prove nothing."""
    assert len("".join(SAMPLE_CV_LINES)) > MIN_EXTRACTED_CHARS


def test_fixture_round_trips_and_keeps_line_breaks() -> None:
    """Every fixture line comes back as its own line, in order."""
    text = extract_text(build_pdf(SAMPLE_CV_LINES))
    assert text.splitlines() == list(SAMPLE_CV_LINES)


def test_re_exported_fixture_has_different_bytes_but_same_text() -> None:
    """The layout variants change the bytes and leave the text alone."""
    original = build_pdf(SAMPLE_CV_LINES)
    re_exported = build_pdf(SAMPLE_CV_LINES, PdfLayout("Exported", True))
    assert original != re_exported
    assert extract_text(original) == extract_text(re_exported)


def test_text_in_parentheses_survives_escaping() -> None:
    """Special PDF string characters are escaped and read back unchanged."""
    lines = [r"Skills (Python) and C:\temp", *SAMPLE_CV_LINES]
    assert extract_text(build_pdf(lines)).splitlines() == lines


def test_empty_bytes_are_unreadable() -> None:
    """An empty upload is a typed error, not a pypdf one."""
    with pytest.raises(UnreadablePdfError):
        extract_text(b"")


def test_random_bytes_are_unreadable() -> None:
    """Bytes that are not a PDF at all are rejected."""
    garbage = random.Random(RANDOM_SEED).randbytes(RANDOM_BYTE_COUNT)
    with pytest.raises(UnreadablePdfError):
        extract_text(garbage)


@pytest.mark.parametrize("prefix_length", [TRUNCATED_PREFIX_BYTES, None])
def test_truncated_pdf_is_unreadable(prefix_length: int | None) -> None:
    """Cutting a valid PDF short (a fixed prefix, or in half) is rejected."""
    valid = build_pdf(SAMPLE_CV_LINES)
    cut_at = prefix_length if prefix_length is not None else len(valid) // 2
    with pytest.raises(UnreadablePdfError):
        extract_text(valid[:cut_at])


@pytest.mark.parametrize("user_password", [NON_EMPTY_USER_PASSWORD, EMPTY_USER_PASSWORD])
def test_encrypted_pdf_is_unreadable(user_password: str) -> None:
    """Encrypted files are rejected, even when pypdf could open them unaided."""
    encrypted = encrypt_pdf(build_pdf(SAMPLE_CV_LINES), user_password)
    with pytest.raises(UnreadablePdfError):
        extract_text(encrypted)


def test_both_error_types_share_a_base_class() -> None:
    """Callers can catch every ingest failure with one except clause."""
    assert issubclass(UnreadablePdfError, PdfIngestError)
    assert issubclass(TextTooShortError, PdfIngestError)


def test_pdf_without_text_is_too_short() -> None:
    """A valid PDF with no text (as a scan would be) is rejected as too short."""
    with pytest.raises(TextTooShortError):
        extract_text(build_pdf([]))


def test_text_of_199_characters_is_too_short() -> None:
    """One character below the minimum is rejected."""
    with pytest.raises(TextTooShortError):
        extract_text(pdf_with_stripped_length(MIN_EXTRACTED_CHARS - 1))


def test_text_of_exactly_200_characters_is_accepted() -> None:
    """The minimum itself is accepted."""
    text = extract_text(pdf_with_stripped_length(MIN_EXTRACTED_CHARS))
    assert len(text.strip()) == MIN_EXTRACTED_CHARS


def test_whitespace_does_not_count_towards_the_minimum() -> None:
    """Characters are counted after stripping, so blank padding cannot help."""
    padding = [" " * MIN_EXTRACTED_CHARS] * MARKER_LINE_COUNT
    short_text = [FILLER_CHARACTER * (MIN_EXTRACTED_CHARS - 1)]
    with pytest.raises(TextTooShortError):
        extract_text(build_pdf([*padding, *short_text, *padding]))


def test_too_short_error_message_has_no_cv_text() -> None:
    """The message states counts only, never the extracted text."""
    with pytest.raises(TextTooShortError) as raised:
        extract_text(build_pdf([CV_TEXT_MARKER]))
    assert CV_TEXT_MARKER not in str(raised.value)
    assert CV_TEXT_MARKER not in repr(raised.value)


class RecordCollector(logging.Handler):
    """Collects the fully formatted text of every log record it receives."""

    def __init__(self) -> None:
        super().__init__(level=LOGGER_CAPTURE_LEVEL)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the formatted message of the record."""
        self.messages.append(self.format(record))


def format_traceback(error: BaseException) -> str:
    """Return the full formatted traceback, including any chained errors."""
    return "".join(traceback.format_exception(error))


def mutate_randomly(pdf_bytes: bytes, generator: random.Random) -> bytes:
    """Overwrite a random number of random bytes of the PDF."""
    mutable = bytearray(pdf_bytes)
    for _ in range(generator.randint(1, MAX_BYTES_PER_MUTATION)):
        mutable[generator.randrange(len(mutable))] = generator.randrange(256)
    return bytes(mutable)


@pytest.mark.parametrize("error_type", MALFORMED_PDF_ERRORS)
def test_malformed_pdf_failure_leaves_no_chain_and_no_cv_text(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Every caught type is converted with no `__cause__`, no `__context__`."""

    def failing_reader(_stream: io.BytesIO) -> None:
        raise error_type(f"pypdf quoted: {CV_TEXT_MARKER}")

    monkeypatch.setattr("app.ingest.pdf.PdfReader", failing_reader)
    with pytest.raises(UnreadablePdfError) as raised:
        extract_text(b"%PDF-irrelevant")
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert CV_TEXT_MARKER not in format_traceback(raised.value)


def test_corrupted_pdfs_leak_no_cv_text_through_errors_logs_or_warnings() -> None:
    """Real byte-corrupted CVs: no exception, log record or warning has CV text."""
    valid = build_pdf(marker_cv_lines())
    generator = random.Random(FUZZ_SEED)
    collector = RecordCollector()
    watched_loggers = [logging.getLogger(PYPDF_LOGGER_NAME), logging.getLogger()]
    leaks: list[str] = []
    for watched_logger in watched_loggers:
        watched_logger.addHandler(collector)
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            for _ in range(FUZZ_ITERATIONS):
                leaks.extend(find_error_leaks(mutate_randomly(valid, generator)))
    finally:
        for watched_logger in watched_loggers:
            watched_logger.removeHandler(collector)
    leaks.extend(collector.messages)
    leaks.extend(str(caught.message) for caught in caught_warnings)
    assert [leak for leak in leaks if CV_TEXT_MARKER in leak] == []


def find_error_leaks(pdf_bytes: bytes) -> list[str]:
    """Run extraction and return the error texts it produced.

    Only `UnreadablePdfError` and `TextTooShortError` may escape; anything else
    propagates and fails the calling test.
    """
    try:
        extract_text(pdf_bytes)
    except (UnreadablePdfError, TextTooShortError) as error:
        return [str(error), repr(error), format_traceback(error)]
    return []
