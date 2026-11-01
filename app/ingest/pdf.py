"""In-memory PDF text extraction.

Uploaded PDFs are never written to disk (Green rule 8), so extraction reads
straight from bytes. Error messages here are fixed strings: they must never
carry extracted CV text (KVKK).
"""

import io
import logging

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MIN_EXTRACTED_CHARS = 200
PAGE_SEPARATOR = "\n"

UNREADABLE_MESSAGE = "The PDF is corrupt, empty or could not be parsed."
ENCRYPTED_MESSAGE = "The PDF is encrypted and cannot be read."
TOO_SHORT_MESSAGE_TEMPLATE = (
    "The PDF yielded {found} characters of text, fewer than the required "
    "{required}. Scanned (image-only) PDFs are not supported."
)

PYPDF_LOGGER_NAME = "pypdf"
LOG_LEVEL_ABOVE_CRITICAL = logging.CRITICAL + 1

# pypdf does not wrap every parsing failure in its own error hierarchy. Fuzzing
# truncated and byte-flipped PDFs (pypdf 6.11.0) also surfaced these built-in
# types, so they are caught explicitly instead of using a blanket handler.
MALFORMED_PDF_ERRORS = (
    PyPdfError,
    KeyError,
    AttributeError,
    ValueError,
    TypeError,
    IndexError,
)


class PdfIngestError(Exception):
    """Base class for every error raised while ingesting a PDF."""


class UnreadablePdfError(PdfIngestError):
    """The input is corrupt, empty or encrypted."""


class TextTooShortError(PdfIngestError):
    """The PDF holds too little extractable text (for example a scan)."""


def silence_pypdf_logging() -> None:
    """Stop pypdf from emitting any log record in this process.

    KVKK: pypdf quotes the document's own bytes in its log messages (for
    example `Invalid Elementary Object ... stream BT ...`), so a log record
    from it can carry CV text. Logs may contain only ids and short hashes.
    """
    pypdf_logger = logging.getLogger(PYPDF_LOGGER_NAME)
    pypdf_logger.setLevel(LOG_LEVEL_ABOVE_CRITICAL)
    pypdf_logger.propagate = False


silence_pypdf_logging()


def read_all_pages(pdf_bytes: bytes) -> str:
    """Extract the text of every page, joined by newlines.

    Args:
        pdf_bytes: The complete PDF file.

    Raises:
        UnreadablePdfError: If the file is corrupt, empty or encrypted. The
            error carries no `__cause__` and no `__context__`.
    """
    text = ""
    failure_message: str | None = None
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            failure_message = ENCRYPTED_MESSAGE
        else:
            text = PAGE_SEPARATOR.join(page.extract_text() for page in reader.pages)
    except MALFORMED_PDF_ERRORS:
        failure_message = UNREADABLE_MESSAGE
    # KVKK: raised outside the `except` block on purpose. pypdf messages can
    # quote CV text, and raising inside the block would keep the original
    # exception reachable through `__context__` (`from None` still sets it),
    # so it would show up in any traceback.
    if failure_message is not None:
        raise UnreadablePdfError(failure_message)
    return text


def extract_text(pdf_bytes: bytes) -> str:
    """Extract text from an in-memory PDF, keeping its line structure.

    No normalization happens here: the section splitter matches headings at
    line starts, so the line breaks must survive.

    Args:
        pdf_bytes: The complete PDF file.

    Returns:
        The extracted text, one page after another.

    Raises:
        UnreadablePdfError: If the file is corrupt, empty or encrypted.
        TextTooShortError: If fewer than `MIN_EXTRACTED_CHARS` characters
            remain after stripping.
    """
    text = read_all_pages(pdf_bytes)
    found = len(text.strip())
    if found < MIN_EXTRACTED_CHARS:
        raise TextTooShortError(
            TOO_SHORT_MESSAGE_TEMPLATE.format(
                found=found, required=MIN_EXTRACTED_CHARS
            )
        )
    return text
