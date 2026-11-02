"""Masking of candidate tokens in the access log (KVKK).

The token in `/drafts/<token>` and `/applications/<token>` is the candidate's
only credential: whoever holds it can read the contact details in the
preview. uvicorn writes the full request path to its access log, so a filter
on `uvicorn.access` replaces the token segment before any handler sees it.
"""

import logging
import re

ACCESS_LOGGER_NAME = "uvicorn.access"
TOKEN_PATH_PATTERN = re.compile(r"(/(?:drafts|applications)/)[^/?#\s\"']+")
TOKEN_REPLACEMENT = r"\1***"


def mask_tokens(text: str) -> str:
    """Replace the token segment of candidate paths inside a text."""
    return TOKEN_PATH_PATTERN.sub(TOKEN_REPLACEMENT, text)


def mask_value(value: object) -> object:
    """Mask a log argument when it is text; other values pass unchanged."""
    return mask_tokens(value) if isinstance(value, str) else value


class TokenMaskingFilter(logging.Filter):
    """Logging filter that masks candidate tokens in a record's message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask the record in place; the record is always kept."""
        record.msg = mask_value(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(mask_value(argument) for argument in record.args)
        return True


def install_access_log_masking() -> None:
    """Attach the masking filter to `uvicorn.access` once (idempotent)."""
    access_logger = logging.getLogger(ACCESS_LOGGER_NAME)
    if not any(isinstance(item, TokenMaskingFilter) for item in access_logger.filters):
        access_logger.addFilter(TokenMaskingFilter())
