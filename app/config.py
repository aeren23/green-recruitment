"""Application settings loaded from environment variables and the `.env` file."""

import math
from datetime import date
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import SectionType

ENV_FILE_NAME = ".env"
DEFAULT_BENCH_CV_REFERENCE_DATE = date(2018, 1, 1)
SECTION_ALPHA_ENTRY_SEPARATOR = ","
SECTION_ALPHA_KEY_VALUE_SEPARATOR = ":"


def _parse_alpha_entry(entry: str) -> tuple[SectionType, float]:
    """Parse one `type:value` entry of `SECTION_ALPHA`."""
    parts = entry.split(SECTION_ALPHA_KEY_VALUE_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(
            f"SECTION_ALPHA entry {entry!r} must look like 'type:value'."
        )
    type_name, raw_value = (part.strip() for part in parts)
    try:
        section_type = SectionType(type_name)
    except ValueError:
        known_types = ", ".join(member.value for member in SectionType)
        raise ValueError(
            f"SECTION_ALPHA has unknown section type {type_name!r} "
            f"(known types: {known_types})."
        ) from None
    try:
        coefficient = float(raw_value)
    except ValueError:
        raise ValueError(
            f"SECTION_ALPHA value {raw_value!r} for {type_name!r} is not a number."
        ) from None
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError(
            f"SECTION_ALPHA value for {type_name!r} must be finite and non-negative."
        )
    return section_type, coefficient


def parse_section_alpha(raw_value: str) -> dict[SectionType, float]:
    """Parse the `SECTION_ALPHA` setting into coefficients keyed by section type.

    Args:
        raw_value: Comma-separated `type:value` pairs, for example
            `experience:1.0,skills:0.6`. Every section type must appear exactly
            once, so the scorer never meets a section without a coefficient.

    Returns:
        A mapping from each `SectionType` to its non-negative coefficient.

    Raises:
        ValueError: On a malformed entry, an unknown or duplicate section type,
            a non-numeric or negative value, or a missing section type.
    """
    coefficients: dict[SectionType, float] = {}
    for entry in raw_value.split(SECTION_ALPHA_ENTRY_SEPARATOR):
        section_type, coefficient = _parse_alpha_entry(entry)
        if section_type in coefficients:
            raise ValueError(f"SECTION_ALPHA lists {section_type.value!r} twice.")
        coefficients[section_type] = coefficient
    missing = [member.value for member in SectionType if member not in coefficients]
    if missing:
        raise ValueError(f"SECTION_ALPHA is missing section types: {', '.join(missing)}.")
    return coefficients


class Settings(BaseSettings):
    """All runtime configuration described in SPEC section 10.

    `hash_secret` and `employer_token` have no defaults on purpose: a missing
    value must stop the application at start-up instead of running insecurely.
    """

    model_config = SettingsConfigDict(env_file=ENV_FILE_NAME, extra="ignore")

    database_url: str = "sqlite:///data/app.db"
    hash_secret: str = Field(min_length=1)
    employer_token: str = Field(min_length=1)
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_device: Literal["cuda", "cpu"] = "cuda"
    # When true the embedding model is read from the local Hugging Face cache only,
    # so start-up never depends on the network.
    embed_local_files_only: bool = True
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    top_k: int = 5
    max_cv_chars: int = 6000
    max_upload_bytes: int = 1048576
    quality_k: int = 10
    draft_ttl_hours: int = 24
    decision_retention_days: int = 30
    shortlist_retention_days: int = 30
    # Kept as the raw string (validated at start-up); read the parsed mapping
    # through `section_type_coefficients`.
    section_alpha: str = (
        "experience:1.0,project:1.0,certification:0.8,"
        "education:0.8,summary:0.7,skills:0.6"
    )
    country_iso: str = "TUR"
    green_tracking: bool = True
    # Fixed parse date for the benchmark (SPEC 6.3, 7.3); the dataset predates it.
    bench_cv_reference_date: date = DEFAULT_BENCH_CV_REFERENCE_DATE

    @field_validator("section_alpha")
    @classmethod
    def validate_section_alpha(cls, raw_value: str) -> str:
        """Stop the application at start-up when `SECTION_ALPHA` is malformed."""
        parse_section_alpha(raw_value)
        return raw_value

    @property
    def section_type_coefficients(self) -> dict[SectionType, float]:
        """The parsed `section_alpha`, keyed by section type."""
        return parse_section_alpha(self.section_alpha)
