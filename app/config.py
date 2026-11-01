"""Application settings loaded from environment variables and the `.env` file."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_NAME = ".env"


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
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    top_k: int = 5
    max_cv_chars: int = 6000
    max_upload_bytes: int = 1048576
    quality_k: int = 10
    draft_ttl_hours: int = 24
    decision_retention_days: int = 30
    shortlist_retention_days: int = 30
    # Kept as the raw string for now; a later task parses it into weights.
    section_alpha: str = (
        "experience:1.0,project:1.0,certification:0.8,"
        "education:0.8,summary:0.7,skills:0.6"
    )
    country_iso: str = "TUR"
    green_tracking: bool = True
