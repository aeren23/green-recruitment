"""Parsing and validation of the `SECTION_ALPHA` setting."""

import pytest
from pydantic import ValidationError

from app.config import Settings, parse_section_alpha
from app.models import SectionType

SPEC_DEFAULT_COEFFICIENTS = {
    SectionType.EXPERIENCE: 1.0,
    SectionType.PROJECT: 1.0,
    SectionType.CERTIFICATION: 0.8,
    SectionType.EDUCATION: 0.8,
    SectionType.SUMMARY: 0.7,
    SectionType.SKILLS: 0.6,
}
VALID_ALPHA = (
    "experience:1.0,project:1.0,certification:0.8,"
    "education:0.8,summary:0.7,skills:0.6"
)
TEST_SECRET = "test-secret"


def make_settings(**overrides: str) -> Settings:
    """Settings that ignore any real `.env` file."""
    return Settings(
        _env_file=None, hash_secret=TEST_SECRET, employer_token=TEST_SECRET, **overrides
    )


def test_default_alpha_matches_the_spec() -> None:
    """The documented default parses into the SPEC 6.5 coefficients."""
    assert make_settings().section_type_coefficients == SPEC_DEFAULT_COEFFICIENTS


def test_parse_returns_every_section_type_as_a_float() -> None:
    """The mapping is keyed by the `SectionType` enum with float values."""
    coefficients = parse_section_alpha(VALID_ALPHA)
    assert set(coefficients) == set(SectionType)
    assert all(isinstance(value, float) for value in coefficients.values())


def test_parse_accepts_any_order_whitespace_and_integers() -> None:
    """Order and spaces do not matter, and `1` is as good as `1.0`."""
    raw_value = (
        " skills : 0.5 , summary:0.7,education:0.8,"
        "certification:0.8, project:1 ,experience:2"
    )
    coefficients = parse_section_alpha(raw_value)
    assert coefficients[SectionType.SKILLS] == 0.5
    assert coefficients[SectionType.PROJECT] == 1.0
    assert coefficients[SectionType.EXPERIENCE] == 2.0


@pytest.mark.parametrize(
    ("raw_value", "message"),
    [
        ("", "must look like"),
        (VALID_ALPHA + ",", "must look like"),
        ("experience", "must look like"),
        ("experience=1.0", "must look like"),
        ("experience:1.0:2.0", "must look like"),
        (VALID_ALPHA.replace("skills:0.6", "hobbies:0.6"), "unknown section type 'hobbies'"),
        (VALID_ALPHA.replace("skills:0.6", "languages:0.6"), "unknown section type 'languages'"),
        (VALID_ALPHA.replace("skills:0.6", "Skills:0.6"), "unknown section type 'Skills'"),
        (VALID_ALPHA.replace("skills:0.6", "skills:high"), "not a number"),
        (VALID_ALPHA.replace("skills:0.6", "skills:"), "not a number"),
        (VALID_ALPHA.replace("skills:0.6", "skills:-0.1"), "non-negative"),
        (VALID_ALPHA.replace("skills:0.6", "skills:nan"), "non-negative"),
        (VALID_ALPHA.replace("skills:0.6", "skills:inf"), "non-negative"),
        (VALID_ALPHA + ",skills:0.5", "twice"),
        ("experience:1.0,project:1.0", "missing section types"),
    ],
)
def test_malformed_alpha_raises_value_error(raw_value: str, message: str) -> None:
    """Malformed input, unknown types, bad numbers and gaps all fail fast."""
    with pytest.raises(ValueError, match=message):
        parse_section_alpha(raw_value)


def test_settings_reject_malformed_alpha_at_startup() -> None:
    """A bad `SECTION_ALPHA` stops settings construction, not a later scoring run."""
    with pytest.raises(ValidationError, match="unknown section type"):
        make_settings(section_alpha="hobbies:1.0")


def test_settings_accept_custom_alpha() -> None:
    """A valid custom value is kept and parsed."""
    settings = make_settings(section_alpha=VALID_ALPHA.replace("skills:0.6", "skills:0.1"))
    assert settings.section_type_coefficients[SectionType.SKILLS] == 0.1


def test_settings_read_alpha_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`SECTION_ALPHA` from the environment is validated like a direct argument."""
    monkeypatch.setenv("SECTION_ALPHA", "experience:1.0")
    with pytest.raises(ValidationError, match="missing section types"):
        Settings(_env_file=None, hash_secret=TEST_SECRET, employer_token=TEST_SECRET)
