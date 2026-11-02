"""View models of the employer pages (T-15): anonymous rows, matrix and contact.

Queries and decisions about *what may be shown* live here, not in the routes.
Contact data is exposed by exactly one function, `visible_contact`, and only
for a shortlisted application (KVKK).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Application,
    ApplicationStatus,
    CandidateContact,
    CandidateProfile,
    JobPosting,
    Requirement,
    Section,
    SectionType,
)
from app.pipeline.ranking import ScoreBounds
from app.pipeline.scoring import (
    MATRIX_ADJUSTED,
    MATRIX_REQUIREMENT_ID,
    MATRIX_SECTION_ID,
    MATRIX_SECTION_TYPE,
    MATRIX_SIMILARITY,
)
from app.pipeline.shared import load_profile_sections
from app.scoring.section_roles import is_claim_section, is_evidence_section
from app.web import messages

MATRIX_DECIMALS = 3


@dataclass(frozen=True)
class RankedRowView:
    """One anonymous row of the ranked list."""

    application_id: int
    label: str
    relative_score: int
    missing_information: bool
    shortlisted: bool


@dataclass(frozen=True)
class MatrixRowView:
    """One requirement with the section that matched it best."""

    requirement_text: str
    weight: int
    section_type_label: str | None
    marker: str | None
    similarity: float | None
    adjusted: float | None
    section_text: str | None


@dataclass(frozen=True)
class ContactView:
    """Contact data of a shortlisted candidate."""

    name: str | None
    email: str | None
    phone: str | None


@dataclass(frozen=True)
class ApplicationDetailView:
    """Everything the application detail page shows."""

    application_id: int
    job_id: int
    label: str
    status: ApplicationStatus
    relative_score: int
    summary: str | None
    matrix: list[MatrixRowView]
    contact: ContactView | None
    can_shortlist: bool


def candidate_label(application_id: int) -> str:
    """The anonymous label of a candidate."""
    return messages.CANDIDATE_LABEL.format(application_id=application_id)


def profile_is_incomplete(profile: CandidateProfile) -> bool:
    """True when years of experience, languages or location is unknown."""
    return profile.years_experience is None or not profile.languages or not profile.location


def build_ranked_rows(
    applications: Sequence[Application], bounds: ScoreBounds | None
) -> list[RankedRowView]:
    """Anonymous rows for one page of ranked applications."""
    return [
        RankedRowView(
            application_id=application.id,
            label=candidate_label(application.id),
            relative_score=_relative_percent(application, bounds),
            missing_information=profile_is_incomplete(application.profile),
            shortlisted=application.status is ApplicationStatus.SHORTLISTED,
        )
        for application in applications
    ]


def posting_filter_lines(posting: JobPosting) -> list[tuple[str, str]]:
    """Label and value of each hard filter the posting sets."""
    lines: list[tuple[str, str]] = []
    if posting.required_languages:
        names = [messages.LANGUAGE_LABELS.get(name, name) for name in posting.required_languages]
        lines.append((messages.FILTER_LANGUAGES, ", ".join(names)))
    if posting.location:
        lines.append((messages.FILTER_LOCATION, posting.location))
        remote = messages.FILTER_REMOTE_YES if posting.remote_ok else messages.FILTER_REMOTE_NO
        lines.append((messages.FILTER_REMOTE, remote))
    if posting.min_years is not None:
        years = messages.FILTER_MIN_YEARS_VALUE.format(years=f"{posting.min_years:g}")
        lines.append((messages.FILTER_MIN_YEARS, years))
    return lines


def visible_contact(session: Session, application: Application) -> ContactView | None:
    """Contact data, only for a shortlisted application (KVKK); otherwise `None`."""
    if application.status is not ApplicationStatus.SHORTLISTED:
        return None
    contact = session.scalar(
        select(CandidateContact).where(CandidateContact.profile_id == application.profile_id)
    )
    if contact is None:
        return ContactView(None, None, None)
    return ContactView(contact.name, contact.email, contact.phone)


def build_application_detail(
    session: Session, application: Application, bounds: ScoreBounds | None
) -> ApplicationDetailView:
    """The detail page content of a ranked or shortlisted application."""
    return ApplicationDetailView(
        application_id=application.id,
        job_id=application.job_id,
        label=candidate_label(application.id),
        status=application.status,
        relative_score=_relative_percent(application, bounds),
        summary=application.summary,
        matrix=_matrix_rows(session, application),
        contact=visible_contact(session, application),
        can_shortlist=application.status is ApplicationStatus.RANKED,
    )


def _relative_percent(application: Application, bounds: ScoreBounds | None) -> int:
    """The relative score rounded to a whole number."""
    if bounds is None or application.score is None:
        return 0
    return round(bounds.relative(application.score))


def _matrix_rows(session: Session, application: Application) -> list[MatrixRowView]:
    """One row per requirement of the posting, in definition order."""
    requirements = session.scalars(
        select(Requirement)
        .where(Requirement.job_id == application.job_id)
        .order_by(Requirement.id)
    )
    entries = {
        entry[MATRIX_REQUIREMENT_ID]: entry for entry in application.matrix_json or []
    }
    sections = {
        section.id: section for section in load_profile_sections(session, application.profile_id)
    }
    return [_matrix_row(item, entries.get(item.id), sections) for item in requirements]


def _matrix_row(
    requirement: Requirement, entry: dict[str, Any] | None, sections: dict[int, Section]
) -> MatrixRowView:
    """A matrix row; a requirement without a match keeps its text and weight."""
    if entry is None:
        return MatrixRowView(requirement.text, requirement.weight, None, None, None, None, None)
    section_type = SectionType(entry[MATRIX_SECTION_TYPE])
    section = sections.get(entry[MATRIX_SECTION_ID])
    return MatrixRowView(
        requirement_text=requirement.text,
        weight=requirement.weight,
        section_type_label=messages.SECTION_TYPE_LABELS[section_type],
        marker=_marker_of(section_type),
        similarity=round(entry[MATRIX_SIMILARITY], MATRIX_DECIMALS),
        adjusted=round(entry[MATRIX_ADJUSTED], MATRIX_DECIMALS),
        section_text=section.text if section else None,
    )


def _marker_of(section_type: SectionType) -> str:
    """Evidence (work shown), claim (skills only) or other."""
    if is_evidence_section(section_type):
        return messages.MATCH_EVIDENCE
    if is_claim_section(section_type):
        return messages.MATCH_CLAIM
    return messages.MATCH_OTHER
