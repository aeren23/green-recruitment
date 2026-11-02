"""Helpers for the employer review tests: ranked candidates with planted contact data.

Everything goes through the real candidate flow (upload, preview, confirm) with
the controlled embedder, so the stored rows are exactly what production writes.
"""

from dataclasses import dataclass

from candidate_support import (
    confirm_form,
    input_value_of,
    section_texts_of,
    selected_types_of,
    token_of,
    upload_cv,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pdf_fixtures import build_pdf
from sqlalchemy import select

from app.models import Application, CandidateContact

CONTACT_FIELDS = ("contact_name", "contact_email", "contact_phone")
COMPLETE_PROFILE = {"years_experience": "5", "languages": "English", "location": "Denizli"}
INCOMPLETE_PROFILE = {"years_experience": "", "languages": "", "location": ""}
REDIRECT = 303
CONFLICT = 409
NOT_FOUND = 404


@dataclass(frozen=True)
class Person:
    """A fictional candidate with unique, searchable contact data."""

    name: str
    email: str
    phone: str


def person_number(number: int) -> Person:
    """A unique person: the name, e-mail and phone all carry the number.

    The name uses letters for the digits, because the first-line name
    heuristic of the parser rejects a line with digits.
    """
    suffix = "".join(chr(ord("a") + int(digit)) for digit in str(number))
    return Person(
        name=f"Zeynep Kaplanoglu{suffix}",
        email=f"zeynep.kaplanoglu{number}@example.org",
        phone=f"+90 555 987 65 {number:02d}",
    )


def build_person_cv(
    person: Person, tag: str, quality: int, experience_line: str = "", in_skills: bool = False
) -> bytes:
    """A CV whose best match sits in experience, or in skills when `in_skills`."""
    marker = f"QSCORE{quality}"
    experience_marker = "" if in_skills else marker
    skills_marker = marker if in_skills else ""
    lines = [
        person.name,
        f"{person.email} {person.phone}",
        "SUMMARY",
        f"Candidate profile {tag} seeking backend engineering roles.",
        "EXPERIENCE",
        "Software Engineer, Example Corp, 2016 - 2024",
        f"Built services {experience_line} {experience_marker} EXPERIENCETEXTMARKER",
        "SKILLS",
        f"Python SQL Docker {skills_marker}",
    ]
    return build_pdf(lines)


def confirm_unchanged(client: TestClient, token: str, **overrides: str) -> None:
    """Confirm a draft with the preview's own sections and contact, plus overrides."""
    page = client.get(f"/drafts/{token}").text
    rows = zip(selected_types_of(page), section_texts_of(page), strict=True)
    kept = [(row_type, text) for row_type, text in rows if text]
    fields = {name: input_value_of(page, name) for name in CONTACT_FIELDS}
    fields.update(COMPLETE_PROFILE)
    fields.update(overrides)
    response = client.post(f"/drafts/{token}", data=confirm_form(kept, **fields))
    assert response.status_code == REDIRECT


def rank_candidate(
    client: TestClient, application: FastAPI, job_id: int, pdf: bytes, **overrides: str
) -> int:
    """Upload and confirm one CV; return the application id."""
    token = token_of(upload_cv(client, job_id, pdf))
    confirm_unchanged(client, token, **overrides)
    with application.state.session_factory() as session:
        return session.scalar(
            select(Application.id).where(Application.draft_token == token)
        )


def stored_contact_values(application: FastAPI, application_id: int) -> list[str]:
    """The stored name, e-mail and phone of an application's candidate."""
    with application.state.session_factory() as session:
        profile_id = session.get(Application, application_id).profile_id
        contact = session.scalar(
            select(CandidateContact).where(CandidateContact.profile_id == profile_id)
        )
        return [contact.name, contact.email, contact.phone]
