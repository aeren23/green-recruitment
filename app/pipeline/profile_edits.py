"""The candidate's corrections to a parsed profile (SPEC 6.1, confirmation step).

Edits re-scrub contact data from section texts, drop sections that normalize
to empty, and sync the sections by `text_hash` so that unchanged sections keep
their row id and embedding (SC-03). Sections an edit removes are deleted only
through retention.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.ingest.contact import ExtractedContact, remove_contact
from app.ingest.headings import fold_heading
from app.ingest.languages import LANGUAGE_LOOKUP
from app.ingest.normalize import normalize
from app.ingest.sections import ParsedSection, build_profile_json
from app.models import CandidateContact, CandidateProfile, Section
from app.pipeline.context import ContactEdit, PipelineContext, ProfileEdits, SectionEdit
from app.pipeline.shared import load_profile_sections, retention_context
from app.privacy.retention import discard_replaced_sections
from app.scoring.embed import compute_text_hash

SECTIONS_ATTRIBUTE = "sections"


@dataclass(frozen=True)
class SectionSync:
    """The sections of a profile after an edit, and whether anything changed."""

    sections: list[Section]
    changed: bool


def apply_edits(
    context: PipelineContext, profile: CandidateProfile, edits: ProfileEdits | None
) -> bool:
    """Apply the candidate's corrections to the profile and mark it confirmed.

    Returns:
        True when the set of sections or a section type changed.
    """
    sections_changed = False
    if edits is not None:
        sections_changed = store_edits(context, profile, edits)
    profile.confirmed_at = context.now
    profile.last_used_at = context.now
    return sections_changed


def store_edits(
    context: PipelineContext, profile: CandidateProfile, edits: ProfileEdits
) -> bool:
    """Write edited fields, contact and sections; rebuild `profile_json`."""
    contact = effective_contact(profile, edits.contact)
    store_contact(context.session, profile, edits.contact)
    entries = scrubbed_entries(edits.sections, contact)
    sync = sync_sections(context, profile, entries)
    profile.years_experience = edits.years_experience
    profile.languages = normalize_languages(edits.languages)
    profile.location = normalize_location(edits.location)
    rebuild_profile_json(profile, sync.sections)
    context.session.flush()
    context.session.expire(profile, [SECTIONS_ATTRIBUTE])
    return sync.changed


def effective_contact(
    profile: CandidateProfile, edit: ContactEdit | None
) -> ExtractedContact:
    """The contact whose values must not appear in any section text."""
    source = edit if edit is not None else profile.contact
    if source is None:
        return ExtractedContact()
    return ExtractedContact(name=source.name, email=source.email, phone=source.phone)


def store_contact(
    session: Session, profile: CandidateProfile, edit: ContactEdit | None
) -> None:
    """Replace the stored contact details with the candidate's edit, if any."""
    if edit is None:
        return
    if profile.contact is None:
        profile.contact = CandidateContact(profile_id=profile.id)
        session.add(profile.contact)
    profile.contact.name = edit.name
    profile.contact.email = edit.email
    profile.contact.phone = edit.phone


def scrubbed_entries(
    sections: Sequence[SectionEdit], contact: ExtractedContact
) -> list[SectionEdit]:
    """Re-scrub contact data from edited texts and drop sections left empty."""
    scrubbed = [
        SectionEdit(entry.type, remove_contact(entry.text, contact).strip())
        for entry in sections
    ]
    return [entry for entry in scrubbed if normalize(entry.text)]


def sync_sections(
    context: PipelineContext, profile: CandidateProfile, entries: Sequence[SectionEdit]
) -> SectionSync:
    """Match edited sections to stored ones by `text_hash` (SC-03).

    A matching section keeps its row, id and embedding. New texts become new
    rows; stored sections nothing matches are discarded through retention.
    """
    stored = load_profile_sections(context.session, profile.id)
    pool = StoredSectionPool.of(stored, context.embedder.model_version)
    kept: list[Section] = []
    changed = False
    for ordinal, entry in enumerate(entries):
        section, section_changed = reuse_or_create(profile, pool.claim(entry), entry)
        section.ordinal = ordinal
        context.session.add(section)
        kept.append(section)
        changed = changed or section_changed
    leftovers = pool.unclaimed()
    if leftovers:
        discard_replaced_sections(retention_context(context), leftovers)
    return SectionSync(kept, changed or bool(leftovers))


def reuse_or_create(
    profile: CandidateProfile, claimed: Section | None, entry: SectionEdit
) -> tuple[Section, bool]:
    """The section row for an edited entry, and whether it differs from before."""
    if claimed is None:
        return Section(profile=profile, type=entry.type, ordinal=0, text=entry.text), True
    changed = claimed.type is not entry.type
    claimed.type = entry.type
    return claimed, changed


@dataclass
class StoredSectionPool:
    """Stored sections of a profile, waiting to be claimed by edited entries."""

    model_version: str
    groups: dict[str | None, list[Section]]

    @classmethod
    def of(cls, sections: Sequence[Section], model_version: str) -> "StoredSectionPool":
        """Group stored sections by the `text_hash` their text would have."""
        groups: dict[str | None, list[Section]] = {}
        for section in sections:
            key = section_hash(model_version, section.text)
            groups.setdefault(key, []).append(section)
        return cls(model_version, groups)

    def claim(self, entry: SectionEdit) -> Section | None:
        """Take the stored section with the same text hash as the entry, if any."""
        candidates = self.groups.get(section_hash(self.model_version, entry.text))
        return candidates.pop(0) if candidates else None

    def unclaimed(self) -> list[Section]:
        """Stored sections no edited entry claimed."""
        return [section for group in self.groups.values() for section in group]


def section_hash(model_version: str, text: str) -> str | None:
    """The `text_hash` of a text; `None` when nothing is left after normalization."""
    return compute_text_hash(model_version, text) if normalize(text) else None


def normalize_languages(names: Sequence[str]) -> list[str]:
    """Lowercase English language names, deduplicated; Turkish names are mapped."""
    folded = (fold_heading(name) for name in names)
    mapped = (LANGUAGE_LOOKUP.get(name, name) for name in folded if name)
    return list(dict.fromkeys(mapped))


def normalize_location(location: str | None) -> str | None:
    """The folded location string the hard filter compares, `None` when empty."""
    if location is None:
        return None
    return fold_heading(location) or None


def rebuild_profile_json(profile: CandidateProfile, sections: Sequence[Section]) -> None:
    """Rewrite `profile_json` from the edited fields (SPEC 6.3 schema)."""
    parsed = [ParsedSection(section.type, section.ordinal, section.text) for section in sections]
    payload = build_profile_json(parsed, profile.years_experience, list(profile.languages))
    payload["location"] = profile.location
    profile.profile_json = payload
