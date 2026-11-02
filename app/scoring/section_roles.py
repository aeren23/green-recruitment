"""Which section types count as evidence and which as claims (SPEC 6.5).

This lives apart from `matrix.py` on purpose: the matrix is pure numpy and must
not know about section types, while the callers that build `matrix_json` and the
coefficient array do. Only the `SectionType` enum is used, no database access.
"""

from app.models import SectionType

EVIDENCE_SECTION_TYPES = frozenset({SectionType.EXPERIENCE, SectionType.PROJECT})
CLAIM_SECTION_TYPES = frozenset({SectionType.SKILLS})


def is_evidence_section(section_type: SectionType) -> bool:
    """Whether the section shows work done (experience or project).

    Args:
        section_type: The type of the section.
    """
    return section_type in EVIDENCE_SECTION_TYPES


def is_claim_section(section_type: SectionType) -> bool:
    """Whether the section only states a skill without showing work (skills).

    Args:
        section_type: The type of the section.
    """
    return section_type in CLAIM_SECTION_TYPES
