"""Embedding and scoring of an application against its posting (SPEC 6.5).

`embed_and_score` runs the `embed` and `score` stages for one profile's
applications: the one being confirmed and, after an edit that changed the
sections, its other applications (SC-03). The matrix itself is the pure
function in `app.scoring.matrix`; this module only builds its inputs and
turns its result into `matrix_json`.
"""

from collections.abc import Sequence

import numpy as np

from app.models import Application, Requirement, Section, SectionType
from app.pipeline.context import STAGE_EMBED, STAGE_SCORE, PipelineContext
from app.pipeline.shared import load_profile_sections, load_requirements
from app.scoring.embed import blob_to_embedding, ensure_section_embeddings
from app.scoring.matrix import NO_SECTION_INDEX, MatrixInput, MatrixResult, score_matrix
from app.scoring.section_roles import is_evidence_section

MATRIX_REQUIREMENT_ID = "requirement_id"
MATRIX_SECTION_ID = "section_id"
MATRIX_SECTION_TYPE = "section_type"
MATRIX_SIMILARITY = "sim"
MATRIX_ADJUSTED = "adjusted"
MATRIX_EVIDENCE = "evidence"


def embed_and_score(
    context: PipelineContext, profile_id: int, targets: Sequence[Application]
) -> None:
    """Stages `embed` and `score` for the targets, which share one profile."""
    if not targets:
        return
    sections = load_profile_sections(context.session, profile_id)
    with context.tracker.stage(STAGE_EMBED):
        ensure_section_embeddings(context.session, sections, context.embedder)
    with context.tracker.stage(STAGE_SCORE):
        for target in targets:
            score_application(context, target, sections)


def invalidate_summaries(applications: Sequence[Application]) -> None:
    """Forget the summaries of re-scored applications (SC-03).

    A summary was written from the old sections; the top-k check that follows
    writes a fresh one if the application is still in its posting's top-k.
    """
    for application in applications:
        application.summary = None


def score_application(
    context: PipelineContext, application: Application, sections: Sequence[Section]
) -> None:
    """Write the raw score and `matrix_json` of one application (SPEC 6.5)."""
    requirements = load_requirements(context.session, application.job_id)
    coefficients = context.settings.section_type_coefficients
    result = score_matrix(build_matrix_input(requirements, sections, coefficients))
    application.score = result.score
    application.matrix_json = build_matrix_json(requirements, sections, result)


def build_matrix_input(
    requirements: Sequence[Requirement],
    sections: Sequence[Section],
    coefficients: dict[SectionType, float],
) -> MatrixInput:
    """Numpy arrays for the pure `score_matrix` function."""
    return MatrixInput(
        requirement_embeddings=stack_embeddings(requirements),
        section_embeddings=stack_embeddings(sections),
        requirement_weights=np.array([item.weight for item in requirements], dtype=float),
        section_type_coefficients=np.array(
            [coefficients[item.type] for item in sections], dtype=float
        ),
    )


def stack_embeddings(rows: Sequence[Requirement] | Sequence[Section]) -> np.ndarray:
    """One embedding per row; an empty list gives an empty array."""
    if not rows:
        return np.empty((0,))
    return np.stack([blob_to_embedding(row.embedding) for row in rows])


def build_matrix_json(
    requirements: Sequence[Requirement],
    sections: Sequence[Section],
    result: MatrixResult,
) -> list[dict[str, object]]:
    """One entry per requirement with its best section; empty without sections."""
    entries: list[dict[str, object]] = []
    for row, requirement in enumerate(requirements):
        best = int(result.best_section_indices[row])
        if best == NO_SECTION_INDEX:
            continue
        section = sections[best]
        entries.append(
            {
                MATRIX_REQUIREMENT_ID: requirement.id,
                MATRIX_SECTION_ID: section.id,
                MATRIX_SECTION_TYPE: section.type.value,
                MATRIX_SIMILARITY: float(result.similarities[row, best]),
                MATRIX_ADJUSTED: float(result.adjusted_similarities[row, best]),
                MATRIX_EVIDENCE: is_evidence_section(section.type),
            }
        )
    return entries
