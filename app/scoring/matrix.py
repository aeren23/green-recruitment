"""Pure-numpy requirement-section scoring (SPEC 6.5).

This module knows nothing about the database, the embedding model or section
types. Callers pass arrays in and get a result back.

Definitions, with the SPEC letters in parentheses::

    similarities = requirement_embeddings @ section_embeddings.T          (sim)
    adjusted     = similarities * section_type_coefficients[None, :]      (adj)
    best_section = argmax(adjusted, axis=1)
    score        = sum(w_i * adjusted[i, best_section[i]]) / sum(w_i)

Embeddings are expected to be L2-normalized, so the inner product is the cosine
similarity. Ties between sections go to the lowest index (numpy `argmax`).
"""

from dataclasses import dataclass

import numpy as np

NO_SECTION_INDEX = -1
EMPTY_SECTIONS_SCORE = 0.0
COMPUTATION_DTYPE = np.float64
MATRIX_DIMENSIONS = 2
VECTOR_DIMENSIONS = 1


@dataclass(frozen=True)
class MatrixInput:
    """Arrays describing one posting against one candidate.

    Attributes:
        requirement_embeddings: Shape (requirement_count, dimension).
        section_embeddings: Shape (section_count, dimension). With no sections
            this may be shape (0, dimension) or (0,).
        requirement_weights: Shape (requirement_count,), non-negative, sum above 0.
        section_type_coefficients: Shape (section_count,), one per section.
    """

    requirement_embeddings: np.ndarray
    section_embeddings: np.ndarray
    requirement_weights: np.ndarray
    section_type_coefficients: np.ndarray


@dataclass(frozen=True)
class MatrixResult:
    """Outcome of `score_matrix`.

    Attributes:
        score: The raw weighted score; 0.0 when there are no sections.
        best_section_indices: Shape (requirement_count,), index of the best
            section per requirement; `NO_SECTION_INDEX` for every requirement
            when there are no sections.
        similarities: Shape (requirement_count, section_count).
        adjusted_similarities: Same shape, after the section-type coefficients.
    """

    score: float
    best_section_indices: np.ndarray
    similarities: np.ndarray
    adjusted_similarities: np.ndarray


def _as_array(values: np.ndarray) -> np.ndarray:
    """Float64 array of the input; the input itself is never modified."""
    return np.asarray(values, dtype=COMPUTATION_DTYPE)


def _check_requirements(
    requirement_embeddings: np.ndarray, requirement_weights: np.ndarray
) -> None:
    """Reject missing requirements and bad or all-zero weights."""
    is_matrix = requirement_embeddings.ndim == MATRIX_DIMENSIONS
    if not is_matrix or requirement_embeddings.shape[0] == 0:
        raise ValueError(
            "At least one requirement is needed (2-D requirement_embeddings)."
        )
    expected_shape = (requirement_embeddings.shape[0],)
    if requirement_weights.shape != expected_shape:
        raise ValueError(
            f"requirement_weights must have shape {expected_shape}, "
            f"got {requirement_weights.shape}."
        )
    if not np.all(np.isfinite(requirement_weights)) or np.any(requirement_weights < 0):
        raise ValueError("requirement_weights must be finite and non-negative.")
    if requirement_weights.sum() == 0:
        raise ValueError("requirement_weights must not sum to zero.")


def _count_sections(section_embeddings: np.ndarray, dimension: int) -> int:
    """Number of sections, after checking the embedding shape against `dimension`."""
    if section_embeddings.ndim == VECTOR_DIMENSIONS and section_embeddings.size == 0:
        return 0
    if section_embeddings.ndim != MATRIX_DIMENSIONS:
        raise ValueError(
            "section_embeddings must be a 2-D array (sections x dimension), "
            f"got shape {section_embeddings.shape}."
        )
    section_count, section_dimension = section_embeddings.shape
    if section_count > 0 and section_dimension != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: requirements have {dimension}, "
            f"sections have {section_dimension}."
        )
    return section_count


def _check_coefficients_and_numbers(
    matrix_input: MatrixInput, section_count: int
) -> None:
    """Reject coefficient arrays of the wrong length and non-finite numbers."""
    coefficients = matrix_input.section_type_coefficients
    expected_shape = (section_count,)
    if coefficients.shape != expected_shape:
        raise ValueError(
            f"section_type_coefficients must have shape {expected_shape}, "
            f"got {coefficients.shape}."
        )
    named_arrays = (
        ("requirement_embeddings", matrix_input.requirement_embeddings),
        ("section_embeddings", matrix_input.section_embeddings),
        ("section_type_coefficients", coefficients),
    )
    for name, values in named_arrays:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite numbers.")


def _empty_result(requirement_count: int) -> MatrixResult:
    """Result for a candidate without sections: score 0 and empty matrices."""
    empty_matrix_shape = (requirement_count, 0)
    return MatrixResult(
        score=EMPTY_SECTIONS_SCORE,
        best_section_indices=np.full(
            requirement_count, NO_SECTION_INDEX, dtype=np.intp
        ),
        similarities=np.zeros(empty_matrix_shape, dtype=COMPUTATION_DTYPE),
        adjusted_similarities=np.zeros(empty_matrix_shape, dtype=COMPUTATION_DTYPE),
    )


def _validated_copy(matrix_input: MatrixInput) -> tuple[MatrixInput, int]:
    """Float64 arrays of the input, validated, with the number of sections."""
    arrays = MatrixInput(
        requirement_embeddings=_as_array(matrix_input.requirement_embeddings),
        section_embeddings=_as_array(matrix_input.section_embeddings),
        requirement_weights=_as_array(matrix_input.requirement_weights),
        section_type_coefficients=_as_array(matrix_input.section_type_coefficients),
    )
    _check_requirements(arrays.requirement_embeddings, arrays.requirement_weights)
    dimension = arrays.requirement_embeddings.shape[1]
    section_count = _count_sections(arrays.section_embeddings, dimension)
    _check_coefficients_and_numbers(arrays, section_count)
    return arrays, section_count


def score_matrix(matrix_input: MatrixInput) -> MatrixResult:
    """Score one candidate against one posting.

    Args:
        matrix_input: The four arrays, see `MatrixInput`.

    Returns:
        The raw score and the intermediate matrices. The inputs are not modified.

    Raises:
        ValueError: On mismatched dimensions or lengths, no requirements, or
            weights that are invalid or sum to zero.
    """
    arrays, section_count = _validated_copy(matrix_input)
    requirement_count = arrays.requirement_embeddings.shape[0]
    if section_count == 0:
        return _empty_result(requirement_count)
    similarities = arrays.requirement_embeddings @ arrays.section_embeddings.T
    adjusted = similarities * arrays.section_type_coefficients[np.newaxis, :]
    best_indices = np.argmax(adjusted, axis=1)
    best_values = adjusted[np.arange(requirement_count), best_indices]
    weights = arrays.requirement_weights
    score = float(np.dot(weights, best_values) / weights.sum())
    return MatrixResult(score, best_indices, similarities, adjusted)
