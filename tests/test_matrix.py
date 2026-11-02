"""Hand-computed behaviour of `app/scoring/matrix.py` and `section_roles.py`.

All vectors are tiny and the expected values are worked out by hand in the
comments, so a wrong formula cannot hide behind a copy of itself.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from app.config import parse_section_alpha
from app.models import SectionType
from app.scoring import matrix
from app.scoring.matrix import (
    NO_SECTION_INDEX,
    MatrixInput,
    MatrixResult,
    score_matrix,
)
from app.scoring.section_roles import (
    CLAIM_SECTION_TYPES,
    EVIDENCE_SECTION_TYPES,
    is_claim_section,
    is_evidence_section,
)

DEFAULT_SECTION_ALPHA = (
    "experience:1.0,project:1.0,certification:0.8,"
    "education:0.8,summary:0.7,skills:0.6"
)
ALLOWED_MATRIX_IMPORTS = {"numpy", "dataclasses"}
APPROX = {"abs": 1e-12}

IDENTITY_REQUIREMENTS = np.array([[1.0, 0.0], [0.0, 1.0]])


def build_input(
    requirements: list[list[float]],
    sections: list[list[float]],
    weights: list[float],
    coefficients: list[float],
) -> MatrixInput:
    """A `MatrixInput` built from plain lists."""
    return MatrixInput(
        requirement_embeddings=np.array(requirements, dtype=float),
        section_embeddings=np.array(sections, dtype=float),
        requirement_weights=np.array(weights, dtype=float),
        section_type_coefficients=np.array(coefficients, dtype=float),
    )


def default_coefficients(section_types: list[SectionType]) -> list[float]:
    """Coefficients of the SPEC defaults for the given section types."""
    mapping = parse_section_alpha(DEFAULT_SECTION_ALPHA)
    return [mapping[section_type] for section_type in section_types]


def test_score_follows_the_spec_formula() -> None:
    """Requirements x sections with weights and coefficients, worked by hand."""
    # sim row 0 = [1, 0.6, 0]; sim row 1 = [0, 0.8, 0.5]
    # adj (alpha = 1, 0.5, 1): row 0 = [1, 0.3, 0]; row 1 = [0, 0.4, 0.5]
    # best = [0, 2]; score = (1*1 + 3*0.5) / (1 + 3) = 0.625
    result = score_matrix(
        build_input(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.6, 0.8], [0.0, 0.5]],
            [1, 3],
            [1.0, 0.5, 1.0],
        )
    )
    assert result.score == pytest.approx(0.625, **APPROX)
    assert result.similarities == pytest.approx(
        np.array([[1.0, 0.6, 0.0], [0.0, 0.8, 0.5]]), **APPROX
    )
    assert result.adjusted_similarities == pytest.approx(
        np.array([[1.0, 0.3, 0.0], [0.0, 0.4, 0.5]]), **APPROX
    )
    assert isinstance(result, MatrixResult)
    assert isinstance(result.score, float)


def test_best_section_is_chosen_per_requirement_after_adjustment() -> None:
    """The argmax is taken on the adjusted values, not on the raw similarities."""
    # raw sim for requirement 0: [0.9, 0.6]; adj with alpha (0.5, 1.0): [0.45, 0.6]
    result = score_matrix(
        build_input([[1.0, 0.0]], [[0.9, 0.0], [0.6, 0.0]], [2], [0.5, 1.0])
    )
    assert result.best_section_indices.tolist() == [1]
    assert result.score == pytest.approx(0.6, **APPROX)


def test_best_section_differs_per_requirement() -> None:
    """Each requirement reports its own best section index."""
    result = score_matrix(
        build_input(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.2, 0.9], [0.8, 0.1]],
            [1, 1],
            [1.0, 1.0],
        )
    )
    # requirement 0 likes section 1 (0.8), requirement 1 likes section 0 (0.9)
    assert result.best_section_indices.tolist() == [1, 0]
    assert result.score == pytest.approx((0.8 + 0.9) / 2, **APPROX)


def test_equal_adjusted_values_pick_the_first_section() -> None:
    """Ties are broken deterministically towards the lowest index."""
    result = score_matrix(
        build_input([[1.0, 0.0]], [[0.5, 0.0], [0.5, 0.0]], [1], [1.0, 1.0])
    )
    assert result.best_section_indices.tolist() == [0]


@pytest.mark.parametrize(
    "section_types",
    [
        [SectionType.SKILLS, SectionType.EXPERIENCE],
        [SectionType.EXPERIENCE, SectionType.SKILLS],
    ],
)
def test_evidence_section_beats_claim_section_at_same_similarity(
    section_types: list[SectionType],
) -> None:
    """Two sections with the same similarity: experience (1.0) wins over skills (0.6)."""
    same_vector = [0.6, 0.8]  # similarity 0.6 with the requirement [1, 0]
    result = score_matrix(
        build_input(
            [[1.0, 0.0]],
            [same_vector, same_vector],
            [1],
            default_coefficients(section_types),
        )
    )
    expected_best = section_types.index(SectionType.EXPERIENCE)
    assert result.best_section_indices.tolist() == [expected_best]
    # adjusted = 0.6 * 1.0 for experience; skills would only reach 0.6 * 0.6
    assert result.score == pytest.approx(0.6, **APPROX)


def test_candidate_with_evidence_outranks_candidate_with_only_a_claim() -> None:
    """The same vector scores 0.6 as experience and 0.36 as skills."""
    section = [[0.6, 0.8]]
    as_experience = score_matrix(
        build_input([[1.0, 0.0]], section, [1], default_coefficients([SectionType.EXPERIENCE]))
    )
    as_skills = score_matrix(
        build_input([[1.0, 0.0]], section, [1], default_coefficients([SectionType.SKILLS]))
    )
    assert as_experience.score == pytest.approx(0.6, **APPROX)
    assert as_skills.score == pytest.approx(0.36, **APPROX)
    assert as_experience.score > as_skills.score


def test_weights_scale_does_not_change_the_score() -> None:
    """The score divides by the weight sum, so only the ratios matter."""
    sections = [[1.0, 0.0], [0.0, 0.5]]
    light = score_matrix(build_input([[1.0, 0.0], [0.0, 1.0]], sections, [1, 3], [1.0, 1.0]))
    heavy = score_matrix(build_input([[1.0, 0.0], [0.0, 1.0]], sections, [2, 6], [1.0, 1.0]))
    # (1*1 + 3*0.5) / 4 = 0.625
    assert light.score == pytest.approx(0.625, **APPROX)
    assert heavy.score == pytest.approx(light.score, **APPROX)


def test_changing_a_weight_flips_which_candidate_wins() -> None:
    """Candidate A is strong on requirement 0, candidate B is stronger on requirement 1."""
    candidate_a = [[1.0, 0.0], [0.0, 0.2]]  # requirement values 1.0 and 0.2
    candidate_b = [[0.5, 0.0], [0.0, 0.8]]  # requirement values 0.5 and 0.8
    coefficients = [1.0, 1.0]

    def score_with(weights: list[float], sections: list[list[float]]) -> float:
        return score_matrix(
            build_input([[1.0, 0.0], [0.0, 1.0]], sections, weights, coefficients)
        ).score

    # weights (1, 1): A = (1.0 + 0.2) / 2 = 0.6, B = (0.5 + 0.8) / 2 = 0.65
    assert score_with([1, 1], candidate_a) == pytest.approx(0.6, **APPROX)
    assert score_with([1, 1], candidate_b) == pytest.approx(0.65, **APPROX)
    assert score_with([1, 1], candidate_b) > score_with([1, 1], candidate_a)
    # weights (3, 1): A = (3.0 + 0.2) / 4 = 0.8, B = (1.5 + 0.8) / 4 = 0.575
    assert score_with([3, 1], candidate_a) == pytest.approx(0.8, **APPROX)
    assert score_with([3, 1], candidate_b) == pytest.approx(0.575, **APPROX)
    assert score_with([3, 1], candidate_a) > score_with([3, 1], candidate_b)


def test_zero_weight_requirement_is_ignored() -> None:
    """A requirement with weight 0 adds nothing as long as another weight is positive."""
    result = score_matrix(
        build_input(
            [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.2]], [0, 4], [1.0, 1.0]
        )
    )
    assert result.score == pytest.approx(0.2, **APPROX)


def test_float32_embeddings_are_accepted() -> None:
    """Real embeddings are float32 blobs; the result is still a float score."""
    result = score_matrix(
        MatrixInput(
            requirement_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            section_embeddings=np.array([[0.5, 0.5]], dtype=np.float32),
            requirement_weights=np.array([2.0], dtype=np.float32),
            section_type_coefficients=np.array([0.5], dtype=np.float32),
        )
    )
    assert result.score == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("empty_sections", [np.empty((0, 2)), np.empty((0,)), np.array([])])
def test_empty_section_list_gives_score_zero(empty_sections: np.ndarray) -> None:
    """No sections: score 0, no best section, empty matrices, no NaN and no crash."""
    result = score_matrix(
        MatrixInput(
            requirement_embeddings=IDENTITY_REQUIREMENTS,
            section_embeddings=empty_sections,
            requirement_weights=np.array([1.0, 2.0]),
            section_type_coefficients=np.empty((0,)),
        )
    )
    assert result.score == 0.0
    assert not np.isnan(result.score)
    assert result.best_section_indices.tolist() == [NO_SECTION_INDEX, NO_SECTION_INDEX]
    assert result.similarities.shape == (2, 0)
    assert result.adjusted_similarities.shape == (2, 0)


def test_empty_sections_still_validate_weights() -> None:
    """Fail-fast checks on the requirements apply to the empty case too."""
    with pytest.raises(ValueError, match="sum to zero"):
        score_matrix(
            MatrixInput(
                requirement_embeddings=IDENTITY_REQUIREMENTS,
                section_embeddings=np.empty((0, 2)),
                requirement_weights=np.zeros(2),
                section_type_coefficients=np.empty((0,)),
            )
        )


VALID_REQUIREMENTS = [[1.0, 0.0], [0.0, 1.0]]
VALID_SECTIONS = [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    ("bad_input", "message"),
    [
        (build_input(VALID_REQUIREMENTS, [[1.0, 0.0, 0.0]], [1, 1], [1.0]), "dimension mismatch"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1, 1, 1], [1.0, 1.0]), "requirement_weights must have shape"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1], [1.0, 1.0]), "requirement_weights must have shape"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1, 1], [1.0]), "section_type_coefficients must have shape"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1, 1], [1.0, 1.0, 1.0]), "section_type_coefficients must have shape"),
        (
            MatrixInput(
                requirement_embeddings=np.array(VALID_REQUIREMENTS),
                section_embeddings=np.empty((0, 2)),
                requirement_weights=np.array([1.0, 1.0]),
                section_type_coefficients=np.array([1.0]),
            ),
            "section_type_coefficients must have shape",
        ),
        (build_input([[1.0, 0.0]], VALID_SECTIONS, [0], [1.0, 1.0]), "sum to zero"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [0, 0], [1.0, 1.0]), "sum to zero"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [2, -1], [1.0, 1.0]), "non-negative"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1, float("nan")], [1.0, 1.0]), "finite"),
        (build_input(VALID_REQUIREMENTS, [[float("nan"), 0.0]], [1, 1], [1.0]), "finite"),
        (build_input(VALID_REQUIREMENTS, VALID_SECTIONS, [1, 1], [1.0, float("inf")]), "finite"),
    ],
)
def test_bad_shapes_and_values_raise_value_error(
    bad_input: MatrixInput, message: str
) -> None:
    """Every invalid combination fails fast with a clear message."""
    with pytest.raises(ValueError, match=message):
        score_matrix(bad_input)


def test_no_requirements_raises_value_error() -> None:
    """A posting without requirements cannot be scored."""
    with pytest.raises(ValueError, match="At least one requirement"):
        score_matrix(
            MatrixInput(
                requirement_embeddings=np.empty((0, 2)),
                section_embeddings=np.array(VALID_SECTIONS),
                requirement_weights=np.empty((0,)),
                section_type_coefficients=np.array([1.0, 1.0]),
            )
        )


def test_one_dimensional_requirement_embeddings_raise_value_error() -> None:
    """A single vector instead of a matrix is rejected, not silently reshaped."""
    with pytest.raises(ValueError, match="At least one requirement"):
        score_matrix(
            MatrixInput(
                requirement_embeddings=np.array([1.0, 0.0]),
                section_embeddings=np.array(VALID_SECTIONS),
                requirement_weights=np.array([1.0]),
                section_type_coefficients=np.array([1.0, 1.0]),
            )
        )


def test_one_dimensional_non_empty_sections_raise_value_error() -> None:
    """A non-empty 1-D section array is a shape error."""
    with pytest.raises(ValueError, match="2-D"):
        score_matrix(
            MatrixInput(
                requirement_embeddings=IDENTITY_REQUIREMENTS,
                section_embeddings=np.array([1.0, 0.0]),
                requirement_weights=np.array([1.0, 1.0]),
                section_type_coefficients=np.array([1.0, 1.0]),
            )
        )


def test_inputs_are_not_mutated() -> None:
    """Scoring leaves every input array untouched."""
    matrix_input = build_input(
        [[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.8], [0.0, 0.5]], [1, 3], [1.0, 0.5]
    )
    snapshot = [
        array.copy()
        for array in (
            matrix_input.requirement_embeddings,
            matrix_input.section_embeddings,
            matrix_input.requirement_weights,
            matrix_input.section_type_coefficients,
        )
    ]
    for array in (
        matrix_input.requirement_embeddings,
        matrix_input.section_embeddings,
        matrix_input.requirement_weights,
        matrix_input.section_type_coefficients,
    ):
        array.setflags(write=False)
    score_matrix(matrix_input)
    after = [
        matrix_input.requirement_embeddings,
        matrix_input.section_embeddings,
        matrix_input.requirement_weights,
        matrix_input.section_type_coefficients,
    ]
    for before_array, after_array in zip(snapshot, after, strict=True):
        assert np.array_equal(before_array, after_array)


def test_same_input_gives_the_same_result() -> None:
    """The function is pure: repeated calls return equal results."""
    matrix_input = build_input(
        [[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.8], [0.0, 0.5]], [1, 3], [1.0, 0.5]
    )
    first = score_matrix(matrix_input)
    second = score_matrix(matrix_input)
    assert first.score == second.score
    assert np.array_equal(first.best_section_indices, second.best_section_indices)


def test_matrix_module_imports_only_numpy_and_standard_library() -> None:
    """`matrix.py` has no DB, model, or project imports (checked on its AST)."""
    tree = ast.parse(Path(matrix.__file__).read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not allowed in matrix.py"
            imported_roots.add((node.module or "").split(".")[0])
    assert imported_roots == ALLOWED_MATRIX_IMPORTS


def test_evidence_and_claim_sets_follow_the_spec() -> None:
    """Experience and project are evidence, skills is the claim, nothing overlaps."""
    assert EVIDENCE_SECTION_TYPES == {SectionType.EXPERIENCE, SectionType.PROJECT}
    assert CLAIM_SECTION_TYPES == {SectionType.SKILLS}
    assert EVIDENCE_SECTION_TYPES.isdisjoint(CLAIM_SECTION_TYPES)


@pytest.mark.parametrize(
    ("section_type", "evidence", "claim"),
    [
        (SectionType.EXPERIENCE, True, False),
        (SectionType.PROJECT, True, False),
        (SectionType.SKILLS, False, True),
        (SectionType.EDUCATION, False, False),
        (SectionType.CERTIFICATION, False, False),
        (SectionType.SUMMARY, False, False),
    ],
)
def test_section_role_predicates(
    section_type: SectionType, evidence: bool, claim: bool
) -> None:
    """Each section type has exactly the role the SPEC gives it."""
    assert is_evidence_section(section_type) is evidence
    assert is_claim_section(section_type) is claim
