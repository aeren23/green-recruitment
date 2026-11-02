"""Application scenario of a benchmark run (SPEC section 7.3).

`build_scenario` turns a stratified CV sample into the ordered list of
applications both pipelines process. The list depends only on the dataset
records and the `ScenarioConfig`, so the green and the baseline runner see the
same applications for the same seed.

`resubmit_rate` is read as the share of *all* applications that are repeats
(decision T9-2): with `base` first applications and rate `r`, there are
`round(base * r / (1 - r))` repeats, so repeats / (base + repeats) is `r`.
`round` is Python's banker's rounding.
"""

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.seed import SEED_POSTINGS
from bench.dataset import ResumeRecord, sample_stratified

DEFAULT_JOBS_PER_CV = 2
DEFAULT_RESUBMIT_RATE = 0.1
POSTING_TITLES: tuple[str, ...] = tuple(seed.title for seed in SEED_POSTINGS)
MIN_JOBS_PER_CV = 1
SCENARIO_SEED_TEMPLATE = "scenario:{seed}"

JSON_N = "n"
JSON_JOBS_PER_CV = "jobs_per_cv"
JSON_RESUBMIT_RATE = "resubmit_rate"
JSON_SEED = "seed"
JSON_TOP_K = "top_k"
JSON_MODELS = "models"
JSON_EMBEDDING_MODEL = "embedding"
JSON_LLM_MODEL = "llm"
JSON_COMPACT_SEPARATORS = (",", ":")


class ScenarioConfigError(ValueError):
    """Raised when scenario parameters cannot describe a valid scenario."""


@dataclass(frozen=True)
class ScenarioModels:
    """Names of the models a run uses (part of the scenario identity).

    Attributes:
        embedding: Embedding model name.
        llm: LLM name.
    """

    embedding: str
    llm: str


@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters of a scenario; stored as `scenario_json` of every run.

    Attributes:
        n: Number of sampled CVs.
        seed: Seed of the sampling and of every random choice.
        top_k: Size of the per-posting top-k list.
        models: Embedding model and LLM names.
        jobs_per_cv: Distinct postings each CV applies to.
        resubmit_rate: Share of all applications that are repeats, in [0, 1).
    """

    n: int
    seed: int
    top_k: int
    models: ScenarioModels
    jobs_per_cv: int = DEFAULT_JOBS_PER_CV
    resubmit_rate: float = DEFAULT_RESUBMIT_RATE


@dataclass(frozen=True)
class ScenarioApplication:
    """One application of the scenario.

    Attributes:
        record_id: Dataset record id of the CV (an id, never CV content).
        posting_title: Title of the posting the CV applies to.
        is_resubmit: True for a repeat of an earlier identical application.
    """

    record_id: str
    posting_title: str
    is_resubmit: bool = False


def validate_config(config: ScenarioConfig) -> None:
    """Fail fast on parameters that cannot describe a scenario.

    Raises:
        ScenarioConfigError: If `jobs_per_cv` is not between 1 and the number
            of postings, or `resubmit_rate` is not in [0, 1).
    """
    if not MIN_JOBS_PER_CV <= config.jobs_per_cv <= len(POSTING_TITLES):
        raise ScenarioConfigError(
            f"jobs_per_cv must be between {MIN_JOBS_PER_CV} and "
            f"{len(POSTING_TITLES)}, got {config.jobs_per_cv}."
        )
    if not 0.0 <= config.resubmit_rate < 1.0:
        raise ScenarioConfigError(
            f"resubmit_rate must be in [0, 1), got {config.resubmit_rate}."
        )


def resubmit_count(base_count: int, resubmit_rate: float) -> int:
    """Number of repeats so that repeats are `resubmit_rate` of all applications."""
    return round(base_count * resubmit_rate / (1.0 - resubmit_rate))


def first_applications(
    sample: Iterable[ResumeRecord], config: ScenarioConfig, rng: random.Random
) -> list[ScenarioApplication]:
    """Each CV applies to `jobs_per_cv` distinct postings drawn with the seeded RNG."""
    applications = [
        ScenarioApplication(record.id, title)
        for record in sample
        for title in rng.sample(POSTING_TITLES, config.jobs_per_cv)
    ]
    rng.shuffle(applications)
    return applications


def place_resubmits(
    applications: list[ScenarioApplication], count: int, rng: random.Random
) -> list[ScenarioApplication]:
    """Insert `count` repeats, each at a seeded position after its original.

    Originals are chosen with replacement among the first applications, so one
    application may be repeated more than once.
    """
    originals = list(applications)
    placed = list(applications)
    for _ in range(count):
        original = rng.choice(originals)
        position = placed.index(original) + 1
        repeat = ScenarioApplication(original.record_id, original.posting_title, True)
        placed.insert(rng.randint(position, len(placed)), repeat)
    return placed


def build_scenario(
    records: Iterable[ResumeRecord], config: ScenarioConfig
) -> list[ScenarioApplication]:
    """Build the deterministic application list of a scenario.

    Args:
        records: All dataset records (the stratified sample is drawn here).
        config: Scenario parameters.

    Returns:
        First applications in a shuffled order, with the repeats inserted
        after their originals. The same records and config give the same list.

    Raises:
        ScenarioConfigError: On invalid parameters.
        ValueError: If a stratum of the dataset is too small (from sampling).
    """
    validate_config(config)
    sample = sample_stratified(records, config.n, config.seed)
    rng = random.Random(SCENARIO_SEED_TEMPLATE.format(seed=config.seed))
    applications = first_applications(sample, config, rng)
    repeats = resubmit_count(len(applications), config.resubmit_rate)
    return place_resubmits(applications, repeats, rng)


def scenario_json(config: ScenarioConfig) -> dict[str, Any]:
    """The canonical `scenario_json` of a config: keys sorted, models nested.

    Identical scenarios produce identical dictionaries (and identical JSON),
    which is what `bench compare` groups runs by.
    """
    models = {
        JSON_EMBEDDING_MODEL: config.models.embedding,
        JSON_LLM_MODEL: config.models.llm,
    }
    fields: Mapping[str, Any] = {
        JSON_N: config.n,
        JSON_JOBS_PER_CV: config.jobs_per_cv,
        JSON_RESUBMIT_RATE: config.resubmit_rate,
        JSON_SEED: config.seed,
        JSON_TOP_K: config.top_k,
        JSON_MODELS: dict(sorted(models.items())),
    }
    return dict(sorted(fields.items()))


def scenario_key(scenario: Mapping[str, Any]) -> str:
    """Canonical JSON text of a stored `scenario_json`, identical for equal scenarios.

    `bench compare` groups runs by this key (SC-11).
    """
    return json.dumps(scenario, sort_keys=True, separators=JSON_COMPACT_SEPARATORS)
