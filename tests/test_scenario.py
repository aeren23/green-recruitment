"""The benchmark scenario (SPEC 7.3 and decision T9-2)."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from bench_support import (
    CATEGORY_ACCOUNTANT,
    CATEGORY_DISTRACTOR,
    CATEGORY_FINANCE,
    CATEGORY_HR,
    CATEGORY_IT,
    touch_records,
)

from bench.dataset import ResumeRecord
from bench.scenario import (
    POSTING_TITLES,
    ScenarioApplication,
    ScenarioConfig,
    ScenarioConfigError,
    ScenarioModels,
    build_scenario,
    resubmit_count,
    scenario_json,
)

MODELS = ScenarioModels("embed-model", "llm-model")
RECORDS_PER_GROUP = 20
SAMPLE_SIZE = 40
SEED = 7
TOP_K = 5
SHARE_TOLERANCE = 0.01


def make_config(**overrides: object) -> ScenarioConfig:
    """A config with test defaults; `overrides` replace single fields."""
    fields: dict[str, object] = {
        "n": SAMPLE_SIZE,
        "seed": SEED,
        "top_k": TOP_K,
        "models": MODELS,
    }
    fields.update(overrides)
    return ScenarioConfig(**fields)  # type: ignore[arg-type]


@pytest.fixture
def records(tmp_path: Path) -> list[ResumeRecord]:
    """Enough empty-PDF records for every stratum of a 40-CV sample."""
    counts = {
        CATEGORY_IT: RECORDS_PER_GROUP,
        CATEGORY_ACCOUNTANT: RECORDS_PER_GROUP // 2,
        CATEGORY_FINANCE: RECORDS_PER_GROUP // 2,
        CATEGORY_HR: RECORDS_PER_GROUP,
        CATEGORY_DISTRACTOR: RECORDS_PER_GROUP,
    }
    return touch_records(tmp_path, counts)


def first_applications(applications: list[ScenarioApplication]) -> list[ScenarioApplication]:
    """The applications that are not repeats."""
    return [application for application in applications if not application.is_resubmit]


def test_same_seed_gives_the_same_list(records: list[ResumeRecord]) -> None:
    first = build_scenario(records, make_config())
    second = build_scenario(list(reversed(records)), make_config())
    assert first == second


def test_another_seed_gives_another_list(records: list[ResumeRecord]) -> None:
    assert build_scenario(records, make_config()) != build_scenario(
        records, make_config(seed=SEED + 1)
    )


@pytest.mark.parametrize("rate", [0.1, 0.25, 0.5])
def test_resubmit_share_equals_the_rate_within_rounding(
    records: list[ResumeRecord], rate: float
) -> None:
    applications = build_scenario(records, make_config(resubmit_rate=rate))
    repeats = sum(application.is_resubmit for application in applications)
    base = len(applications) - repeats
    assert repeats == resubmit_count(base, rate)
    assert repeats / len(applications) == pytest.approx(rate, abs=SHARE_TOLERANCE)


def test_resubmit_count_follows_decision_t9_2() -> None:
    assert resubmit_count(80, 0.1) == 9  # 80 * 0.1 / 0.9 = 8.89
    assert resubmit_count(80, 0.0) == 0


def test_each_cv_applies_to_distinct_postings(records: list[ResumeRecord]) -> None:
    applications = build_scenario(records, make_config(jobs_per_cv=2))
    titles_by_record: dict[str, list[str]] = defaultdict(list)
    for application in first_applications(applications):
        titles_by_record[application.record_id].append(application.posting_title)
    assert len(titles_by_record) == SAMPLE_SIZE
    for titles in titles_by_record.values():
        assert len(titles) == 2
        assert len(set(titles)) == 2
        assert set(titles) <= set(POSTING_TITLES)


def test_resubmits_follow_their_originals(records: list[ResumeRecord]) -> None:
    applications = build_scenario(records, make_config(resubmit_rate=0.25))
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for application in applications:
        key = (application.record_id, application.posting_title)
        if application.is_resubmit:
            repeats += 1
            assert key in seen
        else:
            seen.add(key)
    assert repeats > 0


def test_no_resubmits_and_one_job_per_cv(records: list[ResumeRecord]) -> None:
    config = make_config(jobs_per_cv=1, resubmit_rate=0.0)
    applications = build_scenario(records, config)
    assert len(applications) == SAMPLE_SIZE
    assert not any(application.is_resubmit for application in applications)
    assert len(Counter(application.record_id for application in applications)) == SAMPLE_SIZE


@pytest.mark.parametrize(
    "overrides",
    [
        {"jobs_per_cv": 0},
        {"jobs_per_cv": len(POSTING_TITLES) + 1},
        {"resubmit_rate": 1.0},
        {"resubmit_rate": -0.1},
    ],
)
def test_invalid_parameters_fail_fast(
    records: list[ResumeRecord], overrides: dict[str, object]
) -> None:
    with pytest.raises(ScenarioConfigError):
        build_scenario(records, make_config(**overrides))


def sorted_at_every_level(value: object) -> bool:
    """Whether every dictionary inside the value has its keys in sorted order."""
    if isinstance(value, dict):
        keys = list(value)
        return keys == sorted(keys) and all(sorted_at_every_level(v) for v in value.values())
    return True


def test_scenario_json_is_canonical() -> None:
    config = make_config()
    document = scenario_json(config)
    assert sorted_at_every_level(document)
    assert json.dumps(document) == json.dumps(document, sort_keys=True)
    assert document == scenario_json(make_config())
    assert document != scenario_json(make_config(seed=SEED + 1))
    assert document["models"] == {"embedding": "embed-model", "llm": "llm-model"}
    assert {"n", "jobs_per_cv", "resubmit_rate", "seed", "top_k", "models"} == set(document)
