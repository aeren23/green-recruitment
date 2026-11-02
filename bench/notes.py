"""Assembles the `measurement_notes` string of a run (SPEC 5 and 7.5).

The notes are `key=value` pairs joined by `"; "`, the same shape as the
tracker's own notes, so they stay greppable. They hold measurement conditions
and aggregate counts only, never CV content.
"""

import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from bench.gpu_sampler import GpuStreamReading
from bench.results import PERCENTILE_P90, PipelineResult, percentile
from bench.scenario import ScenarioConfig, scenario_json

NOTES_SEPARATOR = "; "
FLOAT_FORMAT = ".6f"
RATIO_UNAVAILABLE = "n/a"
CUDA_HELD = "held"
CUDA_NONE = "none"
JSON_COMPACT_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class TimingFacts:
    """Wall-clock facts of the measured phase.

    Attributes:
        measured_wall_s: Wall time from the start of the first application to
            the end of retention.
        staged_duration_s: Summed duration of the measured stages
            (`model_load` excluded).
    """

    measured_wall_s: float
    staged_duration_s: float

    @property
    def unstaged_wall_s(self) -> float:
        """Harness time that no stage measured (file reads, commits, lookups)."""
        return self.measured_wall_s - self.staged_duration_s

    @property
    def unstaged_share(self) -> float:
        """`unstaged_wall_s` as a share of the measured wall time."""
        if self.measured_wall_s <= 0:
            return 0.0
        return self.unstaged_wall_s / self.measured_wall_s


@dataclass(frozen=True)
class NotesInputs:
    """Everything the notes are assembled from."""

    tracker_notes: str
    environment: Mapping[str, str]
    cuda_context_held: bool
    config: ScenarioConfig
    gpu: GpuStreamReading
    gpu_stage_sum_wh: float
    timing: TimingFacts
    result: PipelineResult


def format_float(value: float) -> str:
    """Fixed-precision number for the notes."""
    return format(value, FLOAT_FORMAT)


def gpu_notes(gpu: GpuStreamReading, gpu_stage_sum_wh: float) -> dict[str, str]:
    """GPU cross-check fields (decision D4): stream versus summed stages."""
    stage_sum = {"gpu_stage_sum_wh": format_float(gpu_stage_sum_wh)}
    if not gpu.available:
        return {"gpu_stream": "unavailable", **stage_sum}
    ratio = gpu_stage_sum_wh / gpu.energy_wh if gpu.energy_wh > 0 else None
    return {
        "gpu_stream_wh": format_float(gpu.energy_wh),
        **stage_sum,
        "gpu_stage_sum_over_stream": (
            format_float(ratio) if ratio is not None else RATIO_UNAVAILABLE
        ),
        "gpu_stream_window_s": format_float(gpu.window_s),
        "gpu_stream_samples": str(gpu.sample_count),
        "gpu_stream_malformed_lines": str(gpu.malformed_lines),
    }


def timing_notes(timing: TimingFacts) -> dict[str, str]:
    """Measured wall time and the share no stage covered."""
    return {
        "measured_wall_s": format_float(timing.measured_wall_s),
        "unstaged_wall_s": format_float(timing.unstaged_wall_s),
        "unstaged_share": format_float(timing.unstaged_share),
    }


def llm_notes(result: PipelineResult) -> dict[str, str]:
    """LLM availability and per-call timing statistics."""
    notes = {"llm_status": result.llm_status}
    if result.llm_call_seconds:
        notes["llm_call_median_s"] = format_float(statistics.median(result.llm_call_seconds))
        notes["llm_call_p90_s"] = format_float(
            percentile(result.llm_call_seconds, PERCENTILE_P90)
        )
    return notes


def scenario_notes(config: ScenarioConfig, cuda_context_held: bool) -> dict[str, str]:
    """N, seed, the canonical scenario and the CUDA-context flag."""
    return {
        "cuda_context": CUDA_HELD if cuda_context_held else CUDA_NONE,
        "n": str(config.n),
        "seed": str(config.seed),
        "scenario": json.dumps(
            scenario_json(config), separators=JSON_COMPACT_SEPARATORS, sort_keys=True
        ),
    }


def build_measurement_notes(inputs: NotesInputs) -> str:
    """Join the tracker notes and all run notes into one string."""
    fields: dict[str, str] = {
        **inputs.environment,
        **llm_notes(inputs.result),
        **scenario_notes(inputs.config, inputs.cuda_context_held),
        **gpu_notes(inputs.gpu, inputs.gpu_stage_sum_wh),
        **timing_notes(inputs.timing),
        **inputs.result.notes,
    }
    pairs = [f"{key}={value}" for key, value in fields.items()]
    return NOTES_SEPARATOR.join([inputs.tracker_notes, *pairs])


def parse_measurement_notes(notes: str) -> dict[str, str]:
    """Split stored notes back into `key -> value` (the first `=` of a pair splits).

    The inverse of `build_measurement_notes` for the `key=value` pairs; a
    later pair overrides an earlier one with the same key.
    """
    pairs = (item.split("=", 1) for item in notes.split(NOTES_SEPARATOR) if "=" in item)
    return {key: value for key, value in pairs}
