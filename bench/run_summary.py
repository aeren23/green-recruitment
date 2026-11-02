"""Plain-text aggregate summary printed after each benchmark run.

Only aggregates: energy, durations, counts and sizes. No CV content.
"""

from bench.runner import RunOutcome

MILLI = 1000.0
PERCENT = 100.0
STAGE_NAME_WIDTH = 14
GPU_UNAVAILABLE_LINE = "GPU stream: unavailable (nvidia-smi could not be sampled)"
RATIO_UNAVAILABLE = "n/a"


def format_stage_lines(outcome: RunOutcome) -> list[str]:
    """One line per stage: energy in mWh, duration in seconds and call count."""
    lines = ["Stages (energy mWh, duration s, count):"]
    for name, stage in sorted(outcome.record.stages.items()):
        lines.append(
            f"  {name:<{STAGE_NAME_WIDTH}}"
            f"{stage.energy_wh * MILLI:>12.3f}"
            f"{stage.duration_s:>12.2f}"
            f"{stage.count:>8}"
        )
    return lines


def format_gpu_line(outcome: RunOutcome) -> str:
    """The GPU cross-check: stream Wh versus the Wh summed over the stages."""
    gpu, stage_sum = outcome.gpu, outcome.record.totals.gpu_stage_sum_wh
    if not gpu.available:
        return GPU_UNAVAILABLE_LINE
    ratio = f"{stage_sum / gpu.energy_wh:.3f}" if gpu.energy_wh > 0 else RATIO_UNAVAILABLE
    return (
        f"GPU: stream {gpu.energy_wh:.5f} Wh over {gpu.window_s:.1f} s, "
        f"stage sum {stage_sum:.5f} Wh, stage/stream {ratio}"
    )


def format_total_duration(outcome: RunOutcome) -> str:
    """The run's total duration: summed stage seconds and the measured wall time."""
    return (
        f"Total duration: {outcome.timing.staged_duration_s:.2f} s in stages, "
        f"{outcome.timing.measured_wall_s:.2f} s wall (model_load excluded)"
    )


def format_run_summary(outcome: RunOutcome) -> str:
    """Render the run's totals, counts, storage and GPU cross-check."""
    record, result = outcome.record, outcome.record.result
    counts, totals, timing = result.counts, record.totals, outcome.timing
    lines = [
        f"Run {outcome.run_id} ({record.pipeline.value}), LLM {result.llm_status}",
        *format_stage_lines(outcome),
        f"Gross {totals.energy_wh_gross:.5f} Wh, net {totals.energy_wh_net:.5f} Wh, "
        f"CO2 {totals.co2_g:.4f} g, duration {totals.duration_s:.2f} s "
        f"(idle {record.idle_power_w:.2f} W)",
        f"Applications {counts.submitted}: ranked {counts.ranked}, "
        f"duplicates blocked {counts.blocked}, eliminated {counts.eliminated} "
        f"{dict(counts.eliminated_by_reason)}, unreadable {counts.unreadable}",
        f"Cache hits: parse {result.parse_cache_hits}, embed {result.embed_cache_hits}",
        f"LLM: {result.llm_calls} calls, {result.llm_prompt_tokens} prompt tokens, "
        f"{result.llm_completion_tokens} completion tokens",
        f"Storage: active {result.storage_bytes_active} B, "
        f"after retention {result.storage_bytes_after_retention} B",
        format_gpu_line(outcome),
        f"Unstaged wall {timing.unstaged_wall_s:.2f} s "
        f"({timing.unstaged_share * PERCENT:.1f}% of {timing.measured_wall_s:.2f} s)",
    ]
    return "\n".join(lines)
