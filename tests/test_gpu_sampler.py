"""The `nvidia-smi` stream parser, the integral and the sampler lifecycle."""

import io
import itertools
import subprocess
from typing import Any

import pytest

from bench.gpu_sampler import (
    NvidiaSmiSampler,
    PowerSample,
    SampleCollector,
    integrate_trapezoid_wh,
    parse_power_line,
)

SECONDS_PER_HOUR = 3600.0
STREAM_LINES = ["10.0\n", "20.0\n", "20.0\n"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [("29.84\n", 29.84), (" 5 \r\n", 5.0), ("0.00", 0.0)],
)
def test_valid_lines_parse(line: str, expected: float) -> None:
    assert parse_power_line(line) == expected


@pytest.mark.parametrize(
    "line",
    ["", "\n", "[N/A]", "abc", "-3.0", "nan", "inf", "12.5 W", "Failed to initialize NVML"],
)
def test_malformed_lines_are_rejected(line: str) -> None:
    assert parse_power_line(line) is None


def test_constant_power_integrates_to_power_times_time() -> None:
    samples = [PowerSample(float(second), 60.0) for second in range(61)]
    assert integrate_trapezoid_wh(samples) == pytest.approx(1.0)


def test_trapezoid_uses_the_mean_of_neighbouring_samples() -> None:
    samples = [PowerSample(0.0, 10.0), PowerSample(1.0, 20.0), PowerSample(3.0, 20.0)]
    expected_joules = (10.0 + 20.0) / 2 * 1.0 + 20.0 * 2.0
    assert integrate_trapezoid_wh(samples) == pytest.approx(expected_joules / SECONDS_PER_HOUR)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_samples_cover_no_time(count: int) -> None:
    samples = [PowerSample(0.0, 50.0)][:count]
    assert integrate_trapezoid_wh(samples) == 0.0


def test_collector_skips_and_counts_malformed_lines() -> None:
    times = iter([0.0, 1.0, 3.0])
    collector = SampleCollector(clock=lambda: next(times))
    collector.consume(["10.0\n", "bad\n", "20.0\n", "[N/A]\n", "20.0\n"])
    reading = collector.reading()
    assert reading.available
    assert reading.sample_count == 3
    assert reading.malformed_lines == 2
    assert reading.window_s == pytest.approx(3.0)
    assert reading.energy_wh == pytest.approx((15.0 * 1.0 + 20.0 * 2.0) / SECONDS_PER_HOUR)


def test_collector_with_one_sample_is_unavailable() -> None:
    collector = SampleCollector(clock=lambda: 0.0)
    collector.consume(["10.0\n", "junk\n"])
    reading = collector.reading()
    assert not reading.available
    assert reading.sample_count == 1
    assert reading.malformed_lines == 1


class FakeProcess:
    """Stands in for `subprocess.Popen` of the streaming `nvidia-smi`."""

    def __init__(self, lines: list[str], ignore_terminate: bool = False) -> None:
        self.stdout = io.StringIO("".join(lines))
        self.terminated = 0
        self.killed = 0
        self._ignore_terminate = ignore_terminate

    def terminate(self) -> None:
        """Record the request."""
        self.terminated += 1

    def wait(self, timeout: float | None = None) -> int:
        """Time out once when the process ignores `terminate`."""
        if self._ignore_terminate and not self.killed:
            raise subprocess.TimeoutExpired("nvidia-smi", timeout or 0.0)
        return 0

    def kill(self) -> None:
        """Record the kill."""
        self.killed += 1


def make_sampler(process: FakeProcess) -> NvidiaSmiSampler:
    """A sampler whose clock ticks one second per sample."""
    ticks = itertools.count()
    return NvidiaSmiSampler(launcher=lambda: process, clock=lambda: float(next(ticks)))  # type: ignore[arg-type,return-value]


def test_sampler_integrates_the_stream_and_terminates_the_process() -> None:
    process = FakeProcess(STREAM_LINES)
    sampler = make_sampler(process)
    sampler.start()
    reading = sampler.stop()
    assert process.terminated == 1
    assert reading.available
    assert reading.sample_count == 3
    assert reading.energy_wh == pytest.approx((15.0 + 20.0) / SECONDS_PER_HOUR)


def test_sampler_kills_a_process_that_ignores_terminate() -> None:
    process = FakeProcess(STREAM_LINES, ignore_terminate=True)
    sampler = make_sampler(process)
    sampler.start()
    sampler.stop()
    assert process.terminated == 1
    assert process.killed == 1


def test_stop_is_idempotent() -> None:
    process = FakeProcess(STREAM_LINES)
    sampler = make_sampler(process)
    sampler.start()
    sampler.stop()
    second = sampler.stop()
    assert process.terminated == 1
    assert not second.available


def test_missing_nvidia_smi_is_reported_as_unavailable() -> None:
    def launcher() -> Any:
        raise FileNotFoundError("nvidia-smi")

    sampler = NvidiaSmiSampler(launcher=launcher)
    sampler.start()
    assert not sampler.stop().available
