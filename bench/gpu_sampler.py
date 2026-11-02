"""Whole-run GPU power cross-check (measurement gate decision D4).

CodeCarbon undercounts GPU energy in very short stages. To see by how much, one
long-running `nvidia-smi` process streams the GPU power draw for the whole
measured phase; its samples are integrated with the trapezoid rule and later
compared with the GPU energy summed over the stages.

`GpuPowerSampler` is the small interface the runner depends on. The real
`NvidiaSmiSampler` owns the subprocess and always terminates it; tests inject a
fake sampler.
"""

import logging
import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import IO, Protocol

logger = logging.getLogger(__name__)

NVIDIA_SMI_COMMAND: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=power.draw",
    "--format=csv,noheader,nounits",
    "-lms",
    "500",
)
TERMINATE_TIMEOUT_S = 5.0
READER_JOIN_TIMEOUT_S = 5.0
SECONDS_PER_HOUR = 3600.0
MIN_SAMPLES_FOR_INTEGRAL = 2
PIPE_ENCODING = "utf-8"


@dataclass(frozen=True)
class PowerSample:
    """One streamed reading.

    Attributes:
        time_s: Receipt time in seconds on a monotonic clock.
        power_w: GPU power draw in watts.
    """

    time_s: float
    power_w: float


@dataclass(frozen=True)
class GpuStreamReading:
    """The result of one sampling window.

    Attributes:
        available: False when `nvidia-smi` could not be run or gave too few
            usable samples; the other fields are then zero.
        energy_wh: Trapezoid integral of the samples, in watt-hours.
        window_s: Seconds between the first and the last sample.
        sample_count: Usable samples.
        malformed_lines: Lines that were not a power reading and were skipped.
    """

    available: bool
    energy_wh: float = 0.0
    window_s: float = 0.0
    sample_count: int = 0
    malformed_lines: int = 0


UNAVAILABLE_READING = GpuStreamReading(available=False)


class GpuPowerSampler(Protocol):
    """Streams GPU power between `start()` and `stop()`."""

    def start(self) -> None:
        """Begin sampling; never raises when the GPU tool is missing."""
        ...

    def stop(self) -> GpuStreamReading:
        """End sampling, release the subprocess and return the reading."""
        ...


def parse_power_line(line: str) -> float | None:
    """Parse one `nvidia-smi` output line into watts.

    Args:
        line: A raw line such as `"29.84\\n"`.

    Returns:
        The power in watts, or `None` for anything that is not a finite,
        non-negative number (`[N/A]`, blank lines, error text).
    """
    try:
        watts = float(line.strip())
    except ValueError:
        return None
    if not math.isfinite(watts) or watts < 0:
        return None
    return watts


def integrate_trapezoid_wh(samples: Sequence[PowerSample]) -> float:
    """Energy in watt-hours of a power series, by the trapezoid rule.

    Fewer than two samples cover no time and give zero.
    """
    joules = sum(
        (earlier.power_w + later.power_w) / 2.0 * (later.time_s - earlier.time_s)
        for earlier, later in zip(samples, samples[1:])
    )
    return joules / SECONDS_PER_HOUR


class SampleCollector:
    """Turns a stream of text lines into timestamped power samples.

    Args:
        clock: Monotonic clock in seconds, read once per usable line.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.samples: list[PowerSample] = []
        self.malformed_lines = 0

    def consume(self, lines: Iterable[str]) -> None:
        """Read every line, keeping the readings and counting the rest."""
        for line in lines:
            watts = parse_power_line(line)
            if watts is None:
                self.malformed_lines += 1
                continue
            self.samples.append(PowerSample(self._clock(), watts))

    def reading(self) -> GpuStreamReading:
        """The integral over the samples seen so far."""
        if len(self.samples) < MIN_SAMPLES_FOR_INTEGRAL:
            return GpuStreamReading(
                available=False,
                sample_count=len(self.samples),
                malformed_lines=self.malformed_lines,
            )
        return GpuStreamReading(
            available=True,
            energy_wh=integrate_trapezoid_wh(self.samples),
            window_s=self.samples[-1].time_s - self.samples[0].time_s,
            sample_count=len(self.samples),
            malformed_lines=self.malformed_lines,
        )


def launch_nvidia_smi() -> "subprocess.Popen[str]":
    """Start the streaming `nvidia-smi` process with a text pipe."""
    return subprocess.Popen(
        NVIDIA_SMI_COMMAND,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding=PIPE_ENCODING,
    )


class NvidiaSmiSampler:
    """`GpuPowerSampler` backed by one long-running `nvidia-smi` process.

    Args:
        launcher: Starts the subprocess; tests inject a fake.
        clock: Monotonic clock for the sample timestamps.
    """

    def __init__(
        self,
        launcher: Callable[[], "subprocess.Popen[str]"] = launch_nvidia_smi,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._launcher = launcher
        self._collector = SampleCollector(clock)
        self._process: "subprocess.Popen[str] | None" = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        """Launch the process and its reader thread; log when it cannot start."""
        try:
            self._process = self._launcher()
        except OSError as error:
            logger.warning("nvidia-smi could not be started: %s", error.__class__.__name__)
            return
        stream: IO[str] = self._process.stdout  # type: ignore[assignment]
        self._reader = threading.Thread(
            target=self._collector.consume, args=(stream,), daemon=True
        )
        self._reader.start()

    def stop(self) -> GpuStreamReading:
        """Terminate the process (killing it if needed) and integrate the samples."""
        process, reader = self._process, self._reader
        self._process = self._reader = None
        if process is None:
            return UNAVAILABLE_READING
        self._terminate(process)
        if reader is not None:
            reader.join(READER_JOIN_TIMEOUT_S)
        if process.stdout is not None:
            process.stdout.close()
        return self._collector.reading()

    @staticmethod
    def _terminate(process: "subprocess.Popen[str]") -> None:
        """Stop the subprocess; escalate to kill if it ignores terminate."""
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi ignored terminate; killing it")
            process.kill()
            process.wait()
