"""Per-stage energy and duration tracking (SPEC sections 7.4 and 7.5).

Every measured operation runs inside `tracker.stage("<name>")`. Two
implementations share the `StageTracker` protocol:

* `NoOpStageTracker` is used when `GREEN_TRACKING=false`. It does no timing and
  never imports CodeCarbon.
* `CodeCarbonStageTracker` wraps the CodeCarbon task API
  (`start_task` / `stop_task`). CodeCarbon is imported lazily, only when this
  class builds its own emissions tracker.

The tracker is not thread-safe and does not support nested stages: CodeCarbon
tracks a single active task per emissions tracker, so a second stage opened
while one is running raises `NestedStageError`.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Any, Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

STAGE_NAMES: frozenset[str] = frozenset(
    {
        "store_pdf",
        "extract",
        "llm_score",
        "rank",
        "hash",
        "parse",
        "filter",
        "embed",
        "score",
        "llm_summary",
        "retention",
        "model_load",
    }
)

TRACKING_MODE = "machine"
MEASURE_POWER_SECS = 1
CODECARBON_LOG_LEVEL = "error"
CODECARBON_PACKAGE = "codecarbon"
KWH_TO_WH = 1000.0
IDLE_TASK_NAME = "idle"
WARM_UP_TASK_NAME = "warm_up"
NOTES_SEPARATOR = "; "
NOTES_DISABLED = "tracking disabled (GREEN_TRACKING=false)"
CPU_DESCRIPTION_PREFIX = "CPU("
UNKNOWN_VALUE = "unknown"
HARDWARE_CONF_KEY = "hardware"
GPU_MODEL_KEY = "gpu_model"
CPU_MODEL_KEY = "cpu_model"


class UnknownStageError(ValueError):
    """Raised when a stage name is not in `STAGE_NAMES`."""


class NestedStageError(RuntimeError):
    """Raised when a stage is opened while another one is still running."""


@dataclass(frozen=True)
class StageTotals:
    """Accumulated measurement of one stage name.

    Attributes:
        energy_wh: Energy in watt-hours, summed over every call.
        duration_s: Wall-clock seconds, summed over every call.
        count: Number of times the stage ran.
    """

    energy_wh: float = 0.0
    duration_s: float = 0.0
    count: int = 0


@dataclass
class TaskReading:
    """Filled in by one CodeCarbon task when it closes."""

    energy_wh: float = 0.0
    duration_s: float = 0.0
    closed: bool = False


class StageTracker(Protocol):
    """Role-specific interface the pipeline depends on."""

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Return a context manager that measures the named stage."""
        ...

    def totals(self) -> dict[str, StageTotals]:
        """Return a copy of the accumulated totals per stage name."""
        ...

    def measurement_notes(self) -> str:
        """Describe how the numbers were measured (for `measurement_notes`)."""
        ...


def require_known_stage(name: str) -> None:
    """Fail fast when a stage name is not part of SPEC section 7.4.

    Args:
        name: Stage name to validate.

    Raises:
        UnknownStageError: If the name is not in `STAGE_NAMES`.
    """
    if name not in STAGE_NAMES:
        raise UnknownStageError(
            f"Unknown stage {name!r}; expected one of {sorted(STAGE_NAMES)}"
        )


class NoOpStageTracker:
    """Zero-cost tracker for `GREEN_TRACKING=false`: no timing, no CodeCarbon."""

    _SHARED_CONTEXT = nullcontext()

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Validate the name and return a shared do-nothing context manager."""
        require_known_stage(name)
        return self._SHARED_CONTEXT

    def totals(self) -> dict[str, StageTotals]:
        """Return an empty mapping because nothing is measured."""
        return {}

    def measurement_notes(self) -> str:
        """State that tracking was disabled."""
        return NOTES_DISABLED


def read_hardware_descriptions(emissions_tracker: Any) -> list[str]:
    """Return CodeCarbon's per-device descriptions, e.g. `CPU(Windows Emi)`.

    CodeCarbon exposes them only through its configuration dictionary, so this
    is the single place that touches that private attribute.
    """
    configuration = getattr(emissions_tracker, "_conf", None) or {}
    return list(configuration.get(HARDWARE_CONF_KEY, []))


class CodeCarbonStageTracker:
    """Measures stages through the CodeCarbon task API.

    Args:
        country_iso_code: ISO 3166 alpha-3 code used for carbon intensity.
        embed_device: Embedding device recorded in the measurement notes.
        emissions_tracker: Object with CodeCarbon's task API. Built lazily from
            `OfflineEmissionsTracker` when omitted; tests inject a fake.
    """

    def __init__(
        self,
        country_iso_code: str,
        embed_device: str | None = None,
        emissions_tracker: Any = None,
    ) -> None:
        self._embed_device = embed_device
        self._country_iso_code = country_iso_code
        self._emissions_tracker = emissions_tracker or build_offline_tracker(
            country_iso_code
        )
        self._totals: dict[str, StageTotals] = {}
        self._active_stage: str | None = None
        self._warm_up()

    def _warm_up(self) -> None:
        """Pay CodeCarbon's one-time set-up cost now, not inside the first stage.

        Hardware detection and the first task each take seconds; the result of
        the throwaway task is discarded.
        """
        self._emissions_tracker.get_detected_hardware()
        with self._task(WARM_UP_TASK_NAME, TaskReading()):
            pass

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Validate the name and return the context manager measuring it.

        Raises:
            UnknownStageError: Immediately, if the name is not a known stage.
        """
        require_known_stage(name)
        return self._measured(name)

    def totals(self) -> dict[str, StageTotals]:
        """Return a copy of the accumulated totals per stage name."""
        return dict(self._totals)

    def measure_idle(self, seconds: float) -> TaskReading:
        """Measure the machine while this process only sleeps.

        Args:
            seconds: How long to sleep inside one CodeCarbon task.

        Returns:
            The energy and duration of the idle window.
        """
        reading = TaskReading()
        with self._task(IDLE_TASK_NAME, reading):
            time.sleep(seconds)
        return reading

    def measurement_notes(self) -> str:
        """Describe the CPU method, GPU, CodeCarbon version and embed device."""
        hardware = self._emissions_tracker.get_detected_hardware()
        descriptions = read_hardware_descriptions(self._emissions_tracker)
        cpu_method = next(
            (item for item in descriptions if item.startswith(CPU_DESCRIPTION_PREFIX)),
            UNKNOWN_VALUE,
        )
        fields = {
            "cpu_method": cpu_method,
            "cpu_model": hardware.get(CPU_MODEL_KEY) or UNKNOWN_VALUE,
            "gpu": hardware.get(GPU_MODEL_KEY) or UNKNOWN_VALUE,
            "hardware": ",".join(descriptions) or UNKNOWN_VALUE,
            "codecarbon": metadata.version(CODECARBON_PACKAGE),
            "tracking_mode": TRACKING_MODE,
            "country_iso": self._country_iso_code,
        }
        if self._embed_device is not None:
            fields["embed_device"] = self._embed_device
        return NOTES_SEPARATOR.join(f"{key}={value}" for key, value in fields.items())

    @contextmanager
    def _measured(self, name: str) -> Iterator[None]:
        """Run one stage and add its reading to the totals, even on error."""
        reading = TaskReading()
        try:
            with self._task(name, reading):
                yield
        finally:
            if reading.closed:
                self._accumulate(name, reading)

    @contextmanager
    def _task(self, name: str, reading: TaskReading) -> Iterator[None]:
        """Open one CodeCarbon task, always closing it, and fill `reading`."""
        if self._active_stage is not None:
            raise NestedStageError(
                f"Cannot start {name!r} while {self._active_stage!r} is running"
            )
        self._active_stage = name
        started = time.perf_counter()
        try:
            self._emissions_tracker.start_task(name)
            try:
                yield
            finally:
                self._close_task(started, reading)
        finally:
            self._active_stage = None

    def _close_task(self, started: float, reading: TaskReading) -> None:
        """Stop the active task and copy its energy into `reading`."""
        # No name is passed: CodeCarbon renames a repeated task, and only the
        # no-argument call stops the renamed active one.
        emissions = self._emissions_tracker.stop_task()
        reading.duration_s = time.perf_counter() - started
        reading.closed = True
        if emissions is None:
            logger.warning("CodeCarbon returned no data for %s", self._active_stage)
            return
        reading.energy_wh = emissions.energy_consumed * KWH_TO_WH

    def _accumulate(self, name: str, reading: TaskReading) -> None:
        """Add one reading to the running totals of its stage."""
        previous = self._totals.get(name, StageTotals())
        self._totals[name] = replace(
            previous,
            energy_wh=previous.energy_wh + reading.energy_wh,
            duration_s=previous.duration_s + reading.duration_s,
            count=previous.count + 1,
        )


def build_offline_tracker(country_iso_code: str) -> Any:
    """Build the CodeCarbon offline tracker, importing the library lazily.

    `OfflineEmissionsTracker` is used because it accepts `country_iso_code` and
    avoids a network geolocation lookup.
    """
    from codecarbon import OfflineEmissionsTracker

    return OfflineEmissionsTracker(
        tracking_mode=TRACKING_MODE,
        country_iso_code=country_iso_code,
        measure_power_secs=MEASURE_POWER_SECS,
        save_to_file=False,
        log_level=CODECARBON_LOG_LEVEL,
    )


def create_stage_tracker(settings: Settings) -> StageTracker:
    """Choose the tracker implementation from `GREEN_TRACKING`.

    Args:
        settings: Application settings.

    Returns:
        A CodeCarbon-backed tracker when tracking is on, otherwise the no-op.
    """
    if not settings.green_tracking:
        return NoOpStageTracker()
    return CodeCarbonStageTracker(settings.country_iso, settings.embed_device)
