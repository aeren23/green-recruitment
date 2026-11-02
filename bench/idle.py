"""Idle power measurement: `python -m bench idle --seconds N` (SPEC section 7.5).

The measured idle power is stored in `data/bench/idle_power.json`. Later
benchmark runs read it back to compute
`energy_wh_net = energy_wh_gross - idle_power_w * duration_s / 3600`.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.green.tracker import CodeCarbonStageTracker, TaskReading

IDLE_COMMAND = "idle"
IDLE_HELP = "Measure idle power (W) and store it in data/bench/idle_power.json."
SECONDS_OPTION = "--seconds"
SECONDS_HELP = "How long to measure the idle machine (SPEC recommends 120)."
OUTPUT_OPTION = "--output"
OUTPUT_HELP = "Where to store the result (default: %(default)s)."
DEFAULT_IDLE_PATH = Path("data/bench/idle_power.json")
JSON_INDENT = 2
SECONDS_PER_HOUR = 3600.0
ENCODING = "utf-8"
HANDLER_DESTINATION = "handler"
COUNTRY_ISO_FIELD = "country_iso"

EXIT_OK = 0
EXIT_ERROR = 1
ERROR_PREFIX = "Idle measurement error: "


class IdleDataError(ValueError):
    """Raised when idle input or a stored idle file is unusable."""


@dataclass(frozen=True)
class IdlePowerRecord:
    """Idle power and the context it was measured in.

    Attributes:
        idle_power_w: Mean machine power while idle, in watts.
        energy_wh: Energy measured over the window, in watt-hours.
        duration_s: Length of the measured window, in seconds.
        measured_at: UTC timestamp, ISO 8601.
        measurement_notes: CPU method, GPU, CodeCarbon version.
    """

    idle_power_w: float
    energy_wh: float
    duration_s: float
    measured_at: str
    measurement_notes: str


class IdleMeter(Protocol):
    """The part of the tracker that idle measurement needs."""

    def measure_idle(self, seconds: float) -> TaskReading:
        """Measure the idle machine for the given seconds."""
        ...

    def measurement_notes(self) -> str:
        """Describe how the numbers were measured."""
        ...


def compute_idle_power_w(energy_wh: float, duration_s: float) -> float:
    """Convert energy over a window into mean power.

    Args:
        energy_wh: Energy in watt-hours.
        duration_s: Window length in seconds, must be positive.

    Raises:
        IdleDataError: If the duration is not positive.
    """
    if duration_s <= 0:
        raise IdleDataError(f"Duration must be positive, got {duration_s}")
    return energy_wh * SECONDS_PER_HOUR / duration_s


def run_idle_measurement(
    meter: IdleMeter, seconds: float, measured_at: datetime
) -> IdlePowerRecord:
    """Measure idle power with the given meter.

    Args:
        meter: Tracker able to measure an idle window.
        seconds: Requested window length, must be positive.
        measured_at: Timestamp stored in the record (UTC).

    Raises:
        IdleDataError: If `seconds` is not positive.
    """
    if seconds <= 0:
        raise IdleDataError(f"--seconds must be positive, got {seconds}")
    reading = meter.measure_idle(seconds)
    return IdlePowerRecord(
        idle_power_w=compute_idle_power_w(reading.energy_wh, reading.duration_s),
        energy_wh=reading.energy_wh,
        duration_s=reading.duration_s,
        measured_at=measured_at.astimezone(UTC).isoformat(),
        measurement_notes=meter.measurement_notes(),
    )


def save_idle_record(record: IdlePowerRecord, path: Path) -> None:
    """Write the record as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(record), indent=JSON_INDENT)
    path.write_text(payload, encoding=ENCODING)


def load_idle_record(path: Path = DEFAULT_IDLE_PATH) -> IdlePowerRecord:
    """Read the stored idle record.

    Raises:
        IdleDataError: If the file is missing or malformed, with a hint to run
            `python -m bench idle`.
    """
    if not path.is_file():
        raise IdleDataError(f"{path} not found; run `python -m bench idle` first")
    try:
        return IdlePowerRecord(**json.loads(path.read_text(encoding=ENCODING)))
    except (json.JSONDecodeError, TypeError) as error:
        raise IdleDataError(f"{path} is not a valid idle record: {error}") from error


def format_idle_record(record: IdlePowerRecord) -> str:
    """Render the record as a short plain-text report."""
    return "\n".join(
        [
            f"Idle power: {record.idle_power_w:.2f} W",
            f"Energy: {record.energy_wh:.5f} Wh over {record.duration_s:.1f} s",
            f"Measured at: {record.measured_at}",
            f"Notes: {record.measurement_notes}",
        ]
    )


def run_idle(arguments: argparse.Namespace) -> int:
    """Handle `idle`: measure with CodeCarbon, save, print."""
    default_country = Settings.model_fields[COUNTRY_ISO_FIELD].default
    try:
        record = run_idle_measurement(
            CodeCarbonStageTracker(default_country),
            arguments.seconds,
            datetime.now(UTC),
        )
    except IdleDataError as error:
        print(f"{ERROR_PREFIX}{error}", file=sys.stderr)
        return EXIT_ERROR
    save_idle_record(record, arguments.output)
    print(format_idle_record(record))
    return EXIT_OK


def register_idle(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    """Add the `idle` subcommand to the parser."""
    parser = subparsers.add_parser(IDLE_COMMAND, help=IDLE_HELP)
    parser.add_argument(SECONDS_OPTION, type=float, required=True, help=SECONDS_HELP)
    parser.add_argument(
        OUTPUT_OPTION, type=Path, default=DEFAULT_IDLE_PATH, help=OUTPUT_HELP
    )
    parser.set_defaults(**{HANDLER_DESTINATION: run_idle})
