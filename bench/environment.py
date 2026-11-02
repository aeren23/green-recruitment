"""Facts about the machine and the checkout that go into `measurement_notes`.

Everything here is metadata about the measurement conditions (git revision,
power plan, OneDrive, Ollama); none of it touches CV data. The probe is a small
interface so tests can inject fixed values.
"""

import ctypes
import locale
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
COMMAND_TIMEOUT_S = 15.0
UNKNOWN = "unknown"
WINDOWS_PLATFORM = "win32"
ONEDRIVE_IMAGE_NAME = "OneDrive.exe"
GIT_REVISION = "git"
GIT_DIRTY = "git_dirty"
POWER_PLAN = "power_plan"
ONEDRIVE_RUNNING = "onedrive_running"
BOOLEAN_TRUE = "true"
BOOLEAN_FALSE = "false"
GIT_SHORT_HASH_COMMAND = ("git", "rev-parse", "--short", "HEAD")
GIT_STATUS_COMMAND = ("git", "status", "--porcelain")
POWER_PLAN_COMMAND = ("powercfg", "/getactivescheme")
TASKLIST_COMMAND = ("tasklist", "/FI", f"IMAGENAME eq {ONEDRIVE_IMAGE_NAME}", "/FO", "CSV", "/NH")
OLLAMA_PS_UNAVAILABLE = "ollama ps unavailable"
OLLAMA_EXECUTABLE = "ollama"
OLLAMA_PS_ARGUMENTS = ("ps",)
LOCAL_APP_DATA_VARIABLE = "LOCALAPPDATA"
OLLAMA_INSTALL_RELATIVE_PATH = Path("Programs") / "Ollama" / "ollama.exe"


class EnvironmentProbe(Protocol):
    """Describes the measurement conditions as `key -> value` strings."""

    def describe(self) -> Mapping[str, str]:
        """Return git revision, dirty flag, power plan and OneDrive state."""
        ...

    def ollama_ps(self) -> str:
        """Return the output of `ollama ps`, or a short reason it is missing."""
        ...


def console_encoding() -> str:
    """The code page console tools print in (the OEM code page on Windows)."""
    if sys.platform == WINDOWS_PLATFORM:
        return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    return locale.getpreferredencoding(False)


def run_command(command: tuple[str, ...], working_directory: Path | None = None) -> str | None:
    """Run a command and return its decoded stdout, or `None` if it did not work."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            cwd=working_directory,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.info("%s could not run: %s", command[0], error.__class__.__name__)
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode(console_encoding(), errors="replace")


def find_ollama_executable() -> str | None:
    """The `ollama` executable: on `PATH`, else the per-user Windows install folder."""
    on_path = shutil.which(OLLAMA_EXECUTABLE)
    if on_path:
        return on_path
    local_app_data = os.environ.get(LOCAL_APP_DATA_VARIABLE)
    if not local_app_data:
        return None
    installed = Path(local_app_data) / OLLAMA_INSTALL_RELATIVE_PATH
    return str(installed) if installed.is_file() else None


def parse_power_plan_name(output: str) -> str:
    """The plan name in parentheses of `powercfg /getactivescheme` output."""
    start, end = output.rfind("("), output.rfind(")")
    if start == -1 or end < start:
        return UNKNOWN
    return output[start + 1 : end].strip() or UNKNOWN


def format_bool(value: bool) -> str:
    """Lowercase boolean for the notes."""
    return BOOLEAN_TRUE if value else BOOLEAN_FALSE


class SystemEnvironmentProbe:
    """Reads the real machine: `git`, `powercfg`, `tasklist` and `ollama`."""

    def describe(self) -> Mapping[str, str]:
        """Collect the measurement conditions; unknown values say `unknown`."""
        revision = run_command(GIT_SHORT_HASH_COMMAND, REPOSITORY_ROOT)
        status = run_command(GIT_STATUS_COMMAND, REPOSITORY_ROOT)
        plan = run_command(POWER_PLAN_COMMAND)
        tasks = run_command(TASKLIST_COMMAND)
        return {
            GIT_REVISION: revision.strip() if revision else UNKNOWN,
            GIT_DIRTY: UNKNOWN if status is None else format_bool(bool(status.strip())),
            POWER_PLAN: parse_power_plan_name(plan) if plan else UNKNOWN,
            ONEDRIVE_RUNNING: (
                UNKNOWN if tasks is None else format_bool(ONEDRIVE_IMAGE_NAME in tasks)
            ),
        }

    def ollama_ps(self) -> str:
        """The `ollama ps` table (model names and placement; no CV data)."""
        executable = find_ollama_executable()
        if executable is None:
            return OLLAMA_PS_UNAVAILABLE
        output = run_command((executable, *OLLAMA_PS_ARGUMENTS))
        return output.strip() if output else OLLAMA_PS_UNAVAILABLE
