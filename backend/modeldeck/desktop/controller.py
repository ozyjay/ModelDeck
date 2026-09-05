"""Dependency-light control primitives used by the GTK desktop shell."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MANAGEMENT_HEALTH_URL = "http://127.0.0.1:3600/api/health"
SYSTEMCTL = "/usr/bin/systemctl"
TARGET = "modeldeck.target"


class DesktopServiceError(RuntimeError):
    """The desktop shell could not control or reach ModelDeck."""


@dataclass(frozen=True)
class ServiceHealth:
    status: str
    build_id: str | None


class ServiceController:
    """Use only fixed systemd-user operations and the loopback health endpoint."""

    def __init__(
        self,
        *,
        run: Callable[[Sequence[str]], None] | None = None,
        prepare: Callable[[], None] | None = None,
        read_health: Callable[[], ServiceHealth] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._run = run or _run_systemctl
        self._prepare = prepare or prepare_service_directories
        self._read_health = read_health or read_management_health
        self._sleep = sleep

    def start(self) -> None:
        self._prepare()
        self._run((SYSTEMCTL, "--user", "daemon-reload"))
        self._run((SYSTEMCTL, "--user", "start", TARGET))

    def restart(self) -> None:
        self._prepare()
        self._run((SYSTEMCTL, "--user", "daemon-reload"))
        self._run((SYSTEMCTL, "--user", "restart", TARGET))

    def stop(self) -> None:
        self._run((SYSTEMCTL, "--user", "stop", TARGET))

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 0.25,
    ) -> ServiceHealth:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                health = self._read_health()
                if health.status == "ok":
                    return health
                last_error = DesktopServiceError(f"Management service reported {health.status!r}")
            except DesktopServiceError as error:
                last_error = error
            self._sleep(interval_seconds)
        detail = str(last_error) if last_error else "no response from the management service"
        raise DesktopServiceError(f"ModelDeck did not become ready: {detail}")


def should_prompt_for_restart(*, installed_build_id: str, running_build_id: str | None) -> bool:
    """True only when an already-running service predates the installed package."""

    return bool(running_build_id and running_build_id != installed_build_id)


def prepare_service_directories(*, home: Path | None = None) -> None:
    """Create the fixed writable paths required by the packaged systemd sandbox."""

    user_home = home or Path.home()
    directories = (
        user_home / ".local" / "share" / "modeldeck",
        user_home / ".local" / "state" / "modeldeck",
    )
    try:
        for directory in directories:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        raise DesktopServiceError("Could not prepare the ModelDeck data directories") from error


def _run_systemctl(command: Sequence[str]) -> None:
    try:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, timeout=15)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise DesktopServiceError("Could not control the ModelDeck user services") from error


def read_management_health() -> ServiceHealth:
    try:
        with urlopen(MANAGEMENT_HEALTH_URL, timeout=2) as response:  # noqa: S310 - fixed loopback URL
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
        raise DesktopServiceError("Management service is not reachable on loopback") from error
    if not isinstance(payload, dict):
        raise DesktopServiceError("Management service returned an invalid health response")
    status = payload.get("status")
    if not isinstance(status, str):
        raise DesktopServiceError("Management service did not report a status")
    build_id = payload.get("build_id")
    return ServiceHealth(status=status, build_id=build_id if isinstance(build_id, str) else None)
