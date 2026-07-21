from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GPU_START_LIMIT_C = 55.0
CPU_START_LIMIT_C = 75.0
GPU_TERMINATE_LIMIT_C = 80.0
CPU_TERMINATE_LIMIT_C = 95.0


@dataclass(frozen=True)
class TemperatureSnapshot:
    gpu_edge_celsius: float
    cpu_package_celsius: float


class ThermalGuardError(RuntimeError):
    def __init__(self, code: str, message: str, snapshot: TemperatureSnapshot | None = None) -> None:
        self.code = code
        self.snapshot = snapshot
        super().__init__(message)


class ThermalGuard:
    def __init__(self, reader: Callable[[], TemperatureSnapshot] | None = None) -> None:
        self._reader = reader or read_temperatures

    def sample(self) -> TemperatureSnapshot:
        try:
            return self._reader()
        except ThermalGuardError:
            raise
        except Exception as error:
            raise ThermalGuardError(
                "thermal_monitor_unavailable",
                "Required GPU and CPU temperature sensors are unavailable.",
            ) from error

    def require_start_safe(self) -> TemperatureSnapshot:
        snapshot = self.sample()
        if snapshot.gpu_edge_celsius > GPU_START_LIMIT_C or snapshot.cpu_package_celsius > CPU_START_LIMIT_C:
            raise ThermalGuardError(
                "thermal_cooldown_required",
                "The system must cool before local speech synthesis can begin.",
                snapshot,
            )
        return snapshot

    def termination_reason(
        self, snapshot: TemperatureSnapshot | None = None
    ) -> tuple[str, TemperatureSnapshot] | None:
        snapshot = snapshot or self.sample()
        if snapshot.gpu_edge_celsius >= GPU_TERMINATE_LIMIT_C:
            return "gpu_thermal_limit", snapshot
        if snapshot.cpu_package_celsius >= CPU_TERMINATE_LIMIT_C:
            return "cpu_thermal_limit", snapshot
        return None


def read_temperatures(hwmon_root: Path = Path("/sys/class/hwmon")) -> TemperatureSnapshot:
    gpu: float | None = None
    cpu: float | None = None
    try:
        devices = list(hwmon_root.glob("hwmon*"))
    except OSError as error:
        raise ThermalGuardError(
            "thermal_monitor_unavailable",
            "Required GPU and CPU temperature sensors are unavailable.",
        ) from error
    for device in devices:
        try:
            source = (device / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for input_path in device.glob("temp*_input"):
            label_path = input_path.with_name(input_path.name.replace("_input", "_label"))
            try:
                label = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else ""
                value = float(input_path.read_text(encoding="utf-8").strip()) / 1000
            except (OSError, ValueError):
                continue
            if source == "amdgpu" and label.casefold() == "edge":
                gpu = value
            elif source == "k10temp" and label.casefold() in {"tctl", "tdie"}:
                cpu = value if cpu is None else max(cpu, value)
    if gpu is None or cpu is None:
        raise ThermalGuardError(
            "thermal_monitor_unavailable",
            "Required GPU edge and CPU package temperature sensors are unavailable.",
        )
    return TemperatureSnapshot(gpu_edge_celsius=gpu, cpu_package_celsius=cpu)
