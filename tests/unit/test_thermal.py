from pathlib import Path

import pytest
from modeldeck.thermal import (
    TemperatureSnapshot,
    ThermalGuard,
    ThermalGuardError,
    read_temperatures,
)


def write_sensor(root: Path, device: str, source: str, label: str, value: int) -> None:
    directory = root / device
    directory.mkdir()
    (directory / "name").write_text(source, encoding="utf-8")
    (directory / "temp1_label").write_text(label, encoding="utf-8")
    (directory / "temp1_input").write_text(str(value), encoding="utf-8")


def test_reads_required_gpu_edge_and_cpu_package_sensors(tmp_path: Path) -> None:
    write_sensor(tmp_path, "hwmon0", "amdgpu", "edge", 51_000)
    write_sensor(tmp_path, "hwmon1", "k10temp", "Tctl", 72_500)

    assert read_temperatures(tmp_path) == TemperatureSnapshot(51, 72.5)


def test_temperature_monitor_fails_closed_when_a_required_sensor_is_missing(tmp_path: Path) -> None:
    write_sensor(tmp_path, "hwmon0", "amdgpu", "edge", 45_000)

    with pytest.raises(ThermalGuardError, match="sensors are unavailable"):
        read_temperatures(tmp_path)


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (TemperatureSnapshot(80, 60), "gpu_thermal_limit"),
        (TemperatureSnapshot(60, 95), "cpu_thermal_limit"),
    ],
)
def test_generation_termination_limits_are_code_owned(snapshot, reason) -> None:
    guard = ThermalGuard(lambda: snapshot)

    assert guard.termination_reason() == (reason, snapshot)
