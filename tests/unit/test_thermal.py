import json
from pathlib import Path

import pytest
from modeldeck.thermal import (
    AdmissionAction,
    TemperatureSnapshot,
    ThermalAdmissionController,
    ThermalGuard,
    ThermalGuardError,
    ThermalPolicyConfig,
    ThermalPolicyManager,
    ThermalReading,
    ThermalState,
    ThermalStateMachine,
    WorkloadClass,
    WorkloadRequest,
    capacity_for_state,
    classify_workload,
    read_temperatures,
    read_thermal_status,
    read_thermal_workload_activity,
    select_control_reading,
    write_thermal_status,
    write_thermal_workload_activity,
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


def policy(**changes) -> ThermalPolicyConfig:
    values = {
        "minimum_state_dwell_seconds": 0,
        "recovery_step_seconds": 0,
        "recovery_reading_count": 3,
        "host_policy_status_enabled": False,
    }
    values.update(changes)
    return ThermalPolicyConfig(**values)


def reading(temperature: float, observed: float, sensor: str = "k10temp:Tctl") -> ThermalReading:
    return ThermalReading(sensor, temperature, observed)


def test_configuration_requires_ordered_entry_and_recovery_thresholds() -> None:
    with pytest.raises(ValueError, match="entry thresholds"):
        ThermalPolicyConfig(hot_threshold_c=74)
    with pytest.raises(ValueError, match="recovery thresholds"):
        ThermalPolicyConfig(hot_recovery_c=71)


def test_state_machine_escalates_immediately_and_recovers_one_step_at_a_time() -> None:
    machine = ThermalStateMachine(policy(), monotonic=lambda: 0)
    for moment in range(3):
        machine.update(reading(70, moment), now=moment)
    assert machine.state is ThermalState.NORMAL

    machine.update(reading(83.5, 3), now=3)
    assert machine.state is ThermalState.VERY_HOT
    for moment in range(4, 7):
        machine.update(reading(78, moment), now=moment)
    assert machine.state is ThermalState.HOT
    for moment in range(7, 10):
        machine.update(reading(75, moment), now=moment)
    assert machine.state is ThermalState.WARM
    for moment in range(10, 13):
        machine.update(reading(71, moment), now=moment)
    assert machine.state is ThermalState.NORMAL


def test_state_machine_requires_dwell_and_distinct_recovery_readings() -> None:
    machine = ThermalStateMachine(
        policy(minimum_state_dwell_seconds=10, recovery_reading_count=2), monotonic=lambda: 0
    )
    machine.update(reading(81, 0), now=0)
    assert machine.state is ThermalState.HOT
    machine.update(reading(75, 1), now=1)
    machine.update(reading(75, 1), now=11)
    assert machine.state is ThermalState.HOT
    machine.update(reading(75, 11), now=11)
    assert machine.state is ThermalState.WARM


def test_degraded_telemetry_does_not_restore_warm_capacity_on_one_reading() -> None:
    machine = ThermalStateMachine(policy(), monotonic=lambda: 0)
    machine.update(reading(76, 0), now=0)
    machine.update(reading(76, 1), now=1)
    assert machine.state is ThermalState.TELEMETRY_DEGRADED
    machine.update(reading(76, 2), now=2)
    assert machine.state is ThermalState.WARM


def test_state_machine_fails_closed_for_stale_invalid_and_changed_sensor_readings() -> None:
    machine = ThermalStateMachine(policy(telemetry_stale_seconds=5), monotonic=lambda: 0)
    machine.update(reading(70, 0), now=6)
    assert machine.state is ThermalState.TELEMETRY_DEGRADED
    assert machine.reason_code == "thermal_telemetry_stale"

    machine.update(reading(200, 7), now=7)
    assert machine.reason_code == "thermal_reading_invalid"

    pinned = ThermalStateMachine(policy(), monotonic=lambda: 0)
    pinned.update(reading(75, 0), now=0)
    pinned.update(reading(70, 1, "amdgpu:edge"), now=1)
    assert pinned.reason_code == "thermal_sensor_identity_changed"


def test_critical_state_has_precedence_and_does_not_recover_on_one_reading() -> None:
    machine = ThermalStateMachine(policy(), monotonic=lambda: 0)
    machine.update(reading(85, 0), now=0)
    assert machine.state is ThermalState.CRITICAL
    machine.update(reading(70, 1), now=1)
    assert machine.state is ThermalState.CRITICAL


@pytest.mark.asyncio
async def test_manager_invokes_critical_worker_isolation_once_per_transition(tmp_path: Path) -> None:
    calls = 0

    async def isolate_workers() -> None:
        nonlocal calls
        calls += 1

    manager = ThermalPolicyManager(
        policy(),
        data_dir=tmp_path,
        telemetry_reader=lambda: [{"source": "k10temp", "label": "Tctl", "celsius": 86}],
        monotonic=lambda: 10,
        critical_handler=isolate_workers,
    )

    first = await manager.poll_once()
    second = await manager.poll_once()

    assert first["state"] == "critical"
    assert second["state"] == "critical"
    assert calls == 1


@pytest.mark.parametrize(
    ("state", "workload", "action"),
    [
        (ThermalState.WARM, WorkloadClass.BENCHMARK, AdmissionAction.PAUSE),
        (ThermalState.HOT, WorkloadClass.MODEL_LOAD, AdmissionAction.REJECT_RETRYABLE),
        (ThermalState.HOT, WorkloadClass.INTERACTIVE, AdmissionAction.ALLOW_DEGRADED),
        (ThermalState.VERY_HOT, WorkloadClass.INTERACTIVE, AdmissionAction.REJECT_RETRYABLE),
        (ThermalState.CRITICAL, WorkloadClass.INTERACTIVE, AdmissionAction.CANCEL),
        (
            ThermalState.TELEMETRY_DEGRADED,
            WorkloadClass.HEAVY_INFERENCE,
            AdmissionAction.REJECT_RETRYABLE,
        ),
    ],
)
def test_admission_prioritises_interactive_work_and_pauses_background_first(state, workload, action) -> None:
    decision = ThermalAdmissionController(policy()).evaluate(
        WorkloadRequest("work", workload), state, temperature_c=81
    )
    assert decision.action is action
    assert decision.reason_code


def test_warm_model_load_does_not_overlap_active_or_unknown_inference() -> None:
    controller = ThermalAdmissionController(policy())
    workload = WorkloadRequest("load", WorkloadClass.MODEL_LOAD)

    assert (
        controller.evaluate(workload, ThermalState.WARM, active_heavy_workloads=0).action
        is AdmissionAction.ALLOW
    )
    assert (
        controller.evaluate(workload, ThermalState.WARM, active_heavy_workloads=1).reason_code
        == "model_load_blocked_by_active_inference"
    )
    assert (
        controller.evaluate(workload, ThermalState.WARM, active_heavy_workloads=None).action
        is AdmissionAction.REJECT_RETRYABLE
    )


def test_capacity_reduces_without_raising_limits_for_degraded_telemetry() -> None:
    configured = policy(configured_heavy_concurrency=4, configured_background_concurrency=2)
    assert capacity_for_state(configured, ThermalState.NORMAL).heavy_inference_concurrency == 4
    assert capacity_for_state(configured, ThermalState.HOT).heavy_inference_concurrency == 1
    assert capacity_for_state(configured, ThermalState.VERY_HOT).heavy_inference_concurrency == 0
    assert capacity_for_state(configured, ThermalState.TELEMETRY_DEGRADED).model_load_concurrency == 0


def test_workload_classification_uses_compute_intent() -> None:
    assert classify_workload("health") is WorkloadClass.LIGHT_CONTROL
    assert classify_workload("worker_start") is WorkloadClass.MODEL_LOAD
    assert classify_workload("scene_analysis") is WorkloadClass.INTERACTIVE
    assert classify_workload("scene_analysis", automatic=True) is WorkloadClass.BACKGROUND
    assert classify_workload("benchmark") is WorkloadClass.BENCHMARK


def test_sensor_selection_prefers_the_existing_apu_control_sensor() -> None:
    selected = select_control_reading(
        [
            {"source": "amdgpu", "label": "edge", "celsius": 80},
            {"source": "k10temp", "label": "Tctl", "celsius": 75},
        ],
        None,
        4,
    )
    assert selected is not None
    assert (selected.sensor_id, selected.celsius, selected.observed_monotonic) == (
        "k10temp:Tctl",
        75,
        4,
    )


def test_shared_status_reader_fails_closed_when_snapshot_is_stale(tmp_path: Path) -> None:
    configured = policy(telemetry_stale_seconds=5)
    path = tmp_path / "thermal-status.json"
    write_thermal_status(
        path,
        {
            "state": "normal",
            "published_monotonic": 10,
            "telemetry_age_seconds": 1,
            "heavy_concurrency_limit": 2,
        },
    )
    assert read_thermal_status(path, configured, monotonic=lambda: 12)["state"] == "normal"
    degraded = read_thermal_status(path, configured, monotonic=lambda: 20)
    assert degraded["state"] == "telemetry_degraded"
    assert degraded["model_loading_allowed"] is False


def test_shared_workload_activity_expires_instead_of_reporting_false_idle(tmp_path: Path) -> None:
    configured = policy(telemetry_stale_seconds=5)
    path = tmp_path / "thermal-workloads.json"
    write_thermal_workload_activity(path, 2)
    assert read_thermal_workload_activity(path, configured) == 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["published_monotonic"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_thermal_workload_activity(path, configured, monotonic=lambda: 10) is None
