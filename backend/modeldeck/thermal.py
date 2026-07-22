from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

GPU_START_LIMIT_C = 55.0
CPU_START_LIMIT_C = 75.0
GPU_TERMINATE_LIMIT_C = 80.0
CPU_TERMINATE_LIMIT_C = 95.0
THERMAL_STATUS_FILENAME = "thermal-status.json"
THERMAL_WORKLOAD_FILENAME = "thermal-workloads.json"
_MIN_CREDIBLE_C = -20.0
_MAX_CREDIBLE_C = 125.0
_LOGGER = logging.getLogger(__name__)


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
    """Existing critical safety guard; throttling must not weaken these limits."""

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


class ThermalState(StrEnum):
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    VERY_HOT = "very_hot"
    CRITICAL = "critical"
    TELEMETRY_DEGRADED = "telemetry_degraded"


class WorkloadClass(StrEnum):
    LIGHT_CONTROL = "light_control"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    BENCHMARK = "benchmark"
    MODEL_LOAD = "model_load"
    HEAVY_INFERENCE = "heavy_inference"


class AdmissionAction(StrEnum):
    ALLOW = "allow"
    ALLOW_DEGRADED = "allow_degraded"
    QUEUE = "queue"
    PAUSE = "pause"
    REJECT_RETRYABLE = "reject_retryable"
    CANCEL = "cancel"


@dataclass(frozen=True)
class ThermalPolicyConfig:
    enabled: bool = True
    sensor_id: str | None = None
    warm_threshold_c: float = 75.0
    hot_threshold_c: float = 80.0
    very_hot_threshold_c: float = 83.0
    critical_threshold_c: float = 85.0
    warm_recovery_c: float = 72.0
    hot_recovery_c: float = 76.0
    very_hot_recovery_c: float = 79.0
    telemetry_stale_seconds: float = 15.0
    poll_interval_seconds: float = 5.0
    minimum_state_dwell_seconds: float = 30.0
    recovery_step_seconds: float = 30.0
    recovery_reading_count: int = 3
    configured_heavy_concurrency: int = 2
    configured_background_concurrency: int = 1
    warm_scene_interval_seconds: float = 5.0
    hot_scene_interval_seconds: float = 10.0
    stop_automatic_scene_when_very_hot: bool = True
    host_policy_status_enabled: bool = True
    host_policy_service_name: str = "framework-thermal-policy.service"

    def __post_init__(self) -> None:
        entries = (
            self.warm_threshold_c,
            self.hot_threshold_c,
            self.very_hot_threshold_c,
            self.critical_threshold_c,
        )
        recoveries = (self.warm_recovery_c, self.hot_recovery_c, self.very_hot_recovery_c)
        if not all(math.isfinite(value) for value in (*entries, *recoveries)):
            raise ValueError("Thermal thresholds must be finite numbers")
        if not (_MIN_CREDIBLE_C < entries[0] < entries[1] < entries[2] < entries[3] <= _MAX_CREDIBLE_C):
            raise ValueError("Thermal entry thresholds must be ordered warm < hot < very hot < critical")
        if not (
            _MIN_CREDIBLE_C < recoveries[0] < recoveries[1] < recoveries[2]
            and recoveries[0] < entries[0]
            and recoveries[1] < entries[1]
            and recoveries[2] < entries[2]
        ):
            raise ValueError("Thermal recovery thresholds must be ordered and below their entry thresholds")
        if self.telemetry_stale_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("Thermal telemetry and polling intervals must be greater than zero")
        if self.minimum_state_dwell_seconds < 0 or self.recovery_step_seconds < 0:
            raise ValueError("Thermal dwell and recovery intervals cannot be negative")
        if self.recovery_reading_count < 1:
            raise ValueError("Thermal recovery requires at least one reading")
        if self.configured_heavy_concurrency < 1 or self.configured_background_concurrency < 1:
            raise ValueError("Configured thermal concurrency must be at least one")
        if not self.host_policy_service_name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@_.-"
            for character in self.host_policy_service_name
        ):
            raise ValueError("Host policy service name contains unsupported characters")


@dataclass(frozen=True)
class ThermalReading:
    sensor_id: str
    celsius: float
    observed_monotonic: float
    observed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class WorkloadRequest:
    workload_id: str
    workload_class: WorkloadClass
    automatic: bool = False
    backend: str | None = None
    model: str | None = None


def classify_workload(operation: str, *, automatic: bool = False) -> WorkloadClass:
    """Classify trusted operations by expected compute impact, not caller-provided commands."""
    normalised = operation.strip().casefold()
    if normalised in {"health", "status", "metrics", "cancel", "logs"}:
        return WorkloadClass.LIGHT_CONTROL
    if normalised in {"benchmark", "compatibility_test", "stability_test"}:
        return WorkloadClass.BENCHMARK
    if normalised in {"worker_start", "model_load", "preload"}:
        return WorkloadClass.MODEL_LOAD
    if automatic:
        return WorkloadClass.BACKGROUND
    if normalised in {
        "chat",
        "completion",
        "scene_analysis",
        "translation",
        "speech",
        "refinement",
        "diffusion",
    }:
        return WorkloadClass.INTERACTIVE
    return WorkloadClass.HEAVY_INFERENCE


@dataclass(frozen=True)
class ThermalCapacity:
    heavy_inference_concurrency: int
    model_load_concurrency: int
    background_concurrency: int


@dataclass(frozen=True)
class AdmissionDecision:
    action: AdmissionAction
    reason_code: str
    reason: str
    thermal_state: ThermalState
    temperature_c: float | None
    suggested_retry_seconds: float | None
    degradation: Mapping[str, Any]
    timestamp: str
    telemetry_age_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["thermal_state"] = self.thermal_state.value
        return value


class ThermalAdmissionError(RuntimeError):
    def __init__(self, decision: AdmissionDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)


@dataclass(frozen=True)
class ThermalTransition:
    timestamp: str
    previous_state: ThermalState
    new_state: ThermalState
    temperature_c: float | None
    sensor_id: str | None
    telemetry_age_seconds: float | None
    reason_code: str


class ThermalStateMachine:
    _SEVERITY = {
        ThermalState.NORMAL: 0,
        ThermalState.WARM: 1,
        ThermalState.HOT: 2,
        ThermalState.VERY_HOT: 3,
        ThermalState.CRITICAL: 4,
    }

    def __init__(
        self,
        config: ThermalPolicyConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        now = monotonic()
        self.state = ThermalState.TELEMETRY_DEGRADED if config.enabled else ThermalState.NORMAL
        self.state_entered_monotonic = now
        self.last_recovery_monotonic = now
        self.reading: ThermalReading | None = None
        self.reason_code = "awaiting_fresh_telemetry" if config.enabled else "thermal_throttling_disabled"
        self.recovery_readings = 0
        self.transitions: list[ThermalTransition] = []
        self._last_observation: tuple[str, float] | None = None
        self._selected_sensor_id = config.sensor_id

    def update(self, reading: ThermalReading | None, *, now: float | None = None) -> ThermalState:
        now = self._monotonic() if now is None else now
        if not self.config.enabled:
            self.state = ThermalState.NORMAL
            self.reason_code = "thermal_throttling_disabled"
            return self.state
        invalid_reason = self._invalid_reason(reading, now)
        if invalid_reason is not None:
            self.reading = reading
            self.recovery_readings = 0
            self._transition(ThermalState.TELEMETRY_DEGRADED, invalid_reason, now)
            return self.state
        assert reading is not None
        observation = (reading.sensor_id, reading.observed_monotonic)
        if observation == self._last_observation:
            return self.state
        self._last_observation = observation
        self.reading = reading
        if self._selected_sensor_id is None:
            self._selected_sensor_id = reading.sensor_id

        target = self._entry_state(reading.celsius)
        if target is ThermalState.CRITICAL:
            self.recovery_readings = 0
            self._transition(target, "critical_threshold_exceeded", now)
            return self.state
        if self.state is not ThermalState.TELEMETRY_DEGRADED and (
            self._SEVERITY[target] > self._SEVERITY[self.state]
        ):
            self.recovery_readings = 0
            self._transition(target, f"{target.value}_threshold_exceeded", now)
            return self.state
        if self.state is ThermalState.TELEMETRY_DEGRADED:
            self._recover_from_degraded(target, now)
            return self.state
        self._recover_one_step(reading.celsius, now)
        return self.state

    def refresh(self, *, now: float | None = None) -> ThermalState:
        """Re-evaluate freshness without treating the same sample as a recovery reading."""
        now = self._monotonic() if now is None else now
        if self._invalid_reason(self.reading, now) is not None:
            return self.update(None, now=now)
        return self.state

    def telemetry_age(self, *, now: float | None = None) -> float | None:
        if self.reading is None:
            return None
        now = self._monotonic() if now is None else now
        return max(0.0, now - self.reading.observed_monotonic)

    def _invalid_reason(self, reading: ThermalReading | None, now: float) -> str | None:
        if reading is None:
            return "thermal_telemetry_unavailable"
        if not reading.sensor_id:
            return "thermal_sensor_missing"
        if self._selected_sensor_id is not None and reading.sensor_id != self._selected_sensor_id:
            return "thermal_sensor_identity_changed"
        if (
            not isinstance(reading.celsius, (int, float))
            or isinstance(reading.celsius, bool)
            or not math.isfinite(reading.celsius)
            or not (_MIN_CREDIBLE_C <= reading.celsius <= _MAX_CREDIBLE_C)
        ):
            return "thermal_reading_invalid"
        age = now - reading.observed_monotonic
        if age < 0:
            return "thermal_clock_invalid"
        if age > self.config.telemetry_stale_seconds:
            return "thermal_telemetry_stale"
        return None

    def _entry_state(self, temperature: float) -> ThermalState:
        if temperature >= self.config.critical_threshold_c:
            return ThermalState.CRITICAL
        if temperature >= self.config.very_hot_threshold_c:
            return ThermalState.VERY_HOT
        if temperature >= self.config.hot_threshold_c:
            return ThermalState.HOT
        if temperature >= self.config.warm_threshold_c:
            return ThermalState.WARM
        return ThermalState.NORMAL

    def _recover_from_degraded(self, target: ThermalState, now: float) -> None:
        self.recovery_readings += 1
        if target in {ThermalState.VERY_HOT, ThermalState.HOT}:
            self._transition(target, f"{target.value}_threshold_exceeded", now)
            return
        if self._recovery_ready(now):
            self._transition(target, "fresh_telemetry_restored", now)

    def _recover_one_step(self, temperature: float, now: float) -> None:
        recovery = {
            ThermalState.CRITICAL: (self.config.very_hot_recovery_c, ThermalState.VERY_HOT),
            ThermalState.VERY_HOT: (self.config.very_hot_recovery_c, ThermalState.HOT),
            ThermalState.HOT: (self.config.hot_recovery_c, ThermalState.WARM),
            ThermalState.WARM: (self.config.warm_recovery_c, ThermalState.NORMAL),
        }.get(self.state)
        if recovery is None:
            self.recovery_readings = 0
            return
        threshold, target = recovery
        if temperature >= threshold:
            self.recovery_readings = 0
            return
        self.recovery_readings += 1
        if self._recovery_ready(now):
            self._transition(target, f"recovered_below_{self.state.value}_threshold", now)

    def _recovery_ready(self, now: float) -> bool:
        return (
            self.recovery_readings >= self.config.recovery_reading_count
            and now - self.state_entered_monotonic >= self.config.minimum_state_dwell_seconds
            and now - self.last_recovery_monotonic >= self.config.recovery_step_seconds
        )

    def _transition(self, state: ThermalState, reason_code: str, now: float) -> None:
        self.reason_code = reason_code
        if state is self.state:
            return
        transition = ThermalTransition(
            timestamp=datetime.now(UTC).isoformat(),
            previous_state=self.state,
            new_state=state,
            temperature_c=self.reading.celsius if self.reading else None,
            sensor_id=self.reading.sensor_id if self.reading else self._selected_sensor_id,
            telemetry_age_seconds=self.telemetry_age(now=now),
            reason_code=reason_code,
        )
        self.state = state
        self.state_entered_monotonic = now
        self.last_recovery_monotonic = now
        self.recovery_readings = 0
        self.transitions.append(transition)
        _LOGGER.warning(
            "ModelDeck thermal state changed",
            extra={"thermal_transition": asdict(transition)},
        )


def capacity_for_state(config: ThermalPolicyConfig, state: ThermalState) -> ThermalCapacity:
    if not config.enabled:
        return ThermalCapacity(
            config.configured_heavy_concurrency, 1, config.configured_background_concurrency
        )
    if state is ThermalState.NORMAL:
        return ThermalCapacity(
            config.configured_heavy_concurrency, 1, config.configured_background_concurrency
        )
    if state is ThermalState.WARM:
        return ThermalCapacity(config.configured_heavy_concurrency, 1, 0)
    if state in {ThermalState.HOT, ThermalState.TELEMETRY_DEGRADED}:
        return ThermalCapacity(1, 0, 0)
    return ThermalCapacity(0, 0, 0)


class ThermalAdmissionController:
    def __init__(self, config: ThermalPolicyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        request: WorkloadRequest,
        state: ThermalState,
        *,
        temperature_c: float | None = None,
        telemetry_age_seconds: float | None = None,
        active_heavy_workloads: int | None = 0,
    ) -> AdmissionDecision:
        action, code, message = self._outcome(request, state, active_heavy_workloads)
        retry = (
            self.config.recovery_step_seconds
            if action
            in {
                AdmissionAction.QUEUE,
                AdmissionAction.PAUSE,
                AdmissionAction.REJECT_RETRYABLE,
            }
            else None
        )
        return AdmissionDecision(
            action=action,
            reason_code=code,
            reason=message,
            thermal_state=state,
            temperature_c=temperature_c,
            suggested_retry_seconds=retry,
            degradation=self._degradation(request, state, action),
            timestamp=datetime.now(UTC).isoformat(),
            telemetry_age_seconds=telemetry_age_seconds,
        )

    def _outcome(
        self,
        request: WorkloadRequest,
        state: ThermalState,
        active_heavy_workloads: int | None,
    ) -> tuple[AdmissionAction, str, str]:
        workload = request.workload_class
        if not self.config.enabled:
            return AdmissionAction.ALLOW, "thermal_throttling_disabled", "Thermal throttling is disabled."
        if workload is WorkloadClass.LIGHT_CONTROL:
            return AdmissionAction.ALLOW, "light_control_allowed", "Lightweight control work is permitted."
        if state is ThermalState.CRITICAL:
            return (
                AdmissionAction.CANCEL,
                "critical_thermal_limit",
                "The existing critical thermal limit is active.",
            )
        if workload in {WorkloadClass.BACKGROUND, WorkloadClass.BENCHMARK}:
            if state is not ThermalState.NORMAL:
                return (
                    AdmissionAction.PAUSE,
                    "background_paused_for_thermal_state",
                    "Background benchmarking is paused while the Framework Desktop cools.",
                )
            return AdmissionAction.ALLOW, "thermal_capacity_available", "Background capacity is available."
        if workload is WorkloadClass.MODEL_LOAD:
            if state is ThermalState.WARM and (active_heavy_workloads is None or active_heavy_workloads > 0):
                return (
                    AdmissionAction.REJECT_RETRYABLE,
                    "model_load_blocked_by_active_inference",
                    "Model loading cannot overlap heavy inference while thermal mitigation is active.",
                )
            if state in {
                ThermalState.HOT,
                ThermalState.VERY_HOT,
                ThermalState.TELEMETRY_DEGRADED,
            }:
                return (
                    AdmissionAction.REJECT_RETRYABLE,
                    "model_load_blocked_for_thermal_state",
                    "Model loading is blocked until safe thermal capacity is restored.",
                )
            return AdmissionAction.ALLOW, "model_load_serialised", "The model load may proceed serially."
        if state is ThermalState.TELEMETRY_DEGRADED:
            if workload is WorkloadClass.INTERACTIVE and not request.automatic:
                return (
                    AdmissionAction.ALLOW_DEGRADED,
                    "telemetry_degraded_interactive_only",
                    "Fresh thermal telemetry is unavailable; only reduced interactive work is permitted.",
                )
            return (
                AdmissionAction.REJECT_RETRYABLE,
                "thermal_telemetry_degraded",
                "Fresh thermal telemetry is required for this workload.",
            )
        if state is ThermalState.VERY_HOT:
            if workload is WorkloadClass.INTERACTIVE and not request.automatic:
                return (
                    AdmissionAction.REJECT_RETRYABLE,
                    "cooldown_required",
                    "ModelDeck is cooling down and cannot admit new heavy work.",
                )
            return AdmissionAction.PAUSE, "very_hot_work_paused", "The workload is paused for cooldown."
        if state is ThermalState.HOT:
            if workload is WorkloadClass.INTERACTIVE:
                return (
                    AdmissionAction.ALLOW_DEGRADED,
                    "hot_interactive_degraded",
                    "Interactive work is permitted at reduced thermal capacity.",
                )
            return (
                AdmissionAction.QUEUE,
                "hot_heavy_concurrency_limited",
                "Heavy work is queued during hot state.",
            )
        if state is ThermalState.WARM and workload is WorkloadClass.INTERACTIVE:
            return (
                AdmissionAction.ALLOW_DEGRADED,
                "warm_interactive_degraded",
                "Interactive work is permitted while background activity remains paused.",
            )
        return AdmissionAction.ALLOW, "thermal_capacity_available", "Thermal capacity is available."

    def _degradation(
        self, request: WorkloadRequest, state: ThermalState, action: AdmissionAction
    ) -> Mapping[str, Any]:
        if action is not AdmissionAction.ALLOW_DEGRADED:
            return {}
        interval = (
            self.config.hot_scene_interval_seconds
            if state in {ThermalState.HOT, ThermalState.TELEMETRY_DEGRADED}
            else self.config.warm_scene_interval_seconds
        )
        return {
            "active": True,
            "minimum_frame_interval_seconds": interval,
            "prevent_overlapping_analysis": True,
            "automatic_capture_allowed": not (
                request.automatic
                and state is ThermalState.VERY_HOT
                and self.config.stop_automatic_scene_when_very_hot
            ),
        }


class HostPowerPolicyStatusReader:
    """Read fixed, bounded host-policy diagnostics without providing mutation capability."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name

    def read(self) -> dict[str, Any]:
        return {
            "available": shutil.which("systemctl") is not None or shutil.which("tuned-adm") is not None,
            "service_active": self._service_active(),
            "tuned_profile": self._tuned_profile(),
            "control": "external_read_only",
        }

    async def read_async(self) -> dict[str, Any]:
        service_active, tuned_profile = await asyncio.gather(
            self._service_active_async(), self._tuned_profile_async()
        )
        return {
            "available": shutil.which("systemctl") is not None or shutil.which("tuned-adm") is not None,
            "service_active": service_active,
            "tuned_profile": tuned_profile,
            "control": "external_read_only",
        }

    async def _service_active_async(self) -> bool | None:
        result = await self._run_async("systemctl", "is-active", self.service_name)
        if result is None:
            return None
        return result[0] == 0 and result[1] == "active"

    async def _tuned_profile_async(self) -> str | None:
        result = await self._run_async("tuned-adm", "active")
        if result is None or result[0] != 0:
            return None
        return result[1].split(":", 1)[-1].strip() or None

    @staticmethod
    async def _run_async(command: str, *arguments: str) -> tuple[int, str] | None:
        executable = shutil.which(command)
        if executable is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2)
        except (OSError, TimeoutError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        return process.returncode or 0, stdout.decode(errors="replace").strip()

    def _service_active(self) -> bool | None:
        executable = shutil.which("systemctl")
        if executable is None:
            return None
        try:
            result = subprocess.run(
                [executable, "is-active", self.service_name],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.returncode == 0 and result.stdout.strip() == "active"

    @staticmethod
    def _tuned_profile() -> str | None:
        executable = shutil.which("tuned-adm")
        if executable is None:
            return None
        try:
            result = subprocess.run(
                [executable, "active"], capture_output=True, text=True, timeout=2, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        return output.split(":", 1)[-1].strip() if output else None


class ThermalPolicyManager:
    def __init__(
        self,
        config: ThermalPolicyConfig,
        *,
        data_dir: Path,
        telemetry_reader: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        host_status_reader: HostPowerPolicyStatusReader | None = None,
        critical_handler: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.machine = ThermalStateMachine(config, monotonic=monotonic)
        self.admission = ThermalAdmissionController(config)
        self._data_dir = data_dir
        self._status_path = data_dir / THERMAL_STATUS_FILENAME
        self._workload_path = data_dir / THERMAL_WORKLOAD_FILENAME
        self._telemetry_reader = telemetry_reader or _normalised_temperatures
        self._monotonic = monotonic
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._host_reader = host_status_reader or HostPowerPolicyStatusReader(config.host_policy_service_name)
        self.critical_handler = critical_handler
        self._host_status: dict[str, Any] = {
            "available": False,
            "service_active": None,
            "tuned_profile": None,
            "control": "external_read_only",
        }
        self._transition_cursor = 0
        self._last_poll_monotonic: float | None = None
        self._state_seconds = {state.value: 0.0 for state in ThermalState}
        self._peak_temperature_c: float | None = None
        self._threshold_seconds = {"75": 0.0, "80": 0.0, "83": 0.0, "85": 0.0}

    async def start(self) -> None:
        if self._task is not None:
            return
        self._data_dir.mkdir(parents=True, exist_ok=True)
        await self.poll_once()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="modeldeck-thermal-policy")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.poll_interval_seconds)
            except TimeoutError:
                await self.poll_once()

    async def poll_once(self) -> dict[str, Any]:
        now = self._monotonic()
        elapsed = 0.0 if self._last_poll_monotonic is None else max(0.0, now - self._last_poll_monotonic)
        self._state_seconds[self.machine.state.value] += elapsed
        previous_temperature = self.machine.reading.celsius if self.machine.reading else None
        if previous_temperature is not None:
            for threshold in (75, 80, 83, 85):
                if previous_temperature >= threshold:
                    self._threshold_seconds[str(threshold)] += elapsed
        self._last_poll_monotonic = now
        reading: ThermalReading | None = None
        if self.config.enabled:
            try:
                readings = self._telemetry_reader()
                reading = select_control_reading(readings, self.machine._selected_sensor_id, now)
            except Exception:
                _LOGGER.exception("ModelDeck thermal telemetry poll failed")
        transition_count = len(self.machine.transitions)
        self.machine.update(reading, now=now)
        if (
            self.critical_handler is not None
            and len(self.machine.transitions) > transition_count
            and self.machine.state is ThermalState.CRITICAL
        ):
            try:
                await self.critical_handler()
            except Exception:
                _LOGGER.exception("Critical thermal worker shutdown failed")
        if reading is not None and (
            self._peak_temperature_c is None or reading.celsius > self._peak_temperature_c
        ):
            self._peak_temperature_c = reading.celsius
        if self.config.enabled and self.config.host_policy_status_enabled:
            self._host_status = await self._host_reader.read_async()
        status = self.status(now=now)
        write_thermal_status(self._status_path, status)
        self._log_new_transitions(status)
        return status

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        now = self._monotonic() if now is None else now
        self.machine.refresh(now=now)
        capacity = capacity_for_state(self.config, self.machine.state)
        reading = self.machine.reading
        active_heavy = read_thermal_workload_activity(
            self._workload_path, self.config, monotonic=self._monotonic
        )
        scene_degraded = self.machine.state in {
            ThermalState.WARM,
            ThermalState.HOT,
            ThermalState.VERY_HOT,
            ThermalState.TELEMETRY_DEGRADED,
        }
        scene_interval = (
            self.config.hot_scene_interval_seconds
            if self.machine.state
            in {
                ThermalState.HOT,
                ThermalState.VERY_HOT,
                ThermalState.TELEMETRY_DEGRADED,
            }
            else self.config.warm_scene_interval_seconds
            if self.machine.state is ThermalState.WARM
            else 0.0
        )
        return {
            "enabled": self.config.enabled,
            "state": self.machine.state.value,
            "temperature_c": reading.celsius if reading else None,
            "sensor_id": reading.sensor_id if reading else self.machine._selected_sensor_id,
            "telemetry_age_seconds": self.machine.telemetry_age(now=now),
            "heavy_concurrency_limit": capacity.heavy_inference_concurrency,
            "active_heavy_concurrency": active_heavy,
            "model_load_concurrency_limit": capacity.model_load_concurrency,
            "background_concurrency_limit": capacity.background_concurrency,
            "background_paused": capacity.background_concurrency == 0,
            "model_loading_allowed": capacity.model_load_concurrency > 0,
            "scenechat_degradation": {
                "active": scene_degraded,
                "minimum_frame_interval_seconds": scene_interval,
                "automatic_capture_allowed": not (
                    self.machine.state in {ThermalState.VERY_HOT, ThermalState.CRITICAL}
                    and self.config.stop_automatic_scene_when_very_hot
                ),
            },
            "reason_code": self.machine.reason_code,
            "timestamp": datetime.now(UTC).isoformat(),
            "published_monotonic": now,
            "host_power_policy": self._host_status,
            "metrics": {
                "state_transition_count": len(self.machine.transitions),
                "state_seconds": dict(self._state_seconds),
                "peak_temperature_c": self._peak_temperature_c,
                "time_at_or_above_75_seconds": self._threshold_seconds["75"],
                "time_at_or_above_80_seconds": self._threshold_seconds["80"],
                "time_at_or_above_83_seconds": self._threshold_seconds["83"],
                "time_at_or_above_85_seconds": self._threshold_seconds["85"],
            },
        }

    def active_heavy_workloads(self) -> int | None:
        return read_thermal_workload_activity(self._workload_path, self.config, monotonic=self._monotonic)

    def _log_new_transitions(self, status: Mapping[str, Any]) -> None:
        for transition in self.machine.transitions[self._transition_cursor :]:
            _LOGGER.warning(
                "Thermal mitigation changed: %s -> %s (%s); capacity=%s",
                transition.previous_state.value,
                transition.new_state.value,
                transition.reason_code,
                {
                    "heavy": status["heavy_concurrency_limit"],
                    "model_load": status["model_load_concurrency_limit"],
                    "background": status["background_concurrency_limit"],
                },
            )
        self._transition_cursor = len(self.machine.transitions)


def select_control_reading(
    readings: Sequence[Mapping[str, Any]], sensor_id: str | None, observed_monotonic: float
) -> ThermalReading | None:
    candidates: list[tuple[int, str, float]] = []
    for item in readings:
        source = str(item.get("source") or "").strip()
        label = str(item.get("label") or source).strip()
        identifier = str(item.get("sensor_id") or f"{source}:{label}")
        try:
            value = float(item["celsius"])
        except (KeyError, TypeError, ValueError):
            continue
        if sensor_id is not None and identifier == sensor_id:
            return ThermalReading(identifier, value, observed_monotonic)
        priority = 0
        if source.casefold() == "k10temp" and label.casefold() in {"tctl", "tdie"}:
            priority = 3
        elif source.casefold() == "amdgpu" and label.casefold() == "edge":
            priority = 2
        elif "amd" in source.casefold():
            priority = 1
        if sensor_id is None and priority:
            candidates.append((priority, identifier, value))
    if sensor_id is not None or not candidates:
        return None
    _, identifier, value = max(candidates, key=lambda candidate: (candidate[0], candidate[2]))
    return ThermalReading(identifier, value, observed_monotonic)


def write_thermal_status(path: Path, status: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def write_thermal_workload_activity(path: Path, active_heavy: int) -> None:
    write_thermal_status(
        path,
        {
            "active_heavy_concurrency": max(0, active_heavy),
            "published_monotonic": time.monotonic(),
        },
    )


def read_thermal_workload_activity(
    path: Path,
    config: ThermalPolicyConfig,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> int | None:
    if not config.enabled:
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        published = float(payload["published_monotonic"])
        active = int(payload["active_heavy_concurrency"])
        age = monotonic() - published
        if active < 0 or age < 0 or age > config.telemetry_stale_seconds:
            return None
        return active
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def read_thermal_status(
    path: Path,
    config: ThermalPolicyConfig,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if not config.enabled:
        capacity = capacity_for_state(config, ThermalState.NORMAL)
        return {
            "enabled": False,
            "state": ThermalState.NORMAL.value,
            "temperature_c": None,
            "sensor_id": config.sensor_id,
            "telemetry_age_seconds": None,
            "heavy_concurrency_limit": capacity.heavy_inference_concurrency,
            "active_heavy_concurrency": 0,
            "model_load_concurrency_limit": capacity.model_load_concurrency,
            "background_concurrency_limit": capacity.background_concurrency,
            "background_paused": False,
            "model_loading_allowed": True,
            "scenechat_degradation": {"active": False, "minimum_frame_interval_seconds": 0.0},
            "reason_code": "thermal_throttling_disabled",
            "host_power_policy": {"available": False, "control": "external_read_only"},
        }
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        published = float(status["published_monotonic"])
        age = monotonic() - published
        ThermalState(status["state"])
        if age < 0 or age > config.telemetry_stale_seconds:
            raise ValueError("stale status")
        status["telemetry_age_seconds"] = max(float(status.get("telemetry_age_seconds") or 0.0) + age, 0.0)
        return status
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        capacity = capacity_for_state(config, ThermalState.TELEMETRY_DEGRADED)
        return {
            "enabled": True,
            "state": ThermalState.TELEMETRY_DEGRADED.value,
            "temperature_c": None,
            "sensor_id": config.sensor_id,
            "telemetry_age_seconds": None,
            "heavy_concurrency_limit": capacity.heavy_inference_concurrency,
            "active_heavy_concurrency": None,
            "model_load_concurrency_limit": capacity.model_load_concurrency,
            "background_concurrency_limit": capacity.background_concurrency,
            "background_paused": True,
            "model_loading_allowed": False,
            "scenechat_degradation": {
                "active": True,
                "minimum_frame_interval_seconds": config.hot_scene_interval_seconds,
                "automatic_capture_allowed": False,
            },
            "reason_code": "thermal_status_unavailable_or_stale",
            "host_power_policy": {"available": False, "control": "external_read_only"},
        }


def _normalised_temperatures() -> Sequence[Mapping[str, Any]]:
    from modeldeck.hardware.probe import read_temperature_telemetry

    return read_temperature_telemetry()


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
