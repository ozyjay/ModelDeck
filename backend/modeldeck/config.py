from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from modeldeck.thermal import ThermalPolicyConfig


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    management_port: int = 3600
    gateway_port: int = 8600
    data_dir: Path = Path(".modeldeck")
    log_dir: Path = Path("var/log/workers")
    open_day: bool = False
    allow_downloads: bool = False
    diagnostic_capture: bool = False
    diffusion_timeout_seconds: float = 900.0
    scenechat_timeout_seconds: float = 75.0
    translation_timeout_seconds: float = 65.0
    speech_synthesis_timeout_seconds: float = 130.0
    speech_recognition_timeout_seconds: float = 35.0
    # Directly constructed Settings retain legacy behaviour for embedded/test apps.
    # Production Settings.from_env() enables the conservative thermal policy by default.
    thermal_throttling: ThermalPolicyConfig = field(
        default_factory=lambda: ThermalPolicyConfig(enabled=False)
    )

    @classmethod
    def from_env(cls) -> Settings:
        open_day = _bool_env("MODELDECK_OPEN_DAY")
        allow_downloads = _bool_env("MODELDECK_ALLOW_DOWNLOADS") and not open_day
        thermal_defaults = ThermalPolicyConfig()
        sensor_id = os.getenv("MODELDECK_THERMAL_SENSOR_ID") or None
        thermal_throttling = ThermalPolicyConfig(
            enabled=_bool_env("MODELDECK_THERMAL_THROTTLING_ENABLED", True),
            sensor_id=sensor_id,
            warm_threshold_c=_float_env(
                "MODELDECK_THERMAL_WARM_THRESHOLD_C", thermal_defaults.warm_threshold_c
            ),
            hot_threshold_c=_float_env("MODELDECK_THERMAL_HOT_THRESHOLD_C", thermal_defaults.hot_threshold_c),
            very_hot_threshold_c=_float_env(
                "MODELDECK_THERMAL_VERY_HOT_THRESHOLD_C", thermal_defaults.very_hot_threshold_c
            ),
            critical_threshold_c=_float_env(
                "MODELDECK_THERMAL_CRITICAL_THRESHOLD_C", thermal_defaults.critical_threshold_c
            ),
            warm_recovery_c=_float_env("MODELDECK_THERMAL_WARM_RECOVERY_C", thermal_defaults.warm_recovery_c),
            hot_recovery_c=_float_env("MODELDECK_THERMAL_HOT_RECOVERY_C", thermal_defaults.hot_recovery_c),
            very_hot_recovery_c=_float_env(
                "MODELDECK_THERMAL_VERY_HOT_RECOVERY_C", thermal_defaults.very_hot_recovery_c
            ),
            telemetry_stale_seconds=_float_env(
                "MODELDECK_THERMAL_TELEMETRY_STALE_SECONDS",
                thermal_defaults.telemetry_stale_seconds,
            ),
            poll_interval_seconds=_float_env(
                "MODELDECK_THERMAL_POLL_INTERVAL_SECONDS", thermal_defaults.poll_interval_seconds
            ),
            minimum_state_dwell_seconds=_float_env(
                "MODELDECK_THERMAL_MINIMUM_STATE_DWELL_SECONDS",
                thermal_defaults.minimum_state_dwell_seconds,
            ),
            recovery_step_seconds=_float_env(
                "MODELDECK_THERMAL_RECOVERY_STEP_SECONDS", thermal_defaults.recovery_step_seconds
            ),
            recovery_reading_count=_int_env(
                "MODELDECK_THERMAL_RECOVERY_READING_COUNT", thermal_defaults.recovery_reading_count
            ),
            configured_heavy_concurrency=_int_env(
                "MODELDECK_THERMAL_HEAVY_CONCURRENCY",
                thermal_defaults.configured_heavy_concurrency,
            ),
            configured_background_concurrency=_int_env(
                "MODELDECK_THERMAL_BACKGROUND_CONCURRENCY",
                thermal_defaults.configured_background_concurrency,
            ),
            warm_scene_interval_seconds=_float_env(
                "MODELDECK_THERMAL_WARM_SCENE_INTERVAL_SECONDS",
                thermal_defaults.warm_scene_interval_seconds,
            ),
            hot_scene_interval_seconds=_float_env(
                "MODELDECK_THERMAL_HOT_SCENE_INTERVAL_SECONDS",
                thermal_defaults.hot_scene_interval_seconds,
            ),
            stop_automatic_scene_when_very_hot=_bool_env(
                "MODELDECK_THERMAL_STOP_AUTOMATIC_SCENE_WHEN_VERY_HOT", True
            ),
            host_policy_status_enabled=_bool_env("MODELDECK_HOST_POLICY_STATUS_ENABLED", True),
            host_policy_service_name=os.getenv(
                "MODELDECK_HOST_POLICY_SERVICE_NAME", thermal_defaults.host_policy_service_name
            ),
        )
        return cls(
            host=os.getenv("MODELDECK_HOST", "127.0.0.1"),
            management_port=int(os.getenv("MODELDECK_MANAGEMENT_PORT", "3600")),
            gateway_port=int(os.getenv("MODELDECK_GATEWAY_PORT", "8600")),
            data_dir=Path(os.getenv("MODELDECK_DATA_DIR", ".modeldeck")),
            log_dir=Path(os.getenv("MODELDECK_LOG_DIR", "var/log/workers")),
            open_day=open_day,
            allow_downloads=allow_downloads,
            diagnostic_capture=_bool_env("MODELDECK_DIAGNOSTIC_CAPTURE"),
            diffusion_timeout_seconds=float(os.getenv("MODELDECK_DIFFUSION_TIMEOUT_SECONDS", "900")),
            scenechat_timeout_seconds=float(os.getenv("MODELDECK_SCENECHAT_TIMEOUT_SECONDS", "75")),
            translation_timeout_seconds=float(os.getenv("MODELDECK_TRANSLATION_TIMEOUT_SECONDS", "65")),
            speech_synthesis_timeout_seconds=float(
                os.getenv("MODELDECK_SPEECH_SYNTHESIS_TIMEOUT_SECONDS", "130")
            ),
            speech_recognition_timeout_seconds=float(
                os.getenv("MODELDECK_SPEECH_RECOGNITION_TIMEOUT_SECONDS", "35")
            ),
            thermal_throttling=thermal_throttling,
        )
