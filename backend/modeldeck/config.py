from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from ipaddress import ip_address
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


def _desktop_path(kind: str) -> Path:
    """Return the XDG location used by the packaged Fedora user services."""

    if kind == "data":
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    elif kind == "state":
        root = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    else:  # pragma: no cover - defensive guard for fixed internal callers
        raise ValueError(f"Unsupported XDG location: {kind}")
    return root / "modeldeck"


def _default_data_dir() -> Path:
    return _desktop_path("data") if _bool_env("MODELDECK_DESKTOP") else Path(".modeldeck")


def _default_log_dir() -> Path:
    if _bool_env("MODELDECK_DESKTOP"):
        return _desktop_path("state") / "logs" / "workers"
    return Path("var/log/workers")


def _gateway_host_from_env(*, docker_bridge_enabled: bool) -> str:
    """Return a gateway bind address permitted by the local-only policy."""

    raw_host = os.getenv("MODELDECK_GATEWAY_HOST", "127.0.0.1").strip()
    if not raw_host:
        raise ValueError("MODELDECK_GATEWAY_HOST must be an IP address literal")
    try:
        host = ip_address(raw_host)
    except ValueError as error:
        raise ValueError("MODELDECK_GATEWAY_HOST must be an IP address literal") from error

    docker_default_bridge = ip_address("172.17.0.1")
    if not (host.is_loopback or (docker_bridge_enabled and host == docker_default_bridge)):
        raise ValueError(
            "MODELDECK_GATEWAY_HOST must be a loopback address; set "
            "MODELDECK_ENABLE_DOCKER_BRIDGE=1 only for the launcher-managed Docker "
            "bridge listener at 172.17.0.1"
        )
    return str(host)


def gateway_base_url(host: str, port: int) -> str:
    """Build an HTTP base URL for a validated gateway bind address."""

    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{port}"


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    management_port: int = 3600
    gateway_port: int = 8600
    gateway_host: str = "127.0.0.1"
    docker_bridge_enabled: bool = False
    data_dir: Path = Path(".modeldeck")
    log_dir: Path = Path("var/log/workers")
    configuration_locked: bool = False
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
        configuration_locked = _configuration_locked_from_env()
        docker_bridge_enabled = _bool_env("MODELDECK_ENABLE_DOCKER_BRIDGE")
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
            gateway_host=_gateway_host_from_env(docker_bridge_enabled=docker_bridge_enabled),
            docker_bridge_enabled=docker_bridge_enabled,
            management_port=int(os.getenv("MODELDECK_MANAGEMENT_PORT", "3600")),
            gateway_port=int(os.getenv("MODELDECK_GATEWAY_PORT", "8600")),
            data_dir=Path(os.getenv("MODELDECK_DATA_DIR", str(_default_data_dir()))),
            log_dir=Path(os.getenv("MODELDECK_LOG_DIR", str(_default_log_dir()))),
            configuration_locked=configuration_locked,
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


def _configuration_locked_from_env() -> bool:
    if os.getenv("MODELDECK_CONFIGURATION_LOCKED") is not None:
        return _bool_env("MODELDECK_CONFIGURATION_LOCKED")
    if os.getenv("MODELDECK_OPEN_DAY") is not None:
        warnings.warn(
            "MODELDECK_OPEN_DAY is deprecated; use MODELDECK_CONFIGURATION_LOCKED instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _bool_env("MODELDECK_OPEN_DAY")
    return False
