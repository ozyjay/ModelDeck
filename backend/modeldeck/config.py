from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    management_port: int = 3600
    gateway_port: int = 8600
    data_dir: Path = Path(".modeldeck")
    open_day: bool = False
    allow_downloads: bool = False
    diagnostic_capture: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        open_day = _bool_env("MODELDECK_OPEN_DAY")
        allow_downloads = _bool_env("MODELDECK_ALLOW_DOWNLOADS") and not open_day
        return cls(
            host=os.getenv("MODELDECK_HOST", "127.0.0.1"),
            management_port=int(os.getenv("MODELDECK_MANAGEMENT_PORT", "3600")),
            gateway_port=int(os.getenv("MODELDECK_GATEWAY_PORT", "8600")),
            data_dir=Path(os.getenv("MODELDECK_DATA_DIR", ".modeldeck")),
            open_day=open_day,
            allow_downloads=allow_downloads,
            diagnostic_capture=_bool_env("MODELDECK_DIAGNOSTIC_CAPTURE"),
        )
