from __future__ import annotations

import pytest
from modeldeck.config import Settings


def test_configuration_lock_uses_the_deployment_neutral_environment_name(monkeypatch) -> None:
    monkeypatch.setenv("MODELDECK_CONFIGURATION_LOCKED", "1")
    monkeypatch.delenv("MODELDECK_OPEN_DAY", raising=False)

    assert Settings.from_env().configuration_locked is True


def test_legacy_open_day_environment_name_is_deprecated(monkeypatch) -> None:
    monkeypatch.delenv("MODELDECK_CONFIGURATION_LOCKED", raising=False)
    monkeypatch.setenv("MODELDECK_OPEN_DAY", "1")

    with pytest.deprecated_call(match="MODELDECK_OPEN_DAY"):
        settings = Settings.from_env()

    assert settings.configuration_locked is True


def test_current_configuration_lock_takes_precedence_over_legacy_environment_name(monkeypatch) -> None:
    monkeypatch.setenv("MODELDECK_CONFIGURATION_LOCKED", "0")
    monkeypatch.setenv("MODELDECK_OPEN_DAY", "1")

    assert Settings.from_env().configuration_locked is False
