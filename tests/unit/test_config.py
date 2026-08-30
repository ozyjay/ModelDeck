from __future__ import annotations

import pytest
from modeldeck.config import Settings, gateway_base_url, state_store_metadata
from modeldeck.gateway import app as gateway_app


def test_gateway_host_defaults_to_loopback(monkeypatch) -> None:
    monkeypatch.delenv("MODELDECK_GATEWAY_HOST", raising=False)

    assert Settings.from_env().gateway_host == "127.0.0.1"


def test_state_store_metadata_distinguishes_desktop_and_checkout_state(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MODELDECK_DESKTOP", raising=False)
    assert state_store_metadata(tmp_path)["kind"] == "checkout-development"

    monkeypatch.setenv("MODELDECK_DESKTOP", "1")
    metadata = state_store_metadata(tmp_path)

    assert metadata == {
        "kind": "desktop-standalone",
        "label": "Desktop standalone state",
        "directory": str(tmp_path.resolve()),
    }


def test_gateway_host_does_not_change_legacy_settings_positional_arguments() -> None:
    settings = Settings("127.0.0.1", 13600, 18600)

    assert settings.management_port == 13600
    assert settings.gateway_port == 18600
    assert settings.gateway_host == "127.0.0.1"


def test_docker_bridge_is_an_explicit_secondary_listener_option(monkeypatch) -> None:
    monkeypatch.setenv("MODELDECK_HOST", "127.0.0.1")
    monkeypatch.delenv("MODELDECK_GATEWAY_HOST", raising=False)
    monkeypatch.setenv("MODELDECK_ENABLE_DOCKER_BRIDGE", "1")

    settings = Settings.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.gateway_host == "127.0.0.1"
    assert settings.docker_bridge_enabled is True
    assert gateway_base_url(settings.gateway_host, settings.gateway_port) == "http://127.0.0.1:8600"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "172.17.0.1", "not-an-address"])
def test_gateway_host_rejects_unsafe_or_invalid_bind_addresses(monkeypatch, host: str) -> None:
    monkeypatch.setenv("MODELDECK_GATEWAY_HOST", host)
    monkeypatch.delenv("MODELDECK_ENABLE_DOCKER_BRIDGE", raising=False)

    with pytest.raises(ValueError, match="MODELDECK_GATEWAY_HOST"):
        Settings.from_env()


def test_gateway_process_binds_to_the_configured_loopback_host(monkeypatch, tmp_path) -> None:
    settings = Settings(gateway_host="127.0.0.1", gateway_port=18600, data_dir=tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(gateway_app.Settings, "from_env", classmethod(lambda _cls: settings))
    monkeypatch.setattr(gateway_app, "create_gateway_app", lambda **_kwargs: object())
    monkeypatch.setattr(gateway_app.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))

    gateway_app.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 18600


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
