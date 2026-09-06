from __future__ import annotations

import httpx
import pytest
from modeldeck.config import Settings, gateway_base_url, state_store_metadata
from modeldeck.gateway import app as gateway_app
from modeldeck.main import create_app


def test_gateway_host_defaults_to_loopback(monkeypatch) -> None:
    monkeypatch.delenv("MODELDECK_GATEWAY_HOST", raising=False)

    assert Settings.from_env().gateway_host == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "not-an-address"])
def test_management_host_rejects_unsafe_or_invalid_bind_addresses(monkeypatch, host: str) -> None:
    monkeypatch.setenv("MODELDECK_HOST", host)

    with pytest.raises(ValueError, match="MODELDECK_HOST"):
        Settings.from_env()


def test_application_construction_does_not_create_operational_files(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs")

    create_app(settings)
    gateway_app.create_gateway_app(settings=settings)

    assert not settings.data_dir.exists()
    assert not settings.log_dir.exists()


@pytest.mark.asyncio
async def test_management_rejects_mutations_from_a_non_loopback_browser_origin(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post("/api/workers/stop-all", headers={"Origin": "https://example.invalid"})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_native_management_client_without_origin_remains_supported(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post("/api/workers/stop-all")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_rejects_mutations_from_a_non_loopback_browser_origin(tmp_path) -> None:
    app = gateway_app.create_gateway_app(settings=Settings(data_dir=tmp_path / "data"))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/completions",
            headers={"Origin": "https://example.invalid"},
            json={"model": "not-used", "prompt": "not-used"},
        )

    assert response.status_code == 403


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


def test_docker_bridge_does_not_allow_the_authoritative_gateway_to_bind_the_bridge(monkeypatch) -> None:
    monkeypatch.setenv("MODELDECK_GATEWAY_HOST", "172.17.0.1")
    monkeypatch.setenv("MODELDECK_ENABLE_DOCKER_BRIDGE", "1")

    with pytest.raises(ValueError, match="must be a loopback address"):
        Settings.from_env()


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
