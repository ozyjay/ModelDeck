from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.desktop.app import read_installed_build_id
from modeldeck.desktop.controller import (
    SYSTEMCTL,
    TARGET,
    DesktopServiceError,
    ServiceController,
    ServiceHealth,
    should_prompt_for_restart,
)
from modeldeck.state_import import StateImportError, import_state_directory, validate_state_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _state(directory: Path) -> Path:
    directory.mkdir(parents=True)
    CompatibilityStore(directory / "modeldeck.sqlite3").initialise_v4()
    (directory / "thermal-status.json").write_text('{"state":"normal"}', encoding="utf-8")
    trusted = directory / "trusted-runtime-manifests"
    trusted.mkdir()
    (trusted / "manifest.json").write_text("{}", encoding="utf-8")
    return directory


def test_desktop_settings_use_xdg_paths_only_when_packaged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODELDECK_DESKTOP", "1")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("MODELDECK_DATA_DIR", raising=False)
    monkeypatch.delenv("MODELDECK_LOG_DIR", raising=False)

    settings = Settings.from_env()

    assert settings.data_dir == tmp_path / "data/modeldeck"
    assert settings.log_dir == tmp_path / "state/modeldeck/logs/workers"


def test_service_controller_uses_only_fixed_user_systemd_commands() -> None:
    commands: list[tuple[str, ...]] = []
    controller = ServiceController(run=lambda command: commands.append(tuple(command)))

    controller.start()
    controller.restart()
    controller.stop()

    assert commands == [
        (SYSTEMCTL, "--user", "daemon-reload"),
        (SYSTEMCTL, "--user", "start", TARGET),
        (SYSTEMCTL, "--user", "daemon-reload"),
        (SYSTEMCTL, "--user", "restart", TARGET),
        (SYSTEMCTL, "--user", "stop", TARGET),
    ]


def test_service_controller_waits_for_a_ready_management_service() -> None:
    results = iter(
        (
            DesktopServiceError("not yet"),
            ServiceHealth(status="starting", build_id="old"),
            ServiceHealth(status="ok", build_id="build-1"),
        )
    )

    def health() -> ServiceHealth:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    controller = ServiceController(read_health=health, sleep=lambda _delay: None)

    assert controller.wait_until_ready(timeout_seconds=1).build_id == "build-1"


def test_update_prompt_requires_a_distinct_running_build() -> None:
    assert should_prompt_for_restart(installed_build_id="0.1.0-2", running_build_id="0.1.0-1")
    assert not should_prompt_for_restart(installed_build_id="0.1.0-2", running_build_id="0.1.0-2")
    assert not should_prompt_for_restart(installed_build_id="0.1.0-2", running_build_id=None)


def test_state_import_copies_validated_state_and_leaves_source_untouched(tmp_path: Path) -> None:
    source = _state(tmp_path / "source")
    destination = tmp_path / "xdg/modeldeck"

    result = import_state_directory(source, destination)

    assert result.source == source.resolve()
    assert result.destination == destination.resolve()
    assert result.backup is None
    assert (destination / "thermal-status.json").read_text(encoding="utf-8") == '{"state":"normal"}'
    assert (source / "trusted-runtime-manifests/manifest.json").exists()
    with sqlite3.connect(destination / "modeldeck.sqlite3") as database:
        assert database.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_state_import_requires_confirmation_before_replacing_existing_state(tmp_path: Path) -> None:
    source = _state(tmp_path / "source")
    destination = _state(tmp_path / "xdg/modeldeck")
    (destination / "existing.txt").write_text("keep as backup", encoding="utf-8")

    with pytest.raises(StateImportError, match="confirm replacement"):
        import_state_directory(source, destination)

    result = import_state_directory(source, destination, replace_existing=True)

    assert result.backup is not None
    assert (result.backup / "existing.txt").read_text(encoding="utf-8") == "keep as backup"
    assert not (destination / "existing.txt").exists()


def test_state_import_rejects_legacy_or_invalid_data(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with sqlite3.connect(legacy / "modeldeck.sqlite3") as database:
        database.execute("CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT)")
        database.execute("INSERT INTO schema_metadata VALUES ('schema_version', '3')")

    with pytest.raises(StateImportError, match="schema version 4"):
        validate_state_directory(legacy)


def test_release_metadata_requires_a_build_identifier(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    release.write_text('{"build_id":"0.1.0-1"}', encoding="utf-8")

    assert read_installed_build_id(release) == "0.1.0-1"

    release.write_text("{}", encoding="utf-8")
    with pytest.raises(DesktopServiceError, match="build ID"):
        read_installed_build_id(release)


def test_fedora_assets_keep_services_loopback_only_and_package_models_externally() -> None:
    management_path = PROJECT_ROOT / "packaging/fedora/modeldeck-management.service.in"
    gateway_path = PROJECT_ROOT / "packaging/fedora/modeldeck-gateway.service.in"
    management = management_path.read_text(encoding="utf-8")
    gateway = gateway_path.read_text(encoding="utf-8")
    spec = (PROJECT_ROOT / "packaging/fedora/modeldeck.spec").read_text(encoding="utf-8")
    build_script = (PROJECT_ROOT / "scripts/packaging/build_fedora_rpm.ps1").read_text(encoding="utf-8")

    for service in (management, gateway):
        assert "MODELDECK_HOST=127.0.0.1" in service
        assert "MODELDECK_GATEWAY_HOST=127.0.0.1" in service
        assert "MODELDECK_DESKTOP=1" in service
        assert "XDG_DATA_HOME=%h/.local/share" in service
        assert "XDG_STATE_HOME=%h/.local/state" in service
        assert "MODELDECK_ROCM72_PYTHON=/usr/libexec/modeldeck/rocm72/bin/python" in service
    assert "never includes model weights" in spec
    assert "--no-index" in build_script
    assert "Assert-OfflineWheelhouse" in build_script


def test_fedora_standalone_build_wrapper_uses_the_offline_rpm_builder() -> None:
    wrapper = (PROJECT_ROOT / "scripts/packaging/build_fedora_standalone.ps1").read_text(
        encoding="utf-8"
    )

    assert "build_fedora_rpm.ps1" in wrapper
    assert "@PSBoundParameters" in wrapper
    assert "modeldeck-*.x86_64.rpm" in wrapper
