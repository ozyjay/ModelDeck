from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import tomllib
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


def test_desktop_development_build_id_overrides_packaged_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODELDECK_DESKTOP_BUILD_ID", "development")

    assert read_installed_build_id(tmp_path / "missing-release.json") == "development"


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
    assert "AutoReqProv:    no" in spec
    assert "%global debug_package %{nil}" in spec
    assert "%global __strip /bin/true" in spec
    assert "%global __brp_mangle_shebangs %{nil}" in spec
    assert "%global __os_install_post_build_reproducibility %{nil}" in spec
    assert "%global __brp_check_rpaths %{nil}" in spec
    assert "--no-index" in build_script
    assert "--progress-bar off" in build_script
    assert "--no-cache-dir" in build_script
    assert '"_tmppath $RpmTop/TMP"' in build_script
    assert "rpmbuild --quiet" in build_script
    assert "patchelf --remove-rpath" in build_script
    assert "patchelf is required" in build_script
    assert "Assert-OfflineWheelhouse" in build_script
    assert "Duplicate wheelhouse SHA-256 entry" in build_script
    assert "Wheelhouse contains unlisted files" in build_script
    assert "New-BundledPythonRuntime" in build_script
    assert "Set-PackagedRuntimeLauncher" in build_script
    assert "desktop-python" in build_script
    assert "absolute symbolic links" in build_script
    assert "direct_url.json" in build_script
    assert "__pycache__" in build_script
    assert "backend/modeldeck/__init__.py" in build_script
    assert "modeldeck_version $Version" in build_script
    assert "modeldeck_release $RpmRelease" in build_script
    assert "RpmRelease must be a positive integer" in build_script
    cleanup_position = build_script.index("Get-ChildItem $Libexec -Recurse -Directory -Filter '__pycache__'")
    desktop_copy_position = build_script.index("Copy-Item 'backend/modeldeck/desktop'")
    assert cleanup_position > desktop_copy_position


def test_fedora_standalone_build_wrapper_uses_the_offline_rpm_builder() -> None:
    wrapper = (PROJECT_ROOT / "scripts/packaging/build_fedora_standalone.ps1").read_text(encoding="utf-8")

    assert "build_fedora_rpm.ps1" in wrapper
    assert "Assert-Python312" in wrapper
    assert "Resolve-Python312" in wrapper
    assert "Prepare-OfflineWheelhouse" in wrapper
    assert "-m pip wheel" in wrapper
    assert "--wheel-dir" in wrapper
    assert "--no-cache-dir" in wrapper
    assert "modeldeck-wheelhouse-" in wrapper
    assert "-ReplaceWheelhouse" in wrapper
    assert "$BuildParameters" in wrapper
    assert "RpmRelease = $RpmRelease" in wrapper
    assert "modeldeck-*.x86_64.rpm" in wrapper


def test_development_desktop_launcher_uses_checkout_services() -> None:
    launcher = (PROJECT_ROOT / "scripts/operations/run_desktop.ps1").read_text(encoding="utf-8")

    assert "run.ps1" in launcher
    assert "MODELDECK_DESKTOP_DEVELOPMENT" in launcher
    assert "MODELDECK_DESKTOP_BUILD_ID" in launcher
    assert "/usr/bin/python3 -m modeldeck.desktop.app" in launcher
    assert "Invoke-RestMethod -Uri 'http://127.0.0.1:3600/api/health'" in launcher
    assert "Reusing the healthy local ModelDeck services" in launcher


def test_fedora_packaging_uses_the_hatch_version_source() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    spec = (PROJECT_ROOT / "packaging/fedora/modeldeck.spec").read_text(encoding="utf-8")

    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["hatch"]["version"]["path"] == "backend/modeldeck/__init__.py"
    assert "Version:        %{modeldeck_version}" in spec
    assert "Release:        %{modeldeck_release}%{?dist}" in spec


def test_fedora_release_signing_wrapper_handles_key_selection_and_creation() -> None:
    wrapper = (PROJECT_ROOT / "scripts/packaging/release_fedora_rpm.ps1").read_text(encoding="utf-8")
    signer = (PROJECT_ROOT / "scripts/packaging/sign_fedora_rpm.ps1").read_text(encoding="utf-8")

    assert "sign_fedora_rpm.ps1" in wrapper
    assert "Get-SecretKeyFingerprints" in wrapper
    assert "--quick-generate-key" in wrapper
    assert "-CreateKey" in wrapper
    assert "Multiple secret GPG keys" in wrapper
    assert "AwaitingPrimaryFingerprint" in wrapper
    assert "-VerifyOnly" in wrapper
    assert "rpm --checksig --verbose" in signer
    assert "\\bNOKEY\\b" in signer
    assert "$global:LASTEXITCODE = 0" in signer


def test_packaged_desktop_module_calls_its_main_entrypoint() -> None:
    source = (PROJECT_ROOT / "backend/modeldeck/desktop/app.py").read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":\n    main()' in source
    assert "set_hardware_acceleration_policy(WebKit.HardwareAccelerationPolicy.NEVER)" in source
    assert "set_enable_write_console_messages_to_stdout(self.development_mode)" in source
    assert 'connect("load-failed", self._on_console_load_failed)' in source
    assert 'connect("load-changed", self._on_console_load_changed)' in source
    assert "rootChildren" in source
    assert "WebKit.UserScriptInjectionTime.START" in source
    assert "ModelDeck page error" in source
    assert "WebKit.WebView(settings=settings, user_content_manager=content_manager)" in source


def test_fedora_offline_builder_rejects_unlisted_wheelhouse_files(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    listed = wheelhouse / "listed.whl"
    listed.write_bytes(b"listed")
    (wheelhouse / "unlisted.whl").write_bytes(b"unlisted")
    manifest = tmp_path / "wheelhouse.sha256"
    manifest.write_text(
        f"{hashlib.sha256(listed.read_bytes()).hexdigest()}  {listed.name}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(PROJECT_ROOT / "scripts/packaging/build_fedora_rpm.ps1"),
            "-Wheelhouse",
            str(wheelhouse),
            "-WheelhouseManifest",
            str(manifest),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Wheelhouse contains unlisted files: unlisted.whl" in result.stderr
