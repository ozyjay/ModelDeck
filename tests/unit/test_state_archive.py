from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.state_archive import STATE_ARCHIVE_ROOT
from modeldeck.state_export import StateExportError, export_state_directory
from modeldeck.state_import import StateImportError, import_state_archive, validate_state_directory


def _state(directory: Path, *, thermal_state: str = "normal") -> Path:
    directory.mkdir(parents=True)
    CompatibilityStore(directory / "modeldeck.sqlite3").initialise_v4()
    (directory / "thermal-status.json").write_text(f'{{"state":"{thermal_state}"}}', encoding="utf-8")
    trusted = directory / "trusted-runtime-manifests"
    trusted.mkdir()
    (trusted / "manifest.json").write_text("{}", encoding="utf-8")
    return directory


def _archive_member(name: str, content: bytes = b"") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    return member


def test_exported_archive_round_trips_complete_state(tmp_path: Path) -> None:
    source = _state(tmp_path / "development/modeldeck", thermal_state="warm")
    archive = tmp_path / "exports/modeldeck-state.tar"

    result = export_state_directory(source, archive)
    imported = import_state_archive(archive, tmp_path / "standalone/modeldeck")

    assert result.destination == archive.resolve()
    assert imported.source == archive.resolve()
    assert (imported.destination / "thermal-status.json").read_text(encoding="utf-8") == '{"state":"warm"}'
    assert (imported.destination / "trusted-runtime-manifests/manifest.json").read_text(
        encoding="utf-8"
    ) == "{}"
    validate_state_directory(imported.destination)


def test_export_refuses_to_replace_an_existing_archive(tmp_path: Path) -> None:
    source = _state(tmp_path / "development/modeldeck")
    archive = tmp_path / "exports/modeldeck-state.tar"
    archive.parent.mkdir()
    archive.write_bytes(b"existing export")

    with pytest.raises(StateExportError, match="already exists"):
        export_state_directory(source, archive)

    assert archive.read_bytes() == b"existing export"


def test_export_rejects_an_invalid_source_directory(tmp_path: Path) -> None:
    source = tmp_path / "invalid"
    source.mkdir()

    with pytest.raises(StateExportError, match="modeldeck.sqlite3"):
        export_state_directory(source, tmp_path / "exports/modeldeck-state.tar")


def test_import_requires_confirmation_and_backs_up_existing_state(tmp_path: Path) -> None:
    source = _state(tmp_path / "development/modeldeck", thermal_state="normal")
    archive = tmp_path / "exports/modeldeck-state.tar"
    export_state_directory(source, archive)
    destination = _state(tmp_path / "standalone/modeldeck", thermal_state="critical")
    (destination / "existing.txt").write_text("preserve this", encoding="utf-8")

    with pytest.raises(StateImportError, match="confirm replacement"):
        import_state_archive(archive, destination)

    result = import_state_archive(archive, destination, replace_existing=True)

    assert result.backup is not None
    assert (result.backup / "existing.txt").read_text(encoding="utf-8") == "preserve this"
    assert (destination / "thermal-status.json").read_text(encoding="utf-8") == '{"state":"normal"}'


@pytest.mark.parametrize(
    ("member_name", "member_type", "message"),
    [
        ("../outside", tarfile.REGTYPE, "unsafe path"),
        (f"{STATE_ARCHIVE_ROOT}/linked", tarfile.SYMTYPE, "unsupported file types"),
    ],
)
def test_import_rejects_unsafe_archive_entries(
    tmp_path: Path,
    member_name: str,
    member_type: bytes,
    message: str,
) -> None:
    archive = tmp_path / "unsafe.tar"
    member = _archive_member(member_name)
    member.type = member_type
    with tarfile.open(archive, mode="w") as output:
        output.addfile(member, io.BytesIO())

    with pytest.raises(StateImportError, match=message):
        import_state_archive(archive, tmp_path / "standalone/modeldeck")


def test_import_rejects_duplicate_archive_paths(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.tar"
    member_name = f"{STATE_ARCHIVE_ROOT}/duplicate"
    with tarfile.open(archive, mode="w") as output:
        for content in (b"one", b"two"):
            output.addfile(_archive_member(member_name, content), io.BytesIO(content))

    with pytest.raises(StateImportError, match="duplicate paths"):
        import_state_archive(archive, tmp_path / "standalone/modeldeck")
