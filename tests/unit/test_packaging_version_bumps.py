from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_BUMP = PROJECT_ROOT / "scripts/packaging/bump_version.ps1"
RELEASE_BUMP = PROJECT_ROOT / "scripts/packaging/bump_rpm_release.ps1"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pwsh", "-NoProfile", "-File", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_version_bump_updates_the_canonical_version_and_resets_release(tmp_path: Path) -> None:
    version_file = tmp_path / "__init__.py"
    release_file = tmp_path / "rpm-release"
    package_file = tmp_path / "package.json"
    lock_file = tmp_path / "package-lock.json"
    version_file.write_text('__version__ = "0.1.1"\n', encoding="utf-8")
    release_file.write_text("7\n", encoding="utf-8")
    package_file.write_text('{\n  "name": "modeldeck-ui",\n  "version": "0.1.1"\n}\n', encoding="utf-8")
    lock_file.write_text(
        '{\n  "name": "modeldeck-ui",\n  "version": "0.1.1",\n  "packages": {\n'
        '    "": {\n      "version": "0.1.1"\n    }\n  }\n}\n',
        encoding="utf-8",
    )

    result = _run(
        str(VERSION_BUMP),
        "-Part",
        "Patch",
        "-VersionFile",
        str(version_file),
        "-ReleaseFile",
        str(release_file),
        "-FrontendPackageFile",
        str(package_file),
        "-FrontendLockFile",
        str(lock_file),
    )

    assert result.returncode == 0, result.stderr
    assert version_file.read_text(encoding="utf-8") == '__version__ = "0.1.2"\n'
    assert release_file.read_text(encoding="utf-8") == "1\n"
    assert '"version": "0.1.2"' in package_file.read_text(encoding="utf-8")
    assert lock_file.read_text(encoding="utf-8").count('"version": "0.1.2"') == 2


def test_version_bump_whatif_and_release_increment_preserve_or_update_as_requested(tmp_path: Path) -> None:
    version_file = tmp_path / "__init__.py"
    release_file = tmp_path / "rpm-release"
    package_file = tmp_path / "package.json"
    lock_file = tmp_path / "package-lock.json"
    version_file.write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    release_file.write_text("4\n", encoding="utf-8")
    package_file.write_text('{\n  "name": "modeldeck-ui",\n  "version": "1.2.3"\n}\n', encoding="utf-8")
    lock_file.write_text(
        '{\n  "name": "modeldeck-ui",\n  "version": "1.2.3",\n  "packages": {\n'
        '    "": {\n      "version": "1.2.3"\n    }\n  }\n}\n',
        encoding="utf-8",
    )

    preview = _run(
        str(VERSION_BUMP),
        "-Part",
        "Minor",
        "-VersionFile",
        str(version_file),
        "-ReleaseFile",
        str(release_file),
        "-FrontendPackageFile",
        str(package_file),
        "-FrontendLockFile",
        str(lock_file),
        "-WhatIf",
    )
    increment = _run(str(RELEASE_BUMP), "-Increment", "-ReleaseFile", str(release_file))

    assert preview.returncode == 0, preview.stderr
    assert version_file.read_text(encoding="utf-8") == '__version__ = "1.2.3"\n'
    assert increment.returncode == 0, increment.stderr
    assert release_file.read_text(encoding="utf-8") == "5\n"
