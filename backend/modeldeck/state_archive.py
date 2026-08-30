"""Creation and safe extraction of ModelDeck state archives."""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

STATE_ARCHIVE_ROOT = "modeldeck-state"


class StateArchiveError(RuntimeError):
    """A ModelDeck state archive is unsafe or unreadable."""


def create_state_archive(source: Path, destination: Path) -> None:
    """Write a new tar archive without replacing an existing file."""

    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        if destination.exists():
            raise StateArchiveError("The export destination already exists; choose a new file") from error
        raise StateArchiveError(f"Could not create the state export: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            with tarfile.open(fileobj=output, mode="w") as archive:
                archive.add(source, arcname=STATE_ARCHIVE_ROOT, recursive=True)
    except (OSError, tarfile.TarError) as error:
        destination.unlink(missing_ok=True)
        raise StateArchiveError(f"Could not create the state export: {error}") from error


def extract_state_archive(source: Path, destination: Path) -> None:
    """Safely extract a ModelDeck archive into an empty staging directory."""

    if not source.is_file():
        raise StateArchiveError("Select a ModelDeck state archive file")
    seen_paths: set[PurePosixPath] = set()
    try:
        with tarfile.open(source, mode="r") as archive:
            members = archive.getmembers()
            _validate_members(members, seen_paths)
            for member in members:
                relative_path = PurePosixPath(member.name)
                target = destination.joinpath(*relative_path.parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:  # pragma: no cover - guarded by member type validation
                    raise StateArchiveError(f"Archive entry could not be read: {member.name}")
                with extracted, target.open("xb") as output:
                    shutil.copyfileobj(extracted, output)
    except (OSError, tarfile.TarError) as error:
        raise StateArchiveError(f"The selected state archive is unreadable: {error}") from error


def _validate_members(members: list[tarfile.TarInfo], seen_paths: set[PurePosixPath]) -> None:
    if not members:
        raise StateArchiveError("The selected state archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != STATE_ARCHIVE_ROOT:
            raise StateArchiveError("The selected state archive contains an unsafe path")
        if path in seen_paths:
            raise StateArchiveError("The selected state archive contains duplicate paths")
        seen_paths.add(path)
        if not (member.isdir() or member.isfile()):
            raise StateArchiveError("The selected state archive contains unsupported file types")
