"""Safe, explicit import of a ModelDeck per-user data directory."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class StateImportError(RuntimeError):
    """A source directory cannot safely become packaged-app state."""


@dataclass(frozen=True)
class StateImportResult:
    source: Path
    destination: Path
    backup: Path | None


def validate_state_directory(directory: Path) -> None:
    """Check that a directory holds a complete schema-v4 ModelDeck database."""

    database_path = directory / "modeldeck.sqlite3"
    if not directory.is_dir() or not database_path.is_file():
        raise StateImportError("Select a ModelDeck data directory containing modeldeck.sqlite3")
    try:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
            integrity = database.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise StateImportError("The selected ModelDeck database did not pass SQLite integrity_check")
            row = database.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
    except sqlite3.DatabaseError as error:
        raise StateImportError("The selected ModelDeck database is unreadable") from error
    if row is None or str(row[0]) != "4":
        raise StateImportError(
            "The selected data is not schema version 4. Run the documented database migration first."
        )
    _reject_symlinks(directory)


def import_state_directory(
    source: Path,
    destination: Path,
    *,
    replace_existing: bool = False,
) -> StateImportResult:
    """Copy validated state without modifying the source directory.

    Replacing non-empty state is deliberately opt-in.  The previous destination
    is moved alongside it as a timestamped backup only after the copied state
    has passed validation.
    """

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise StateImportError("The source and destination data directories must differ")
    validate_state_directory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.exists() and any(destination.iterdir())
    if existing and not replace_existing:
        raise StateImportError("The packaged-app data directory is not empty; confirm replacement first")

    with tempfile.TemporaryDirectory(prefix="modeldeck-import-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "state"
        shutil.copytree(source, staged, copy_function=shutil.copy2)
        validate_state_directory(staged)
        backup: Path | None = None
        if destination.exists():
            if existing:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                backup = destination.parent / f"{destination.name}.backup-{stamp}"
                if backup.exists():
                    raise StateImportError(f"Backup destination already exists: {backup}")
                destination.replace(backup)
            else:
                destination.rmdir()
        staged.replace(destination)
    return StateImportResult(source=source, destination=destination, backup=backup)


def _reject_symlinks(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise StateImportError(f"The selected data directory contains a symbolic link: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import existing ModelDeck state into the Fedora desktop app"
    )
    parser.add_argument("source", type=Path, help="Existing ModelDeck data directory, usually .modeldeck")
    parser.add_argument("destination", type=Path, help="XDG ModelDeck data directory")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Back up and replace non-empty destination state",
    )
    args = parser.parse_args()
    try:
        result = import_state_directory(args.source, args.destination, replace_existing=args.replace_existing)
    except StateImportError as error:
        parser.error(str(error))
    print(f"Imported ModelDeck state into {result.destination}")
    if result.backup:
        print(f"Previous state backed up to {result.backup}")


if __name__ == "__main__":
    main()
