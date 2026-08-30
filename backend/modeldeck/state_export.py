"""Safe export of a ModelDeck per-user data directory."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from modeldeck.state_import import StateImportError, validate_state_directory


class StateExportError(RuntimeError):
    """State cannot safely be exported to the requested destination."""


@dataclass(frozen=True)
class StateExportResult:
    source: Path
    destination: Path


def export_state_directory(source: Path, destination: Path) -> StateExportResult:
    """Copy validated state to a new directory without modifying the source.

    The caller must ensure services are stopped before exporting.  A destination
    is never replaced, and the staged copy is validated before it becomes
    visible, so every successful export can be selected by state import.
    """

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if source == destination:
        raise StateExportError("The source and destination data directories must differ")
    if destination.exists():
        raise StateExportError("The export destination already exists; choose a new directory")

    try:
        validate_state_directory(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="modeldeck-export-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "state"
            shutil.copytree(source, staged, copy_function=shutil.copy2)
            validate_state_directory(staged)
            staged.replace(destination)
    except StateImportError as error:
        raise StateExportError(str(error)) from error
    except OSError as error:
        raise StateExportError(f"Could not export ModelDeck state: {error}") from error
    return StateExportResult(source=source, destination=destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ModelDeck state to a new, import-compatible data directory"
    )
    parser.add_argument("source", type=Path, help="Existing XDG ModelDeck data directory")
    parser.add_argument("destination", type=Path, help="New directory for the exported state")
    args = parser.parse_args()
    try:
        result = export_state_directory(args.source, args.destination)
    except StateExportError as error:
        parser.error(str(error))
    print(f"Exported ModelDeck state to {result.destination}")


if __name__ == "__main__":
    main()
