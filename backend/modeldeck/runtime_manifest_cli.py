from __future__ import annotations

import argparse
import os
from pathlib import Path

from modeldeck.registry import install_runtime_manifest, runtime_template_registrations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install an operator-approved ModelDeck runtime template manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sha256", required=True, dest="expected_sha256")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MODELDECK_DATA_DIR", ".modeldeck")),
    )
    arguments = parser.parse_args()
    target = install_runtime_manifest(
        arguments.manifest,
        arguments.data_dir,
        arguments.expected_sha256,
    )
    registrations = runtime_template_registrations(arguments.data_dir)
    installed = [
        registration
        for registration in registrations.values()
        if registration.source == "trusted-local" and target.stem.startswith(registration.package.id)
    ]
    print(f"Installed trusted runtime manifest: {target}")
    for registration in installed:
        print(
            f"  {registration.template.id}: {registration.template.display_name} "
            f"-> {registration.template.runtime}"
        )


if __name__ == "__main__":
    main()
