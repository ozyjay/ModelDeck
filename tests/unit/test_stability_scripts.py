from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BURN_IN = ROOT / "scripts" / "burn_in_diffusiongemma_selected_preset.ps1"


def test_diffusiongemma_selected_preset_burn_in_validates_without_starting_hardware(
    tmp_path: Path,
) -> None:
    json_output = tmp_path / "burn-in.json"
    markdown_output = tmp_path / "burn-in.md"

    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(BURN_IN),
            "-JsonOutput",
            str(json_output),
            "-MarkdownOutput",
            str(markdown_output),
            "-ValidateOnly",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Profile: diffusiongemma-q4-rocm" in result.stdout
    assert "Duration: 120 minutes; interval: 5 seconds" in result.stdout
    assert not json_output.exists()
    assert not markdown_output.exists()
