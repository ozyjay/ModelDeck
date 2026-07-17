from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS = PROJECT_ROOT / "scripts" / "environment_helpers.psm1"
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run.ps1"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def _run_pwsh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_env_loader_imports_allowlisted_literal_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "# local ModelDeck settings",
                "MODELDECK_HOST=127.0.0.2",
                'MODELDECK_SCENECHAT_API_KEY="secret=#literal value"',
                "MODELDECK_SCENECHAT_TIMEOUT_SECONDS='90'",
            )
        ),
        encoding="utf-8",
    )
    result = _run_pwsh(
        "Remove-Item Env:MODELDECK_HOST,Env:MODELDECK_SCENECHAT_API_KEY,"
        "Env:MODELDECK_SCENECHAT_TIMEOUT_SECONDS -ErrorAction SilentlyContinue; "
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{env_file}'; "
        "[pscustomobject]@{ Host=$Env:MODELDECK_HOST; Key=$Env:MODELDECK_SCENECHAT_API_KEY; "
        "Timeout=$Env:MODELDECK_SCENECHAT_TIMEOUT_SECONDS } | ConvertTo-Json -Compress"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "Host": "127.0.0.2",
        "Key": "secret=#literal value",
        "Timeout": "90",
    }


def test_process_environment_takes_precedence_over_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODELDECK_HOST=127.0.0.2\n", encoding="utf-8")
    result = _run_pwsh(
        "$Env:MODELDECK_HOST='127.0.0.9'; "
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{env_file}'; "
        "$Env:MODELDECK_HOST"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "127.0.0.9"


def test_env_loader_rejects_unknown_names_without_echoing_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("UNSAFE_COMMAND=do-not-print-this\n", encoding="utf-8")
    result = _run_pwsh(f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{env_file}'")

    assert result.returncode != 0
    assert "Unsupported .env variable" in result.stderr
    assert "UNSAFE_COMMAND" in result.stderr
    assert "do-not-print-this" not in result.stderr


def test_env_loader_rejects_duplicates_and_malformed_lines(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.env"
    duplicate.write_text(
        "MODELDECK_HOST=127.0.0.1\nMODELDECK_HOST=127.0.0.2\n",
        encoding="utf-8",
    )
    malformed = tmp_path / "malformed.env"
    malformed.write_text("MODELDECK_HOST\n", encoding="utf-8")

    duplicate_result = _run_pwsh(
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{duplicate}'"
    )
    malformed_result = _run_pwsh(
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{malformed}'"
    )

    assert duplicate_result.returncode != 0
    assert "Duplicate .env variable" in duplicate_result.stderr
    assert malformed_result.returncode != 0
    assert "Expected NAME=VALUE" in malformed_result.stderr


def test_run_script_loads_dotenv_before_open_day_overrides() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "environment_helpers.psm1" in script
    assert script.index("Import-ModelDeckEnvironment") < script.index("if ($OpenDay)")


def test_checked_in_env_example_uses_only_supported_names() -> None:
    result = _run_pwsh(
        "Remove-Item Env:MODELDECK_HOST,Env:MODELDECK_MANAGEMENT_PORT,"
        "Env:MODELDECK_GATEWAY_PORT,Env:MODELDECK_DATA_DIR,Env:MODELDECK_LOG_DIR,"
        "Env:MODELDECK_OPEN_DAY,Env:MODELDECK_SCENECHAT_API_KEY,"
        "Env:MODELDECK_ALLOW_DOWNLOADS,Env:MODELDECK_DIAGNOSTIC_CAPTURE,"
        "Env:MODELDECK_DIFFUSION_TIMEOUT_SECONDS,Env:MODELDECK_SCENECHAT_TIMEOUT_SECONDS "
        "-ErrorAction SilentlyContinue; "
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{ENV_EXAMPLE}'"
    )

    assert result.returncode == 0, result.stderr
