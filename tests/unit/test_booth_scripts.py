import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS = PROJECT_ROOT / "scripts" / "booth_helpers.psm1"


def _run_pwsh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _browser_arguments(*, windowed: bool) -> list[str]:
    windowed_argument = " -Windowed" if windowed else ""
    result = _run_pwsh(
        f"Import-Module '{HELPERS}'; "
        "@(Get-BoothBrowserArguments -Url 'http://127.0.0.1:3600' "
        f"-ProfileDirectory '/tmp/modeldeck-booth-test'{windowed_argument}) "
        "| ConvertTo-Json -Compress"
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_booth_browser_arguments_use_isolated_fullscreen_profile() -> None:
    arguments = _browser_arguments(windowed=False)

    assert "--user-data-dir=/tmp/modeldeck-booth-test" in arguments
    assert "--kiosk" in arguments
    assert arguments[-1] == "http://127.0.0.1:3600"


def test_booth_windowed_arguments_use_app_window() -> None:
    arguments = _browser_arguments(windowed=True)

    assert "--kiosk" not in arguments
    assert "--app=http://127.0.0.1:3600" in arguments


def test_booth_browser_lookup_has_clear_missing_browser_error() -> None:
    result = _run_pwsh(
        f"Import-Module '{HELPERS}'; Resolve-BoothBrowser -Browser '/missing/modeldeck-booth-browser'"
    )

    assert result.returncode != 0
    assert "Configured booth browser was not found" in result.stderr
