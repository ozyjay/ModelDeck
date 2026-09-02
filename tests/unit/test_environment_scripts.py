from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPERS = PROJECT_ROOT / "scripts/lib/environment_helpers.psm1"
MODELDECK_HELPERS = PROJECT_ROOT / "scripts/lib/modeldeck_helpers.psm1"
RUN_SCRIPT = PROJECT_ROOT / "scripts/operations/run.ps1"
STOP_SCRIPT = PROJECT_ROOT / "scripts/operations/stop.ps1"
STOP_STALE_WORKERS_SCRIPT = PROJECT_ROOT / "scripts/operations/stop_stale_workers.ps1"
CHECK_PORTS_SCRIPT = PROJECT_ROOT / "scripts/operations/check_ports.ps1"
SETUP_LLAMA_VULKAN_SCRIPT = PROJECT_ROOT / "scripts/setup/setup_llama_vulkan.ps1"
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


def test_run_script_loads_dotenv_before_configuration_lock_overrides() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "environment_helpers.psm1" in script
    assert script.index("Import-ModelDeckEnvironment") < script.index("if ($LockConfiguration)")


def test_run_script_starts_an_explicit_docker_bridge_companion() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert "MODELDECK_ENABLE_DOCKER_BRIDGE" in script
    assert "gateway-docker-bridge.pid" in script
    assert "$Env:MODELDECK_GATEWAY_HOST = '172.17.0.1'" in script
    assert "Remove-Item var/run/gateway-loopback.pid" in script
    assert "-m', 'modeldeck.gateway.app'" in script
    assert "-m', 'modeldeck'" in script


def test_stop_script_reports_each_shutdown_stage_and_service_outcome() -> None:
    script = STOP_SCRIPT.read_text(encoding="utf-8")

    assert "[1/4] Requesting graceful Worker shutdown" in script
    assert "[2/4] Stopping ModelDeck services" in script
    assert "[3/4] Checking for stale ModelDeck Workers" in script
    assert "[4/4] ModelDeck stopped:" in script
    assert "not running (no PID file)" in script
    assert "did not stop gracefully; forcing process" in script
    assert "gateway-docker-bridge" in script
    assert "gateway-loopback" in script
    assert "'stop_stale_workers.ps1')\n" in script


def test_stop_script_recovers_project_local_services_without_pid_files() -> None:
    script = STOP_SCRIPT.read_text(encoding="utf-8")

    assert "Find-ModelDeckProcessIds" in script
    assert ".venv/bin/modeldeck-gateway" in script
    assert "recovered untracked" in script
    assert "modeldeck.gateway.app" in script


def test_stale_worker_cleanup_covers_all_managed_workers_and_private_llama_server() -> None:
    script = STOP_STALE_WORKERS_SCRIPT.read_text(encoding="utf-8")

    assert "@(8610..8699)" in script
    assert "modeldeck.workers.gemma4_chat_worker" in script
    assert "modeldeck.workers.llama_vulkan_worker" in script
    assert "modeldeck.workers.qwen35_chat_worker" in script
    assert "modeldeck.workers.scenechat_worker" in script
    assert '"$Root/.runtime-tools/llama.cpp/bin/llama-server"' in script
    assert "private llama-server" in script
    assert "$Arguments[0] -eq $TrustedLlamaServer" in script


def test_llama_vulkan_setup_accepts_an_explicit_runtime_root() -> None:
    script = SETUP_LLAMA_VULKAN_SCRIPT.read_text(encoding="utf-8")

    assert "[string]$RuntimeRoot = ''" in script
    assert "GetFullPath($RuntimeRoot)" in script
    assert "else { '.runtime-tools/llama.cpp' }" in script


def test_port_check_preserves_the_binding_details_in_its_error() -> None:
    script = CHECK_PORTS_SCRIPT.read_text(encoding="utf-8")

    assert "$Binding = $_" in script
    assert "ModelDeck $($Binding.Name) cannot bind $($Binding.Host):$($Binding.Port)" in script


def test_checked_in_env_example_uses_only_supported_names() -> None:
    result = _run_pwsh(
        "Remove-Item Env:MODELDECK_HOST,Env:MODELDECK_MANAGEMENT_PORT,"
        "Env:MODELDECK_GATEWAY_HOST,Env:MODELDECK_GATEWAY_PORT,Env:MODELDECK_ENABLE_DOCKER_BRIDGE,Env:MODELDECK_DATA_DIR,Env:MODELDECK_LOG_DIR,"
        "Env:MODELDECK_CONFIGURATION_LOCKED,Env:MODELDECK_SCENECHAT_API_KEY,"
        "Env:MODELDECK_DIAGNOSTIC_CAPTURE,"
        "Env:MODELDECK_DIFFUSION_TIMEOUT_SECONDS,Env:MODELDECK_SCENECHAT_TIMEOUT_SECONDS "
        "-ErrorAction SilentlyContinue; "
        f"Import-Module '{HELPERS}' -Force; Import-ModelDeckEnvironment -Path '{ENV_EXAMPLE}'"
    )

    assert result.returncode == 0, result.stderr


def test_worker_resolver_enumerates_invoke_rest_method_json_array() -> None:
    result = _run_pwsh(
        "function global:Invoke-RestMethod { "
        "$workers = @([pscustomobject]@{ id='worker-qwen'; name='Qwen'; runtime='transformers-rocm' }, "
        "[pscustomobject]@{ id='worker-q4'; name='DiffusionGemma Q4'; runtime='text-diffusion-gptq-rocm' }); "
        "return ,$workers }; "
        f"Import-Module '{MODELDECK_HELPERS}' -Force; "
        "Resolve-ModelDeckWorker -ManagementUrl 'http://127.0.0.1:3600' -Worker 'worker-q4' "
        "| Select-Object id,name | ConvertTo-Json -Compress"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"id": "worker-q4", "name": "DiffusionGemma Q4"}
