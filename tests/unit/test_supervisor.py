from __future__ import annotations

import socket
import sys

import pytest
from modeldeck.profiles import default_model_profiles
from modeldeck.protocol import LifecycleClass
from modeldeck.runtime_trust import TRUSTED_RUNTIME_IDS
from modeldeck.supervisor.service import (
    TRUSTED_LAUNCH_BUILDERS,
    WorkerSupervisor,
    build_mock_worker_command,
    build_worker_launch,
    classify_log_level,
    redact_log,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_every_trusted_runtime_has_an_explicit_launch_builder() -> None:
    assert set(TRUSTED_LAUNCH_BUILDERS) == set(TRUSTED_RUNTIME_IDS)


def test_worker_command_is_an_argument_array_with_allowlisted_values() -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    command = build_mock_worker_command(profile)
    assert command[:3] == [sys.executable, "-m", "modeldeck.workers.mock_worker"]
    port_index = command.index("--port")
    assert command[port_index : port_index + 2] == ["--port", "8610"]
    assert all(";" not in argument for argument in command)


def test_rocm_launch_requires_project_local_runtime(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    missing = tmp_path / "missing-python"
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(missing))
    with pytest.raises(ValueError, match="setup.ps1"):
        build_worker_launch(profile)


@pytest.mark.asyncio
async def test_supervisor_registers_and_removes_only_stopped_profiles() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([])
    supervisor.register_profile(base)

    assert supervisor.get_worker(base.id)["state"] == "stopped"
    await supervisor.remove_profile(base.id)

    with pytest.raises(KeyError, match="Unknown worker"):
        supervisor.get_worker(base.id)


def test_rocm_launch_preserves_virtual_environment_entrypoint(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    launch = build_worker_launch(profile)
    assert launch.command[0] == str(runtime_python.absolute())
    assert launch.command[0] != str(runtime_python.resolve())


@pytest.mark.parametrize("profile_id", ["qwen-small-rocm", "qwen-1-5b-rocm", "qwen-3b-rocm"])
def test_qwen_launches_are_allowlisted_offline_and_cache_pinned(monkeypatch, tmp_path, profile_id) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == profile_id)
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.autoregressive_worker",
    ]
    assert launch.command[launch.command.index("--model-id") + 1] == profile.model_id
    assert launch.command[launch.command.index("--revision") + 1] == profile.revision
    assert launch.command[launch.command.index("--port") + 1] == str(profile.port)
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == "/mnt/work/models/huggingface/hub"


def test_diffusion_rocm_launch_is_allowlisted_and_offline(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "diffusiongemma-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    launch = build_worker_launch(profile)
    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.text_diffusion_worker",
    ]
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == "/mnt/work/models/huggingface/hub"
    assert "LD_PRELOAD" not in launch.environment


def test_scenechat_launch_is_allowlisted_offline_and_api_key_scoped(monkeypatch, tmp_path) -> None:
    profile = next(
        profile for profile in default_model_profiles() if profile.id == "scenechat-gemma4-e2b-rocm"
    )
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    monkeypatch.setenv("MODELDECK_SCENECHAT_API_KEY", "test-local-key")

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.scenechat_worker",
    ]
    assert launch.command[launch.command.index("--port") + 1] == "8000"
    assert launch.command[launch.command.index("--cache-root") + 1] == ("/mnt/work/models/huggingface/hub")
    assert launch.command[launch.command.index("--maximum-new-tokens") + 1] == "512"
    assert launch.command[launch.command.index("--generation-timeout-seconds") + 1] == "60"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["MODELDECK_SCENECHAT_API_KEY"] == "test-local-key"


def test_diffusion_q4_launch_uses_isolated_runtime_and_checkpoint(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "diffusiongemma-q4-rocm")
    runtime_python = tmp_path / "q4/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_Q4_PYTHON", str(runtime_python))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    launch = build_worker_launch(profile)

    assert launch.command[0] == str(runtime_python.absolute())
    assert "--cache-root" not in launch.command
    assert launch.command[launch.command.index("--q4-checkpoint-dir") + 1].endswith(
        "/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32"
    )
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_HUB_CACHE" not in launch.environment


@pytest.mark.asyncio
async def test_starting_exclusive_worker_stops_existing_exclusive_worker() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-diffusion")
    first_port = free_port()
    second_port = free_port()
    while second_port == first_port:
        second_port = free_port()
    first = base.model_copy(update={"id": "mock-diffusion-one", "port": first_port})
    second = base.model_copy(update={"id": "mock-diffusion-two", "port": second_port})
    supervisor = WorkerSupervisor([first, second], startup_timeout=8, stop_timeout=2)

    try:
        await supervisor.start(first.id)
        await supervisor.start(second.id)

        assert supervisor.get_worker(first.id)["state"] == "stopped"
        assert supervisor.get_worker(first.id)["pid"] is None
        assert supervisor.get_worker(second.id)["state"] == "ready"
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_on_demand_worker_can_run_with_exclusive_worker() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-diffusion")
    exclusive_port = free_port()
    on_demand_port = free_port()
    while on_demand_port == exclusive_port:
        on_demand_port = free_port()
    exclusive = base.model_copy(update={"id": "mock-exclusive", "port": exclusive_port})
    on_demand = base.model_copy(
        update={
            "id": "mock-on-demand",
            "port": on_demand_port,
            "lifecycle": LifecycleClass.ON_DEMAND,
        }
    )
    supervisor = WorkerSupervisor([exclusive, on_demand], startup_timeout=8, stop_timeout=2)

    try:
        await supervisor.start(exclusive.id)
        await supervisor.start(on_demand.id)

        assert supervisor.get_worker(exclusive.id)["state"] == "ready"
        assert supervisor.get_worker(on_demand.id)["state"] == "ready"
    finally:
        await supervisor.stop_all()


def test_log_redaction_removes_prompt_and_credentials() -> None:
    assert redact_log("prompt=private visitor words") == "prompt=[redacted]"
    assert "secret" not in redact_log('{"api_key":"secret","status":"failed"}')


def test_worker_logs_are_redacted_bounded_and_restored(tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([profile], log_dir=tmp_path)
    supervisor._append_log(profile.id, "stderr", "prompt=private visitor words")
    for index in range(501):
        supervisor._append_log(profile.id, "stderr", f"diagnostic {index}")

    restored = WorkerSupervisor([profile], log_dir=tmp_path)
    logs = restored.logs(profile.id)

    assert len(logs) == 500
    assert all("private visitor words" not in item["message"] for item in logs)
    assert logs[-1]["message"] == "diagnostic 500"
    assert len((tmp_path / "mock-ar.jsonl").read_text().splitlines()) == 500


def test_worker_logs_are_scoped_to_the_current_session_and_classified(tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([profile], log_dir=tmp_path)
    worker = supervisor.workers[profile.id]
    worker.log_session_id = "first"
    supervisor._append_log(profile.id, "stderr", "ERROR: old failure")
    worker.log_session_id = "second"
    supervisor._append_log(profile.id, "stderr", "UserWarning: current warning")

    logs = supervisor.logs(profile.id)

    assert len(logs) == 1
    assert logs[0]["session_id"] == "second"
    assert logs[0]["level"] == "warning"
    assert classify_log_level("Traceback (most recent call last)") == "error"
    assert classify_log_level('{{- raise_exception("Invalid chat-template message") }}') == "info"
    assert classify_log_level("Application startup complete") == "info"
