from __future__ import annotations

import sys

import pytest
from modeldeck.profiles import default_model_profiles
from modeldeck.supervisor.service import (
    WorkerSupervisor,
    build_mock_worker_command,
    build_worker_launch,
    redact_log,
)


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


def test_rocm_launch_preserves_virtual_environment_entrypoint(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    launch = build_worker_launch(profile)
    assert launch.command[0] == str(runtime_python.absolute())
    assert launch.command[0] != str(runtime_python.resolve())


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


def test_diffusion_q4_launch_uses_isolated_runtime_and_checkpoint(monkeypatch, tmp_path) -> None:
    profile = next(
        profile for profile in default_model_profiles() if profile.id == "diffusiongemma-q4-rocm"
    )
    runtime_python = tmp_path / "q4/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_Q4_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[0] == str(runtime_python.absolute())
    assert launch.command[launch.command.index("--cache-root") + 1] == (
        "/mnt/work/models/huggingface/hub"
    )
    assert launch.command[launch.command.index("--q4-checkpoint-dir") + 1].endswith(
        "diffusiongemma-26b-a4b-it-gptq-q4-g32"
    )
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"


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
