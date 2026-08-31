from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

import httpx
import pytest
from modeldeck.workers import llama_vulkan_worker
from modeldeck.workers.llama_vulkan_worker import (
    amd_gpu_memory_metrics,
    classify_llama_startup_failure,
    llama_command,
    remove_reasoning,
)
from modeldeck.workers.moshiko_worker import speech_control_type, validate_start


def test_llama_command_uses_only_fixed_vulkan_presets(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"binary")
    model = tmp_path / "gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(llama_vulkan_worker, "fixed_llama_server", lambda: executable)

    full = llama_command(model=model, port=9630, context_length=8192, preset="vulkan-full")
    cpu_moe = llama_command(model=model, port=9630, context_length=8192, preset="vulkan-cpu-moe")

    assert full[0] == str(executable)
    assert full[full.index("--host") + 1] == "127.0.0.1"
    assert full.count("--flash-attn") == 1
    assert "on" not in full
    assert "--n-cpu-moe" not in full
    assert cpu_moe[-2:] == ["--n-cpu-moe", "20"]
    with pytest.raises(ValueError, match="allowlisted"):
        llama_command(model=model, port=9630, context_length=8192, preset="shell")


def test_llama_command_accepts_official_consolidated_mxfp4(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"binary")
    model = tmp_path / "gpt-oss-120b-MXFP4.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(llama_vulkan_worker, "fixed_llama_server", lambda: executable)

    command = llama_command(
        model=model,
        port=9630,
        context_length=8192,
        preset="vulkan-full",
    )

    assert command[command.index("--model") + 1] == str(model)


def test_llama_process_preserves_hugging_face_snapshot_filename(tmp_path) -> None:
    blob = tmp_path / "blobs" / "opaque-hash"
    blob.parent.mkdir()
    blob.write_bytes(b"gguf")
    snapshot = tmp_path / "snapshot" / "gpt-oss-120b-MXFP4.gguf"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)
    args = argparse.Namespace(port=9630, artifact_path=str(snapshot))

    runtime = llama_vulkan_worker.LlamaProcess(args)

    assert runtime.artifact_path.name == "gpt-oss-120b-MXFP4.gguf"
    assert runtime.artifact_path.is_file()


@pytest.mark.asyncio
async def test_llama_process_inherits_the_worker_process_group(monkeypatch, tmp_path) -> None:
    model = tmp_path / "gpt-oss-120b-MXFP4.gguf"
    model.write_bytes(b"gguf")
    args = argparse.Namespace(
        artifact_path=str(model),
        runtime_profile=None,
        context_length=8192,
        execution_preset="vulkan-full",
    )
    captured = {}

    class FakeProcess:
        returncode = None
        stdout = None
        stderr = None

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fake_subprocess(*command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(llama_vulkan_worker, "llama_command", lambda **_kwargs: ["llama-server"])
    monkeypatch.setattr(llama_vulkan_worker, "allocate_private_port", lambda: 49152)
    monkeypatch.setattr(llama_vulkan_worker.asyncio, "create_subprocess_exec", fake_subprocess)
    runtime = llama_vulkan_worker.LlamaProcess(args)

    await runtime.start()
    try:
        assert captured["command"] == ("llama-server",)
        assert "start_new_session" not in captured["kwargs"]
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_llama_process_records_first_ready_time(monkeypatch, tmp_path) -> None:
    model = tmp_path / "gpt-oss-120b-MXFP4.gguf"
    model.write_bytes(b"gguf")
    args = argparse.Namespace(port=9630, artifact_path=str(model))
    runtime = llama_vulkan_worker.LlamaProcess(args)
    runtime.process = SimpleNamespace(returncode=None)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr(llama_vulkan_worker.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    times = iter((100.0, 112.5, 120.0))
    runtime.started = next(times)
    monkeypatch.setattr(llama_vulkan_worker.time, "monotonic", lambda: next(times, 120.0))

    assert await runtime.ready() is True
    assert runtime.load_seconds == 12.5
    assert await runtime.ready() is True
    assert runtime.load_seconds == 12.5


@pytest.mark.asyncio
async def test_llama_process_reports_a_safe_child_load_failure(tmp_path) -> None:
    model = tmp_path / "gpt-oss-120b-MXFP4.gguf"
    model.write_bytes(b"gguf")
    runtime = llama_vulkan_worker.LlamaProcess(argparse.Namespace(port=9630, artifact_path=str(model)))
    stream = asyncio.StreamReader()
    stream.feed_data(b"ggml_vulkan: failed to allocate device memory\n")
    stream.feed_eof()

    await runtime._capture(stream)
    runtime.process = SimpleNamespace(returncode=1)

    assert runtime.child_failure() == {
        "failure_category": "accelerator_memory_allocation_failed",
        "child_exit_code": 1,
        "error": (
            "llama.cpp child exited during model loading with code 1: accelerator memory allocation failed"
        ),
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("failed to load model", "model_load_failed"),
        ("Vulkan initialization error", "vulkan_initialisation_failed"),
        ("ordinary progress output", None),
    ],
)
def test_llama_startup_failure_categories_do_not_retain_log_content(message, expected) -> None:
    assert classify_llama_startup_failure(message) == expected


@pytest.mark.asyncio
async def test_llama_health_reports_child_exit_as_failed(monkeypatch) -> None:
    failure = {
        "failure_category": "llama_child_exited",
        "child_exit_code": -6,
        "error": "llama.cpp child exited with code -6",
    }

    class FailedRuntime:
        qwen_runtime = None

        async def ready(self):
            return False

        def child_failure(self):
            return failure

    monkeypatch.setattr(llama_vulkan_worker, "LlamaProcess", lambda _args: FailedRuntime())
    app = llama_vulkan_worker.create_app(
        argparse.Namespace(
            worker_id="worker-id",
            model_id="ggml-org/gpt-oss-120b-GGUF",
            revision="a" * 40,
            runtime_profile=None,
            thinking_mode="adaptive",
            execution_preset="vulkan-full",
        )
    )
    health = next(route.endpoint for route in app.routes if route.path == "/health")

    payload = await health()

    assert payload["state"] == "failed"
    assert payload["ready"] is False
    assert payload["failure_category"] == "llama_child_exited"
    assert payload["child_exit_code"] == -6


def test_llama_response_filter_removes_reasoning_channels() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "private chain",
                    "content": "<|analysis|>private<|final|>Public answer",
                }
            }
        ]
    }

    filtered = remove_reasoning(payload)

    assert "reasoning_content" not in filtered["choices"][0]["message"]
    assert filtered["choices"][0]["message"]["content"] == "Public answer"


@pytest.mark.asyncio
async def test_llama_shutdown_requests_server_exit(tmp_path) -> None:
    model = tmp_path / "gpt-oss-120b-MXFP4.gguf"
    model.write_bytes(b"gguf")
    args = argparse.Namespace(
        worker_id="gpt-oss-test",
        model_id="ggml-org/gpt-oss-120b-GGUF",
        revision="pinned",
        port=9630,
        artifact_path=str(model),
    )
    app = llama_vulkan_worker.create_app(args)
    shutdown_requested = asyncio.Event()
    app.state.shutdown_callback = shutdown_requested.set

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/shutdown")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    await asyncio.wait_for(shutdown_requested.wait(), timeout=0.5)


def test_amd_gpu_memory_metrics_reads_fixed_card_sysfs(monkeypatch, tmp_path) -> None:
    drm = tmp_path / "drm"
    device = drm / "card1" / "device"
    device.mkdir(parents=True)
    values = {
        "vendor": "0x1002",
        "mem_info_gtt_used": "100",
        "mem_info_gtt_total": "200",
        "mem_info_vram_used": "10",
        "mem_info_vram_total": "20",
    }
    for name, value in values.items():
        (device / name).write_text(value, encoding="utf-8")
    original_path = llama_vulkan_worker.Path

    def fake_path(value):
        return original_path(drm) if value == "/sys/class/drm" else original_path(value)

    monkeypatch.setattr(llama_vulkan_worker, "Path", fake_path)

    assert amd_gpu_memory_metrics() == {
        "system_gtt_used_bytes": 100,
        "system_gtt_total_bytes": 200,
        "system_vram_used_bytes": 10,
        "system_vram_total_bytes": 20,
    }


def test_moshiko_session_start_is_strict() -> None:
    valid = {
        "type": "session.start",
        "model": "repartee-speech",
        "audio": {"encoding": "pcm_s16le", "sample_rate_hz": 24000, "channels": 1},
    }
    validate_start(valid, "repartee-speech")

    with pytest.raises(ValueError, match="24 kHz"):
        validate_start(
            {**valid, "audio": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}},
            "repartee-speech",
        )
    with pytest.raises(ValueError, match="match"):
        validate_start({**valid, "model": "another-model"}, "repartee-speech")
    assert speech_control_type('{"type":"session.close"}') == "session.close"
    assert speech_control_type('{"type":"response.cancel"}') == "response.cancel"
    with pytest.raises(ValueError, match="Unknown"):
        speech_control_type('{"type":"voice.change"}')
