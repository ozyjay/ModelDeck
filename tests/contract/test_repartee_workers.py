from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

import httpx
import pytest
from modeldeck.workers import llama_vulkan_worker
from modeldeck.workers.llama_vulkan_worker import (
    amd_gpu_memory_metrics,
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
