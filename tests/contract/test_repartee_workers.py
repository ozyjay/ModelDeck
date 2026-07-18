from __future__ import annotations

import pytest
from modeldeck.workers import llama_vulkan_worker
from modeldeck.workers.llama_vulkan_worker import llama_command, remove_reasoning
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
    assert "--n-cpu-moe" not in full
    assert cpu_moe[-2:] == ["--n-cpu-moe", "20"]
    with pytest.raises(ValueError, match="allowlisted"):
        llama_command(model=model, port=9630, context_length=8192, preset="shell")


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
