from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from modeldeck.speechshift import SPEECHSHIFT_MODEL_SPECS
from modeldeck.workers.speech_recognition_worker import (
    IsolatedWhisperRunner,
    RecognitionConfig,
    RecognitionRequestError,
)

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.rocm,
    pytest.mark.large_model,
    pytest.mark.long_running,
    pytest.mark.skipif(
        os.getenv("MODELDECK_RUN_WHISPER_HARDWARE_TESTS") != "1",
        reason="set MODELDECK_RUN_WHISPER_HARDWARE_TESTS=1 on the qualified ROCm host",
    ),
]


def _vram_usage_path() -> Path:
    candidates = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
    if not candidates:
        pytest.skip("The AMD VRAM usage counter is unavailable")
    return candidates[0]


def _vram_used(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


@pytest.mark.asyncio
async def test_whisper_cancellation_recovers_gpu_memory_and_allows_clean_restart() -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    cache_root = Path(os.getenv("HF_HUB_CACHE", "/mnt/work/models/huggingface/hub"))
    runner = IsolatedWhisperRunner(
        RecognitionConfig(spec.model_id, spec.revision, "speechshift-stt", cache_root),
        python_executable=Path(os.getenv("MODELDECK_WHISPER_PYTHON", ".venv-whisper-rocm72/bin/python")),
    )
    usage_path = _vram_usage_path()
    await runner.validate()
    baseline = _vram_used(usage_path)
    task = asyncio.create_task(runner.recognise(bytes(256_000)))
    allocation_seen = False
    for _ in range(1_200):
        if _vram_used(usage_path) > baseline + 256 * 1024 * 1024:
            allocation_seen = True
            break
        if task.done():
            break
        await asyncio.sleep(0.05)
    assert allocation_seen, "Whisper did not demonstrate a measurable GPU allocation"
    started = time.perf_counter()
    await runner.cancel()
    assert time.perf_counter() - started < 0.25
    with pytest.raises(RecognitionRequestError):
        await task

    deadline = time.monotonic() + 10
    while _vram_used(usage_path) > baseline + 64 * 1024 * 1024 and time.monotonic() < deadline:  # noqa: ASYNC110
        await asyncio.sleep(0.1)
    assert _vram_used(usage_path) <= baseline + 64 * 1024 * 1024

    first = await runner.recognise(bytes(3_200))
    second = await runner.recognise(bytes(3_200))
    assert first.inference_seconds > 0
    assert second.inference_seconds > 0
