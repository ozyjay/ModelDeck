"""Opt-in qualification checks for an already-running native-FP8 Qwen3.8 text Worker."""

from __future__ import annotations

import os
import statistics

import httpx
import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.rocm,
    pytest.mark.large_model,
    pytest.mark.long_running,
    pytest.mark.skipif(
        os.getenv("MODELDECK_RUN_QWEN38_FP8_TESTS") != "1",
        reason="set MODELDECK_RUN_QWEN38_FP8_TESTS=1 on the qualified ROCm host",
    ),
]


@pytest.mark.asyncio
async def test_native_fp8_text_worker_meets_promotion_gates_across_five_repeats() -> None:
    worker_url = os.getenv("MODELDECK_QWEN38_FP8_WORKER_URL", "http://127.0.0.1:8610").rstrip("/")
    samples = []
    async with httpx.AsyncClient(base_url=worker_url, timeout=300.0) as client:
        for _ in range(5):
            response = await client.post(
                "/native/autoregressive/trace",
                json={
                    "automatic": True,
                    "prompt": "Summarise the role of local inference in one concise paragraph.",
                    "seed": 7,
                    "max_tokens": 64,
                    "min_tokens": 64,
                    "temperature": 0,
                    "top_k": 1,
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            assert payload["events"]
            samples.append(payload["metrics"])
        metrics_response = await client.get("/metrics")

    metrics_response.raise_for_status()
    runtime = metrics_response.json()
    assert runtime["execution_mode"] == "native_fp8"
    assert runtime["tuning_status"] == "validated"
    assert runtime["memory_allocated_bytes"] <= 36 * 1024**3
    assert runtime["peak_memory_allocated_bytes"] <= 38 * 1024**3
    assert statistics.median(sample["tokens_per_second"] for sample in samples) >= 3.20
    assert statistics.median(sample["first_token_seconds"] for sample in samples) <= 0.50
