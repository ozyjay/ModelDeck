from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx


def load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_models.py"
    spec = importlib.util.spec_from_file_location("modeldeck_benchmark_integration", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


def test_autoregressive_profile_runs_through_mocked_management_worker_and_gateway() -> None:
    requests = {"start": 0, "stop": 0, "trace": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/telemetry":
            return httpx.Response(
                200,
                json={
                    "memory": {"total_bytes": 64, "available_bytes": 32, "percent": 50},
                    "swap": {"total_bytes": 0, "used_bytes": 0, "percent": 0},
                    "temperatures": [{"source": "hwmon", "label": "GPU", "celsius": 55}],
                    "fans": [],
                    "filesystems": [{"path": "/private/path"}],
                    "active_model_processes": [{"command": "secret command"}],
                },
            )
        if request.url.path.endswith("/start"):
            requests["start"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "qwen-small-rocm",
                    "state": "ready",
                    "endpoint": "http://worker",
                },
            )
        if request.url.path.endswith("/stop"):
            requests["stop"] += 1
            return httpx.Response(
                200,
                json={"id": "qwen-small-rocm", "state": "stopped", "pid": None},
            )
        if request.url.path == "/model":
            return httpx.Response(
                200,
                json={
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "commit",
                    "dtype": "float16",
                    "quantization": "none",
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                json={
                    "runtime": "transformers-rocm",
                    "device_name": "AMD test GPU",
                    "torch_version": "test",
                    "hip_version": "7.2",
                    "transformers_version": "test",
                    "load_seconds": 1.25,
                    "memory_allocated_bytes": 1024,
                    "peak_memory_allocated_bytes": 2048,
                    "cache_root": "/private/cache",
                },
            )
        if request.url.path == "/native/autoregressive/trace":
            requests["trace"] += 1
            payload = json.loads(request.content)
            assert payload["min_tokens"] == payload["max_tokens"] == 64
            return httpx.Response(
                200,
                headers={"x-modeldeck-provider": "qwen-small-rocm"},
                json={
                    "prompt_token_ids": [1, 2],
                    "events": [{"text_so_far": "private generated output"}],
                    "metrics": {
                        "first_token_seconds": 0.1,
                        "total_seconds": 1.0,
                        "generated_tokens": 64,
                        "tokens_per_second": 64.0,
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    profile = {
        "id": "qwen-small-rocm",
        "alias": "qwen-0-5b",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "commit",
        "generation_family": "autoregressive",
        "preferred_runtime": "transformers-rocm",
        "dtype": "float16",
    }
    hardware = {
        "configured": {"profile_id": "framework-desktop", "gpu_architecture": "gfx1151"},
        "detected": {"fedora_release": "Fedora 44", "kernel": "test"},
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runner = benchmark.BenchmarkRunner(
            client,
            management_url="http://management",
            gateway_url="http://gateway",
            preset=benchmark.PRESETS["quick"],
        )
        result = runner.benchmark_profile(profile, hardware)

    assert requests == {"start": 1, "stop": 1, "trace": 3}
    assert result["status"] == "success"
    assert result["summary"]["successful_requests"] == 2
    assert result["summary"]["throughput_tokens_per_second"]["median"] == 64.0
    rendered = json.dumps(result)
    assert "private generated output" not in rendered
    assert "/private" not in rendered
    assert "secret command" not in rendered
