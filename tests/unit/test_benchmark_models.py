from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest


def load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_models.py"
    spec = importlib.util.spec_from_file_location("modeldeck_benchmark_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


def profile(family: str, *, profile_id: str = "qwen-small-rocm", alias: str = "qwen-0-5b"):
    return {
        "id": profile_id,
        "alias": alias,
        "model_id": "example/model",
        "revision": "commit",
        "generation_family": family,
        "preferred_runtime": "transformers-rocm",
        "dtype": "float16",
    }


def runner_for(handler, *, repetitions: int = 2):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = benchmark.BenchmarkRunner(
        client,
        management_url="http://management",
        gateway_url="http://gateway",
        preset=benchmark.BenchmarkPreset(repetitions=repetitions),
    )
    return runner, client


def response(payload: Any, *, provider: str | None = None, status: int = 200) -> httpx.Response:
    headers = {"x-modeldeck-provider": provider} if provider else None
    return httpx.Response(status, json=payload, headers=headers)


def test_presets_and_model_selection_are_stable() -> None:
    assert benchmark.PRESETS["quick"].repetitions == 2
    assert benchmark.PRESETS["standard"].repetitions == 5
    assert benchmark.PRESETS["standard"].autoregressive_tokens == 64
    assert benchmark.PRESETS["standard"].diffusion_tokens == 128
    assert benchmark.PRESETS["standard"].diffusion_steps == 24
    assert benchmark.PRESETS["standard"].vision_tokens == 256
    assert benchmark.validate_models(["qwen-small-rocm", "qwen-small-rocm"]) == ["qwen-small-rocm"]

    with pytest.raises(benchmark.BenchmarkError, match="Unknown or non-physical"):
        benchmark.validate_models(["mock-ar"])


def test_summary_uses_nearest_rank_p95_and_tracks_determinism() -> None:
    samples = [
        {"wall_seconds": value, "output_sha256": "same", "throughput_tokens_per_second": 10 + value}
        for value in (1.0, 2.0, 3.0, 4.0, 5.0)
    ]

    summary = benchmark.summarise_samples(samples, [], requested=5)

    assert summary["wall_seconds"] == {
        "minimum": 1.0,
        "median": 3.0,
        "p95": 5.0,
        "maximum": 5.0,
    }
    assert summary["deterministic_outputs"] is True
    assert summary["successful_requests"] == 5


def test_autoregressive_workload_uses_fixed_request_and_provider() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return response(
            {
                "prompt_token_ids": [1, 2, 3],
                "events": [{"text_so_far": "benchmark output"}],
                "metrics": {
                    "first_token_seconds": 0.1,
                    "total_seconds": 1.0,
                    "generated_tokens": 64,
                    "tokens_per_second": 64.0,
                },
            },
            provider="qwen-small-rocm",
        )

    runner, client = runner_for(handler)
    try:
        result = runner.run_autoregressive(profile("autoregressive"))
    finally:
        client.close()

    assert captured["seed"] == 7
    assert captured["min_tokens"] == captured["max_tokens"] == 64
    assert captured["temperature"] == 0
    assert result["generated_tokens"] == 64
    assert "benchmark output" not in json.dumps(result)


def test_diffusion_workload_uses_fixed_request_and_provider() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/diffuse":
            captured.update(json.loads(request.content))
            return response({"job_id": "job-1"}, provider="diffusiongemma-q4-rocm")
        assert request.url.path == "/v1/jobs/job-1"
        return response(
            {
                "job_id": "job-1",
                "state": "complete",
                "text": "benchmark output",
                "metrics": {"total_seconds": 2.5},
            }
        )

    runner, client = runner_for(handler)
    try:
        result = runner.run_diffusion(
            profile(
                "text-diffusion",
                profile_id="diffusiongemma-q4-rocm",
                alias="text-diffusion",
            )
        )
    finally:
        client.close()

    assert captured["max_length"] == captured["block_length"] == 128
    assert captured["denoising_steps"] == 24
    assert captured["seed"] == 11
    assert result["worker_seconds"] == 2.5
    assert "benchmark output" not in json.dumps(result)


def test_vision_workload_uses_synthetic_image_and_approved_contract() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "A synthetic field.",
                                    "objects": [],
                                    "relationships": [],
                                    "uncertainties": [],
                                    "safety_notes": [],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30},
            },
            provider="scenechat-gemma4-e2b-rocm",
        )

    runner, client = runner_for(handler)
    try:
        result = runner.run_vision(
            profile(
                "vision-language",
                profile_id="scenechat-gemma4-e2b-rocm",
                alias="scenechat-vision",
            )
        )
    finally:
        client.close()

    content = captured["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Describe the scene." in content[1]["text"]
    assert captured["max_tokens"] == 256
    assert result["generated_tokens"] == 30
    assert "data:image" not in json.dumps(result)
    assert "A synthetic field" not in json.dumps(result)


def test_workload_rejects_mock_fallback_provider() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return response(
            {
                "prompt_token_ids": [1],
                "events": [{"text_so_far": "mock output"}],
                "metrics": {"generated_tokens": 1},
            },
            provider="mock-ar",
        )

    runner, client = runner_for(handler)
    try:
        with pytest.raises(benchmark.BenchmarkError, match="expected qwen-small-rocm"):
            runner.run_autoregressive(profile("autoregressive"))
    finally:
        client.close()


def test_error_sanitisation_removes_credentials_images_and_local_paths() -> None:
    error = RuntimeError("Bearer secret-token hf_abc123 data:image/png;base64,AAAA /mnt/work/private/model")

    safe = benchmark.sanitise_error(error)

    rendered = json.dumps(safe)
    assert "secret-token" not in rendered
    assert "hf_abc123" not in rendered
    assert "AAAA" not in rendered
    assert "/mnt/work" not in rendered


def test_preflight_refuses_any_busy_managed_worker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/profiles":
            return response([profile("autoregressive")])
        if request.url.path == "/api/workers":
            return response([{"id": "mock-ar", "state": "busy"}])
        raise AssertionError(f"Unexpected request: {request.url}")

    runner, client = runner_for(handler)
    try:
        with pytest.raises(benchmark.BenchmarkError, match="mock-ar is busy"):
            runner.preflight(["qwen-small-rocm"])
    finally:
        client.close()


class FakeLifecycleRunner:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.preset = benchmark.PRESETS["quick"]
        self.stop_calls = 0
        self.restored: list[str] | None = None
        self.interrupt = interrupt

    def preflight(self, _selected):
        profiles = [
            profile("autoregressive", profile_id="qwen-small-rocm"),
            profile("autoregressive", profile_id="qwen-1-5b-rocm", alias="qwen-1-5b"),
        ]
        return profiles, {"configured": {}, "detected": {}}

    def workers(self):
        return [
            {"id": "qwen-small-rocm", "state": "ready"},
            {"id": "qwen-1-5b-rocm", "state": "stopped"},
        ]

    def stop_all(self):
        self.stop_calls += 1

    def benchmark_profile(self, selected_profile, _hardware):
        if selected_profile["id"] == "qwen-small-rocm":
            if self.interrupt:
                raise KeyboardInterrupt
            raise RuntimeError("request failed")
        return {
            "profile_id": selected_profile["id"],
            "generation_family": "autoregressive",
            "status": "success",
            "samples": [],
            "summary": {},
        }

    def restore(self, initially_ready):
        self.restored = initially_ready
        return {
            "requested_ready_profiles": initially_ready,
            "outcomes": [{"profile_id": item, "status": "ready"} for item in initially_ready],
            "passed": True,
        }


def test_run_continues_after_model_failure_and_restores_initial_state() -> None:
    runner = FakeLifecycleRunner()

    report = benchmark.run_benchmark(
        runner,
        selected=["qwen-small-rocm", "qwen-1-5b-rocm"],
        preset_name="quick",
    )

    assert [result["status"] for result in report["results"]] == ["failed", "success"]
    assert report["status"] == "completed-with-failures"
    assert benchmark.report_exit_code(report) == 1
    assert runner.restored == ["qwen-small-rocm"]
    assert runner.stop_calls == 2


def test_run_restores_initial_state_after_interruption() -> None:
    runner = FakeLifecycleRunner(interrupt=True)

    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(
            runner,
            selected=["qwen-small-rocm"],
            preset_name="quick",
        )

    assert runner.restored == ["qwen-small-rocm"]
    assert runner.stop_calls == 2


def test_versioned_reports_do_not_contain_workload_content(tmp_path: Path) -> None:
    report = {
        "format": benchmark.REPORT_FORMAT,
        "format_version": benchmark.REPORT_VERSION,
        "run_id": "benchmark-test",
        "status": "completed",
        "configuration": {"preset": "quick"},
        "results": [
            {
                "profile_id": "qwen-small-rocm",
                "generation_family": "autoregressive",
                "status": "success",
                "cold_start_wall_seconds": 1.0,
                "metrics_after": {"peak_memory_allocated_bytes": 1024},
                "summary": {
                    "wall_seconds": {"median": 2.0, "p95": 2.2},
                    "throughput_tokens_per_second": {"median": 30.0},
                },
            }
        ],
        "restoration": {"passed": True},
    }
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    benchmark.write_reports(report, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format"] == "modeldeck-benchmark"
    assert payload["format_version"] == 1
    assert "autoregressive" in markdown_path.read_text(encoding="utf-8")
