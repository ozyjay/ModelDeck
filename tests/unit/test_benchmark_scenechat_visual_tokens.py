from __future__ import annotations

import importlib.util
import sys
import threading
from collections import Counter
from pathlib import Path


def load_benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_scenechat_visual_tokens.py"
    spec = importlib.util.spec_from_file_location("scenechat_visual_token_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


class SafeThermalGuard:
    def __init__(self, _maximum_celsius: float) -> None:
        self.maximum_observed_celsius = 60.0
        self.maximum_gpu_celsius = 55.0
        self.maximum_observed_sensor = {
            "source": "k10temp",
            "label": "Tctl",
            "celsius": 60.0,
        }
        self.maximum_gpu_sensor = {
            "source": "amdgpu",
            "label": "edge",
            "celsius": 55.0,
        }
        self.trigger_sensor = None
        self.sample_count = 2
        self.abort_reason = None
        self.triggered = threading.Event()
        self.worker_id = None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def wait_for_cooldown(self, _target: float, maximum_wait_seconds: float = 600) -> float:
        del maximum_wait_seconds
        return 1.5


def test_benchmark_summary_retains_metrics_but_not_descriptions() -> None:
    private_analysis = {
        "summary": "PRIVATE booth description",
        "objects": [
            {
                "label": "display",
                "description": "PRIVATE object description",
                "approximate_location": "centre",
            }
        ],
        "relationships": [],
        "uncertainties": [],
        "safety_notes": [],
    }
    samples = [
        {
            "latency_seconds": value,
            "prompt_tokens": 400,
            "completion_tokens": 100 + index,
            "tokens_per_second": 40.0,
            "visual_tokens": 140,
            "preprocessing_seconds": 0.1,
            "inference_seconds": 1.5,
            "validation_seconds": 0.01,
            "total_worker_seconds": 1.8,
            "gateway_overhead_seconds": 0.2,
            "finish_reason": "stop",
            "schema_valid": benchmark._schema_valid(private_analysis),
            "analysis": private_analysis,
        }
        for index, value in enumerate((1.0, 2.0, 3.0))
    ]

    summary = benchmark._summary(
        samples,
        Counter({"timeout": 1}),
        token_limit=1024,
    )

    assert summary["latency_seconds"] == {"p50": 2.0, "p95": 2.9, "p99": 2.98}
    assert summary["valid_responses"] == 3
    assert summary["schema_valid_responses"] == 3
    assert summary["failure_categories"] == {"timeout": 1}
    assert summary["completion_tokens"]["limit"] == 1024
    assert summary["completion_tokens"]["at_limit"] == 0
    assert summary["finish_reasons"] == {"stop": 3}
    assert summary["stage_seconds"]["inference_seconds"]["p95"] == 1.5
    assert "PRIVATE" not in str(summary)


def test_benchmark_requires_matching_pinned_workers_and_exact_budgets() -> None:
    base = {
        "name": "Gemma",
        "generation_family": "vision-language",
        "model_id": "google/gemma-4-12B-it",
        "revision": "pinned",
    }
    benchmark._validate_workers(
        [
            {**base, "settings": {"visual_token_budget": 70}},
            {**base, "settings": {"visual_token_budget": 140}},
            {**base, "settings": {"visual_token_budget": 280}},
        ]
    )

    try:
        benchmark._validate_workers([{**base, "settings": {"visual_token_budget": 141}}])
    except RuntimeError as error:
        assert "allowlisted visual token budget" in str(error)
    else:
        raise AssertionError("A non-allowlisted comparison budget was accepted")


def test_benchmark_skips_measured_arm_when_all_warmups_fail(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_post", lambda _url, *, timeout=900: None)
    monkeypatch.setattr(benchmark, "_wait_for_cooldown", lambda _target: None)
    monkeypatch.setattr(benchmark, "ThermalGuard", SafeThermalGuard)
    calls = 0

    def failed_request(_gateway_url, _worker, _payload, **_kwargs):
        nonlocal calls
        calls += 1
        return None, "generation_timeout"

    monkeypatch.setattr(benchmark, "_run_request", failed_request)
    worker = {
        "id": "worker-140",
        "runtime": "qwen35-vision-language-transformers-rocm",
        "runtime_template_id": "scenechat-qwen35",
        "runtime_template_version": "0.2.0",
        "dtype": "bfloat16",
        "settings": {
            "context_length": 8192,
            "maximum_new_tokens": 1024,
            "visual_token_budget": 140,
            "generation_timeout_seconds": 60,
        },
    }
    other = {"id": "worker-280", "settings": {"visual_token_budget": 280}}

    result = benchmark._benchmark_arm(
        "http://gateway",
        worker,
        [other],
        "data:image/png;base64,fixed",
        warmups=4,
        runs_per_question=10,
        human_review=False,
        maximum_temperature_celsius=80,
        cooldown_temperature_celsius=65,
        requests_per_thermal_batch=2,
        minimum_duration_seconds=0,
    )

    assert calls == 4
    assert result["benchmark_status"] == "not_run_no_valid_warmups"
    assert result["warmup_failure_categories"] == {"generation_timeout": 4}
    assert result["measured_requests"] == 0
    assert result["thermal"] == {
        "maximum_allowed_celsius": 80,
        "cooldown_target_celsius": 65,
        "maximum_observed_celsius": 60.0,
        "maximum_gpu_celsius": 55.0,
        "maximum_observed_sensor": {
            "source": "k10temp",
            "label": "Tctl",
            "celsius": 60.0,
        },
        "maximum_gpu_sensor": {
            "source": "amdgpu",
            "label": "edge",
            "celsius": 55.0,
        },
        "sample_count": 2,
        "abort": False,
        "abort_reason": None,
        "trigger_sensor": None,
    }
    assert result["pacing"] == {
        "requests_per_batch": 2,
        "cooldown_events": 1,
        "total_cooldown_seconds": 1.5,
    }


def test_benchmark_measures_every_curated_question_ten_times(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_post", lambda _url, *, timeout=900: None)
    monkeypatch.setattr(benchmark, "_wait_for_cooldown", lambda _target: None)
    monkeypatch.setattr(benchmark, "ThermalGuard", SafeThermalGuard)
    question_ids: list[str] = []

    def successful_request(_gateway_url, _worker, _payload, *, question_id, token_limit):
        assert token_limit == 1024
        question_ids.append(question_id)
        return (
            {
                "question_id": question_id,
                "latency_seconds": 7.0,
                "prompt_tokens": 400,
                "completion_tokens": 220,
                "tokens_per_second": 35.0,
                "visual_tokens": 140,
                "preprocessing_seconds": 0.1,
                "inference_seconds": 6.5,
                "validation_seconds": 0.01,
                "total_worker_seconds": 6.7,
                "gateway_overhead_seconds": 0.3,
                "finish_reason": "stop",
                "schema_valid": True,
                "analysis": {
                    "summary": "A fixed synthetic scene.",
                    "objects": [],
                    "relationships": [],
                    "uncertainties": [],
                    "safety_notes": [],
                },
            },
            None,
        )

    monkeypatch.setattr(benchmark, "_run_request", successful_request)
    worker = {
        "id": "worker-140",
        "runtime": "qwen35-vision-language-transformers-rocm",
        "runtime_template_id": "scenechat-qwen35",
        "runtime_template_version": "0.2.0",
        "dtype": "bfloat16",
        "settings": {
            "context_length": 8192,
            "maximum_new_tokens": 1024,
            "visual_token_budget": 140,
            "generation_timeout_seconds": 60,
        },
    }

    result = benchmark._benchmark_arm(
        "http://gateway",
        worker,
        [],
        "data:image/png;base64,fixed",
        warmups=2,
        runs_per_question=10,
        human_review=False,
        maximum_temperature_celsius=80,
        cooldown_temperature_celsius=65,
        requests_per_thermal_batch=100,
        minimum_duration_seconds=0,
    )

    assert len(question_ids) == 72
    assert question_ids[:2] == ["question-01", "question-02"]
    assert [question["question_id"] for question in result["questions"]] == [
        f"question-{index:02d}" for index in range(1, 8)
    ]
    assert all(question["measured_requests"] == 10 for question in result["questions"])
    assert result["measured_requests"] == 70
    assert result["acceptance"] == {
        "all_questions_complete": True,
        "zero_failures": True,
        "zero_token_limit_hits": True,
        "median_at_most_8_seconds": True,
        "p95_at_most_12_seconds": True,
        "per_question_completion_p95_below_limit": True,
        "preferred_per_question_completion_p95_at_most_260": True,
    }
    assert "fixed synthetic scene" not in str(result).casefold()


def test_thermal_guard_stops_worker_at_temperature_limit(monkeypatch) -> None:
    stopped: list[tuple[str, int]] = []
    monkeypatch.setattr(
        benchmark,
        "_temperatures",
        lambda: [
            {"source": "amdgpu", "celsius": 72.5},
            {"source": "k10temp", "celsius": 80.0},
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "_post",
        lambda url, *, timeout=900: stopped.append((url, timeout)),
    )
    guard = benchmark.ThermalGuard(80)
    guard.worker_id = "worker-140"

    guard.start()
    assert guard.triggered.wait(2)
    guard.close()

    assert guard.abort_reason == "temperature_limit"
    assert guard.maximum_observed_celsius == 80.0
    assert guard.maximum_gpu_celsius == 72.5
    assert guard.maximum_observed_sensor == {
        "source": "k10temp",
        "label": "k10temp",
        "celsius": 80.0,
    }
    assert guard.maximum_gpu_sensor == {
        "source": "amdgpu",
        "label": "amdgpu",
        "celsius": 72.5,
    }
    assert guard.trigger_sensor == {
        "source": "k10temp",
        "label": "k10temp",
        "celsius": 80.0,
    }
    assert guard.sample_count == 1
    assert stopped == [
        ("http://127.0.0.1:3600/api/workers/worker-140/stop", 30),
    ]
