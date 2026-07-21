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
            "completion_tokens": 100 + index,
            "schema_valid": benchmark._schema_valid(private_analysis),
            "analysis": private_analysis,
        }
        for index, value in enumerate((1.0, 2.0, 3.0))
    ]

    summary = benchmark._summary(samples, Counter({"generation_timeout": 1}))

    assert summary["latency_seconds"] == {"p50": 2.0, "p95": 2.9, "p99": 2.98}
    assert summary["valid_responses"] == 3
    assert summary["schema_valid_responses"] == 3
    assert summary["failure_categories"] == {"generation_timeout": 1}
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

    def failed_request(_gateway_url, _payload):
        nonlocal calls
        calls += 1
        return None, "generation_timeout"

    monkeypatch.setattr(benchmark, "_run_request", failed_request)
    worker = {"id": "worker-140", "settings": {"visual_token_budget": 140}}
    other = {"id": "worker-280", "settings": {"visual_token_budget": 280}}

    result = benchmark._benchmark_arm(
        "http://gateway",
        worker,
        [other],
        {},
        warmups=4,
        runs=50,
        human_review=False,
        maximum_temperature_celsius=80,
        cooldown_temperature_celsius=65,
        requests_per_thermal_batch=2,
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
