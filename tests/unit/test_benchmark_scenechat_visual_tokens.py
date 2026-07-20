from __future__ import annotations

import importlib.util
import sys
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
        {**base, "settings": {"visual_token_budget": 280}},
        {**base, "settings": {"visual_token_budget": 140}},
    )

    try:
        benchmark._validate_workers(
            {**base, "settings": {"visual_token_budget": 280}},
            {**base, "settings": {"visual_token_budget": 141}},
        )
    except RuntimeError as error:
        assert "budget 140" in str(error)
    else:
        raise AssertionError("A non-allowlisted comparison budget was accepted")


def test_benchmark_skips_measured_arm_when_all_warmups_fail(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_post", lambda _url: None)
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
        other,
        {},
        warmups=4,
        runs=50,
        human_review=False,
    )

    assert calls == 4
    assert result["benchmark_status"] == "not_run_no_valid_warmups"
    assert result["warmup_failure_categories"] == {"generation_timeout": 4}
    assert result["measured_requests"] == 0
