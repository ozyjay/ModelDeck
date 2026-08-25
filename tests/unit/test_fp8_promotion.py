from __future__ import annotations

import pytest
from modeldeck.v2_api import (
    _aggregate_native_fp8_text_benchmark,
    _default_runtime_template_id,
    _has_matching_native_fp8_text_promotion,
    _validate_native_fp8_promotion,
)


def test_qwen38_defaults_remain_on_the_independent_bf16_workers() -> None:
    assert _default_runtime_template_id("scenechat-qwen38-fp8", "general-chat") == (
        "qwen35-chat-transformers-rocm"
    )
    assert _default_runtime_template_id("scenechat-qwen38-fp8", "general-image-chat") == ("scenechat-qwen35")
    assert _default_runtime_template_id("scenechat-qwen35", "general-chat") == "scenechat-qwen35"


def _passing_metrics() -> dict[str, object]:
    return {
        "tuning_status": "validated",
        "memory_allocated_bytes": 35 * 1024**3,
        "peak_memory_allocated_bytes": 37 * 1024**3,
    }


def test_native_fp8_chat_promotion_requires_speed_latency_memory_and_tuning() -> None:
    _validate_native_fp8_promotion(
        "qwen38-fp8-chat-transformers-rocm",
        _passing_metrics(),
        {"metrics": {"tokens_per_second": 3.2, "first_token_seconds": 0.5}},
    )

    with pytest.raises(RuntimeError, match="3.20 tokens/s"):
        _validate_native_fp8_promotion(
            "qwen38-fp8-chat-transformers-rocm",
            _passing_metrics(),
            {"metrics": {"tokens_per_second": 3.19, "first_token_seconds": 0.4}},
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tuning_status": "stale"}, "tuning profile"),
        ({"memory_allocated_bytes": 36 * 1024**3 + 1}, "memory_allocated_bytes"),
        ({"peak_memory_allocated_bytes": 38 * 1024**3 + 1}, "peak_memory_allocated_bytes"),
    ],
)
def test_native_fp8_promotion_fails_closed(change, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validate_native_fp8_promotion(
            "qwen38-fp8-chat-transformers-rocm",
            {**_passing_metrics(), **change},
            {"metrics": {"tokens_per_second": 4.0, "first_token_seconds": 0.4}},
        )


def test_bf16_runtime_is_not_subject_to_native_fp8_promotion_gate() -> None:
    _validate_native_fp8_promotion("qwen35-chat-transformers-rocm", {}, {})


def test_native_fp8_promotion_uses_five_repeat_medians() -> None:
    payloads = [
        {
            "events": [{"text_so_far": "ready"}],
            "metrics": {"tokens_per_second": throughput, "first_token_seconds": first_token},
        }
        for throughput, first_token in zip(
            (3.0, 3.1, 3.2, 3.3, 9.0),
            (0.1, 0.2, 0.3, 0.4, 2.0),
            strict=True,
        )
    ]

    aggregate = _aggregate_native_fp8_text_benchmark(payloads)

    assert aggregate["metrics"]["tokens_per_second"] == 3.2
    assert aggregate["metrics"]["first_token_seconds"] == 0.3
    assert aggregate["metrics"]["benchmark_repetitions"] == 5
    with pytest.raises(RuntimeError, match="five benchmark repeats"):
        _aggregate_native_fp8_text_benchmark(payloads[:4])


def test_vision_promotion_requires_matching_native_fp8_text_evidence() -> None:
    metrics = {
        "kernel_commit": "commit",
        "kernel_manifest_sha256": "kernel-hash",
        "tuning_profile_sha256": "tuning-hash",
    }
    evidence = {
        "model_id": "Qwen/Qwen3.8-27B-FP8",
        "model_revision": "revision",
        "runtime": "qwen38-fp8-chat-transformers-rocm",
        "execution_mode": "native_fp8",
        **metrics,
    }

    assert _has_matching_native_fp8_text_promotion(
        [{"result": "tested-working", "evidence": evidence}],
        model_id="Qwen/Qwen3.8-27B-FP8",
        revision="revision",
        metrics=metrics,
    )
    assert not _has_matching_native_fp8_text_promotion(
        [{"result": "transient-failure", "evidence": evidence}],
        model_id="Qwen/Qwen3.8-27B-FP8",
        revision="revision",
        metrics=metrics,
    )
