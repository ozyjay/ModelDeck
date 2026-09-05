import json

from modeldeck.benchmark_history import read_benchmark_history


def test_reads_comparable_standard_throughput_history(tmp_path) -> None:
    report = {
        "format": "modeldeck-benchmark",
        "completed_at": "2026-07-21T12:00:00+00:00",
        "configuration": {"preset": "quick", "autoregressive_tokens": 64},
        "results": [
            {
                "worker_id": "worker-1",
                "worker_name": "Small Qwen",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_revision": "revision-1",
                "generation_family": "autoregressive",
                "runtime": "transformers-rocm",
                "dtype": "float16",
                "status": "success",
                "fingerprint": "fingerprint-1",
                "summary": {
                    "successful_requests": 2,
                    "throughput_tokens_per_second": {"median": 64.0},
                },
            }
        ],
    }
    (tmp_path / "modeldeck-benchmark.json").write_text(json.dumps(report), encoding="utf-8")

    history = read_benchmark_history(tmp_path)

    assert history["reports_scanned"] == 1
    assert history["points"] == [
        {
            "series_key": history["points"][0]["series_key"],
            "observed_at": "2026-07-21T12:00:00+00:00",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": "revision-1",
            "runtime": "transformers-rocm",
            "dtype": "float16",
            "generation_family": "autoregressive",
            "worker_id": "worker-1",
            "worker_name": "Small Qwen",
            "tokens_per_second": 64.0,
            "workload": "Standard · quick · 64 output tokens",
            "configuration_fingerprint": "fingerprint-1",
            "sample_count": 2,
        }
    ]


def test_ignores_failed_thermal_and_unknown_reports(tmp_path) -> None:
    (tmp_path / "failed.json").write_text(
        json.dumps(
            {
                "format": "modeldeck-benchmark",
                "completed_at": "2026-07-21T12:00:00+00:00",
                "configuration": {"preset": "quick"},
                "results": [{"status": "thermal-invalid"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "unknown.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")

    assert read_benchmark_history(tmp_path)["points"] == []
