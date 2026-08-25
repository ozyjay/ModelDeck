from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_tuner():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmarks/tune_qwen38_fp8.py"
    spec = importlib.util.spec_from_file_location("modeldeck_tune_qwen38_fp8", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tuner = _load_tuner()


def test_candidate_timeout_is_recorded_without_aborting_tuning(monkeypatch, tmp_path) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=12)

    monkeypatch.setattr(tuner.subprocess, "run", timeout)
    arguments = SimpleNamespace(
        cache_root=tmp_path / "cache",
        data_dir=tmp_path / "data",
        candidate_timeout_seconds=90.0,
        secondary_timeout_seconds=12.0,
        warmups=2,
        repetitions=5,
    )

    result = tuner._run_candidate(
        arguments,
        {
            "m": 1,
            "n": 1024,
            "k": 5120,
            "block_size_m": 16,
            "dtype": "bfloat16",
            "num_warps": 2,
            "num_stages": 4,
        },
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "timeout"


def test_known_bad_candidates_are_reused_only_for_the_exact_fingerprint_and_shape(tmp_path) -> None:
    path = tmp_path / "tuning.json"
    fingerprint = {"gpu_architecture": "gfx1151", "torch_version": "reviewed"}
    rejected = {
        "m": 1,
        "n": 1024,
        "k": 5120,
        "block_size_m": 16,
        "dtype": "bfloat16",
        "num_warps": 2,
        "num_stages": 4,
        "status": "rejected",
        "reason": "timeout",
    }
    path.write_text(
        json.dumps(
            {
                "format": "modeldeck-fp8-tuning",
                "version": 1,
                "stage": "full",
                "fingerprint": fingerprint,
                "rejected": [rejected],
            }
        ),
        encoding="utf-8",
    )

    assert tuner._load_known_bad(path, fingerprint=fingerprint, stage="full") == {
        (1024, 5120, 16, "bfloat16", 2, 4)
    }
    assert (
        tuner._load_known_bad(
            path,
            fingerprint={**fingerprint, "torch_version": "changed"},
            stage="full",
        )
        == set()
    )
