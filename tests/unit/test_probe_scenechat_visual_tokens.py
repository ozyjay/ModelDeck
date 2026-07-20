from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_probe_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "probe_scenechat_visual_tokens.py"
    spec = importlib.util.spec_from_file_location("scenechat_visual_token_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe_module()


def test_temperature_helpers_report_hottest_readings() -> None:
    readings = [
        {"source": "amdgpu", "celsius": 72.5},
        {"source": "k10temp", "celsius": 78.0},
    ]

    assert probe._maximum_temperature(readings) == 78.0
    assert probe._gpu_temperature(readings) == 72.5


def test_safe_probe_result_never_retains_generated_description() -> None:
    guard = probe.ThermalGuard(80)
    sample = {
        "analysis": {"summary": "PRIVATE generated description"},
        "latency_seconds": 12.34567,
        "completion_tokens": 123,
        "schema_valid": True,
    }
    sample.pop("analysis")

    result = probe._safe_result(70, sample, None, {"error_code": None}, guard)

    assert result["outcome"] == "valid_response"
    assert result["latency_seconds"] == 12.3457
    assert "PRIVATE" not in str(result)
