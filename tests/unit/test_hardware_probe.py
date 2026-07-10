from __future__ import annotations

from modeldeck.hardware import probe_environment
from modeldeck.hardware.probe import _safe_process_command


def test_probe_runs_without_torch_or_rocm_access() -> None:
    result = probe_environment()
    assert result["configured"]["profile_id"] == "framework-desktop-rocm72"
    assert result["configured"]["work_mount"] == "/mnt/work"
    assert result["detected"]["python"]
    assert result["detected"]["torch"]["allocation_test"] in {"not-run", "not-requested"}
    assert "does not imply an NVIDIA GPU" in result["diagnostic_note"]


def test_active_process_diagnostics_redact_command_line_secrets() -> None:
    command = _safe_process_command(["python", "-m", "vllm", "--api-key", "secret", "--hf-token=also-secret"])
    assert "secret" not in command
    assert command.count("[redacted]") == 2
