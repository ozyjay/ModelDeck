from __future__ import annotations

import sys

from modeldeck.profiles import default_model_profiles
from modeldeck.supervisor.service import build_mock_worker_command, redact_log


def test_worker_command_is_an_argument_array_with_allowlisted_values() -> None:
    profile = default_model_profiles()[0]
    command = build_mock_worker_command(profile)
    assert command[:3] == [sys.executable, "-m", "modeldeck.workers.mock_worker"]
    assert command[-2:] == ["--port", "8610"]
    assert all(";" not in argument for argument in command)


def test_log_redaction_removes_prompt_and_credentials() -> None:
    assert redact_log("prompt=private visitor words") == "prompt=[redacted]"
    assert "secret" not in redact_log('{"api_key":"secret","status":"failed"}')
