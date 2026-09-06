from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from modeldeck.config import Settings
from modeldeck.gateway import create_gateway_app
from modeldeck.main import create_app
from modeldeck.supervisor.service import WorkerSupervisor, build_worker_launch, redact_log

from tests.model_profiles import default_model_profiles


@pytest.mark.parametrize("existing", [False, True])
def test_fresh_imports_and_factories_preserve_operational_files(tmp_path, existing):
    if existing:
        for name in (
            "modeldeck.sqlite3",
            "thermal-status.json",
            "thermal-workloads.json",
            "logs/worker.jsonl",
        ):
            path = tmp_path / "data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing state must be untouched")
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    environment = {
        **os.environ,
        "MODELDECK_DATA_DIR": str(tmp_path / "data"),
        "MODELDECK_LOG_DIR": str(tmp_path / "data/logs"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "backend"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from modeldeck.main import create_app; "
            "from modeldeck.gateway import create_gateway_app; create_app(); create_gateway_app()",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        timeout=20,
    )
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_unrelated_secrets_and_python_injection_do_not_reach_worker(monkeypatch, tmp_path):
    for key in ("HF_TOKEN", "OPENAI_API_KEY", "MODELDECK_SCENECHAT_API_KEY", "PYTHONPATH", "LD_PRELOAD"):
        monkeypatch.setenv(key, "private-test-value")
    profile = next(p for p in default_model_profiles() if p.id == "mock-ar")
    launch = build_worker_launch(profile, data_dir=tmp_path)
    assert "private-test-value" not in launch.environment.values()
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "message",
    [
        '  {"nested":[{"apiKey":"private-test-value"}]}',
        '{"nested":{"output_text":"private-test-value"}}',
        "request failed Bearer private-test-value",
        "output=private-test-value",
        '{broken json "password":"private-test-value"',
    ],
)
def test_sensitive_diagnostics_are_redacted(message):
    assert "private-test-value" not in redact_log(message)


def test_log_input_is_bounded_and_deep_json_fails_closed():
    assert len(redact_log("x" * 100_000)) <= 8192
    message = "[" * 2000 + '"private-test-value"' + "]" * 2000
    assert "private-test-value" not in redact_log(message)


def test_persisted_logs_are_only_loaded_and_redacted_at_explicit_start(tmp_path):
    profile = next(p for p in default_model_profiles() if p.id == "mock-ar")
    path = tmp_path / f"{profile.id}.jsonl"
    original = '{"timestamp":"now","source":"stderr","message":"Bearer private-test-value"}\n'
    path.write_text(original)
    supervisor = WorkerSupervisor([profile], log_dir=tmp_path)
    assert path.read_text() == original
    supervisor.start_log_service()
    assert "private-test-value" not in path.read_text()
    assert "private-test-value" not in str(supervisor.logs(profile.id))


@pytest.mark.parametrize("factory", [create_app, create_gateway_app])
async def test_lifespan_initialises_database(factory, tmp_path):
    settings = Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs")
    app = factory(settings=settings)
    assert not settings.data_dir.exists()
    async with app.router.lifespan_context(app):
        assert (settings.data_dir / "modeldeck.sqlite3").exists()
