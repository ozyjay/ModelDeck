"""Test-harness-only runtimes and fixture workers."""

import os

import pytest

os.environ.setdefault("MODELDECK_TEST_HARNESS", "1")


@pytest.fixture(autouse=True)
def isolated_operational_paths(monkeypatch, tmp_path):
    """Default app and launch-builder state must never touch an operator's files."""
    monkeypatch.setenv("MODELDECK_DATA_DIR", str(tmp_path / "default-data"))
    monkeypatch.setenv("MODELDECK_LOG_DIR", str(tmp_path / "default-logs"))
