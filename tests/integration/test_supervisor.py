from __future__ import annotations

import asyncio
import socket

import httpx
import pytest
from modeldeck.profiles import ModelProfile
from modeldeck.protocol import WorkerState
from modeldeck.supervisor import WorkerSupervisor

from tests.model_profiles import default_model_profiles


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_profile(port: int) -> ModelProfile:
    document = next(profile for profile in default_model_profiles() if profile.id == "mock-ar").model_dump()
    document.update({"id": "integration-ar", "alias": "integration-ar", "port": port})
    return ModelProfile.model_validate(document)


@pytest.mark.asyncio
async def test_start_health_restart_and_stop_without_process_leak() -> None:
    supervisor = WorkerSupervisor([make_profile(free_port())], startup_timeout=8, stop_timeout=2)
    try:
        started = await supervisor.start("integration-ar")
        assert started["state"] == WorkerState.READY
        first_pid = started["pid"]
        async with httpx.AsyncClient() as client:
            health = (await client.get(f"{started['endpoint']}/health")).json()
            assert health["generation_family"] == "autoregressive"
        assert any(
            "Starting allowlisted ModelDeck mock worker" in item["message"]
            for item in supervisor.logs("integration-ar")
        )

        restarted = await supervisor.restart("integration-ar")
        assert restarted["state"] == WorkerState.READY
        assert restarted["pid"] != first_pid

        stopped = await supervisor.stop("integration-ar")
        assert stopped["state"] == WorkerState.STOPPED
        assert stopped["pid"] is None
        assert [event["state"] for event in supervisor.event_history()].count("ready") == 2
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_port_collision_is_refused() -> None:
    port = free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    supervisor = WorkerSupervisor([make_profile(port)])
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            await supervisor.start("integration-ar")
        assert supervisor.get_worker("integration-ar")["state"] == WorkerState.FAILED
    finally:
        blocker.close()
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_unexpected_process_exit_is_detected() -> None:
    supervisor = WorkerSupervisor([make_profile(free_port())], startup_timeout=8)
    try:
        await supervisor.start("integration-ar")
        managed = supervisor.workers["integration-ar"]
        assert managed.process is not None
        managed.process.kill()
        await managed.process.wait()
        await asyncio.sleep(0.05)
        assert supervisor.get_worker("integration-ar")["state"] == WorkerState.FAILED
    finally:
        await supervisor.stop_all()
