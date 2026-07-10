from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from modeldeck.config import Settings
from modeldeck.main import create_app


@pytest.mark.asyncio
async def test_management_api_is_gpu_free_and_does_not_start_workers(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/api/health")
            workers = await client.get("/api/workers")
            profiles = await client.get("/api/profiles")
    assert health.status_code == 200
    assert health.json()["downloads_allowed"] is False
    assert all(worker["state"] == "stopped" for worker in workers.json())
    assert {profile["generation_family"] for profile in profiles.json()} == {
        "autoregressive",
        "text-diffusion",
    }
    assert (tmp_path / "modeldeck.sqlite3").exists()


@pytest.mark.asyncio
async def test_unknown_worker_is_not_interpreted_as_a_command(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/workers/echo-danger/start")
    assert response.status_code == 404
