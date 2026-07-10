from __future__ import annotations

import httpx
import pytest
from modeldeck.protocol import GenerationFamily
from modeldeck.workers.mock_worker import create_app


@pytest.mark.asyncio
async def test_autoregressive_contract_includes_top_k_trace() -> None:
    app = create_app(
        worker_id="test-ar",
        model_id="modeldeck/test-ar",
        revision="fixture",
        family=GenerationFamily.AUTOREGRESSIVE,
        startup_delay=0,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = (await client.get("/health")).json()
            trace = (
                await client.post(
                    "/native/autoregressive/trace",
                    json={"prompt": "Welcome", "top_k": 3, "seed": 2, "max_tokens": 4},
                )
            ).json()
            await client.post("/cancel", json={"request_id": "cancel-me"})
            cancelled_stream = await client.post(
                "/v1/chat/completions",
                json={"request_id": "cancel-me", "prompt": "private", "stream": True},
            )
    assert health["protocol_version"] == "1"
    assert health["ready"] is True
    assert trace["events"][0]["selected"]["token"]
    assert len(trace["events"][0]["alternatives"]) == 2
    assert "text_so_far" in trace["events"][0]
    assert "event: cancelled" in cancelled_stream.text
    assert "private" not in cancelled_stream.text


@pytest.mark.asyncio
async def test_text_diffusion_contract_is_seeded_and_has_frames() -> None:
    app = create_app(
        worker_id="test-diffusion",
        model_id="modeldeck/test-diffusion",
        revision="fixture",
        family=GenerationFamily.TEXT_DIFFUSION,
        startup_delay=0,
    )
    request = {"prompt": "A robot arrives at university orientation.", "denoising_steps": 4, "seed": 11}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = (await client.post("/v1/refine", json=request)).json()
            second = (await client.post("/v1/refine", json=request)).json()
            wrong_route = await client.post("/v1/chat/completions", json={"prompt": "wrong engine"})
            job = (await client.post("/v1/diffuse", json=request)).json()
            cancelled = await client.post(f"/v1/jobs/{job['job_id']}/cancel")
            events = await client.get(f"/v1/jobs/{job['job_id']}/events")
    assert first == second
    assert first["frames"][0]["complete"] is False
    assert first["frames"][-1]["complete"] is True
    assert first["frames"][-1]["seed"] == 11
    assert wrong_route.status_code == 404
    assert cancelled.json()["state"] == "cancelled"
    assert "event: cancelled" in events.text
