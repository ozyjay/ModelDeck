from __future__ import annotations

import httpx
import pytest
from modeldeck.contracts.scenechat import SceneAnalysis
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
    assert trace["prompt_tokens"] == ["Welcome"]
    assert len(trace["prompt_token_ids"]) == len(trace["prompt_tokens"])
    assert trace["user_prompt_tokens"] == ["Welcome"]
    assert len(trace["user_prompt_token_ids"]) == len(trace["user_prompt_tokens"])
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
    assert first["frames"][-1]["finish_reason"] == "stop"
    assert first["frames"][-1]["seed"] == 11
    assert wrong_route.status_code == 404
    assert cancelled.json()["state"] == "cancelled"
    assert "event: cancelled" in events.text


@pytest.mark.asyncio
async def test_scenechat_contract_returns_labelled_deterministic_structured_output() -> None:
    app = create_app(
        worker_id="test-scenechat",
        model_id="modeldeck/mock-scenechat-vision",
        revision="fixture-v1",
        family=GenerationFamily.VISION_LANGUAGE,
        startup_delay=0,
    )
    request = {
        "model": "scenechat-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "What is visible?"},
                ],
            }
        ],
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            capability_response = await client.get("/capabilities")
            first = await client.post("/v1/chat/completions", json=request)
            second = await client.post("/v1/chat/completions", json=request)
            smoke = await client.post("/native/vision-language/smoke")

    capabilities = capability_response.json()
    assert capabilities["generation_family"] == "vision-language"
    assert capabilities["image_input"] is True
    assert capabilities["structured_output"] is True
    assert capabilities["streaming"] is False
    assert first.status_code == 200
    assert (
        first.json()["choices"][0]["message"]["content"] == second.json()["choices"][0]["message"]["content"]
    )
    analysis = SceneAnalysis.model_validate_json(first.json()["choices"][0]["message"]["content"])
    assert analysis.summary.startswith("Mock SceneChat")
    assert "not physical model inference" in analysis.safety_notes[0]
    assert smoke.json() == {
        "ok": True,
        "model_id": "modeldeck/mock-scenechat-vision",
        "mock": True,
        "visual_contract": "scene-analysis-v1",
    }
