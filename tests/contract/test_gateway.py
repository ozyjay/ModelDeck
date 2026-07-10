from __future__ import annotations

import httpx
import pytest
from modeldeck.gateway import create_gateway_app


@pytest.mark.asyncio
async def test_gateway_returns_structured_local_unavailable_without_cloud() -> None:
    app = create_gateway_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/completions", json={"model": "fast-chat", "prompt": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "local_provider_unavailable"
    assert response.json()["error"]["cloud_fallback_attempted"] is False
