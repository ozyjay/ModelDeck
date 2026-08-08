from __future__ import annotations

import httpx
import pytest
from modeldeck.workers.embedding_worker import EMBEDDING_DIMENSIONS, EmbeddingEngineConfig, create_app


class FixtureEmbeddingEngine:
    runtime_details = {"device": "cuda:0", "embedding_dimensions": EMBEDDING_DIMENSIONS}

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[float(index)] * EMBEDDING_DIMENSIONS for index, _ in enumerate(inputs)]

    def memory_metrics(self) -> dict[str, int]:
        return {}


@pytest.mark.asyncio
async def test_embedding_worker_returns_1024_dimensions_and_preserves_input_order() -> None:
    app = create_app(
        worker_id="embedding-worker",
        config=EmbeddingEngineConfig(model_id="Qwen/Qwen3-Embedding-0.6B", revision="a" * 40),
        engine=FixtureEmbeddingEngine(),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            warmed = await client.post("/warmup")
            response = await client.post(
                "/v1/embeddings",
                json={"model": "sprintbot-embedding", "input": ["first", "second"]},
            )
            chat = await client.post("/v1/chat/completions", json={"model": "sprintbot-embedding"})

    assert warmed.status_code == 200
    assert response.status_code == 200
    assert response.json()["model"] == "sprintbot-embedding"
    assert [item["index"] for item in response.json()["data"]] == [0, 1]
    assert all(len(item["embedding"]) == EMBEDDING_DIMENSIONS for item in response.json()["data"])
    assert chat.status_code == 404
