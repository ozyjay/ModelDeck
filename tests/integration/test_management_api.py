from __future__ import annotations

from pathlib import Path

import httpx
import modeldeck.main as main_module
import pytest
from modeldeck.config import Settings
from modeldeck.main import create_app


def test_management_defaults_to_loopback_only(monkeypatch) -> None:
    monkeypatch.delenv("MODELDECK_HOST", raising=False)

    assert Settings.from_env().host == "127.0.0.1"


@pytest.mark.asyncio
async def test_management_api_is_gpu_free_and_does_not_start_workers(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
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
        "vision-language",
    }
    assert (tmp_path / "modeldeck.sqlite3").exists()


@pytest.mark.asyncio
async def test_unknown_worker_is_not_interpreted_as_a_command(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/workers/echo-danger/start")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cached_autoregressive_runtime_configuration_is_persistent_and_removable(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "cache" / "models--Example--LocalModel"
    model_dir.mkdir(parents=True)
    cached = {
        "model_id": "Example/LocalModel",
        "revision": "revision-1",
        "cache_location": str(model_dir),
        "physical_size_bytes": 1024,
        "download_state": "installed-untested",
        "generation_family_hint": "autoregressive",
        "configuration_support": "autoregressive-transformers",
        "configuration_support_reason": "Supported by the local Transformers ROCm worker.",
        "runnable": False,
        "runnable_reason": "Compatibility has not been tested for the current stack.",
    }
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [cached])
    settings = Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs")

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/api/profiles",
                json={
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "alias": "local-example",
                    "dtype": "bfloat16",
                    "lifecycle": "on-demand",
                    "context_length": 4096,
                    "maximum_new_tokens": 96,
                },
            )
            workers = await client.get("/api/workers")

    assert created.status_code == 201
    assert created.json()["source"] == "local"
    assert created.json()["settings"]["cache_root"] == str(model_dir.parent)
    assert created.json()["trust_remote_code"] is False
    assert any(worker["id"] == "local-local-example" for worker in workers.json())

    restored = create_app(settings)
    async with restored.router.lifespan_context(restored):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restored), base_url="http://test"
        ) as client:
            profiles = await client.get("/api/profiles")
            removed = await client.delete("/api/profiles/local-local-example")

    assert any(profile["source"] == "local" for profile in profiles.json())
    assert removed.json() == {
        "ok": True,
        "profile_id": "local-local-example",
        "cache_removed": False,
    }


@pytest.mark.asyncio
async def test_runtime_configuration_rejects_unrecognised_and_reserved_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    cached = {
        "model_id": "Example/LocalModel",
        "revision": "revision-1",
        "cache_location": str(tmp_path / "cache" / "models--Example--LocalModel"),
        "physical_size_bytes": 1024,
        "download_state": "installed-untested",
        "generation_family_hint": "autoregressive",
        "configuration_support": "autoregressive-transformers",
        "configuration_support_reason": "Supported by the local Transformers ROCm worker.",
        "runnable": False,
        "runnable_reason": "Untested",
    }
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [cached])
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            reserved = await client.post(
                "/api/profiles",
                json={
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "alias": "fast-chat",
                },
            )
            arbitrary = await client.post(
                "/api/profiles",
                json={
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "alias": "safe-alias",
                    "cache_root": "/tmp/not-allowed",
                },
            )

    assert reserved.status_code == 409
    assert arbitrary.status_code == 422


@pytest.mark.asyncio
async def test_management_configures_allowlisted_vision_and_diffusion_workers(
    monkeypatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "cache" / "model"
    model_dir.mkdir(parents=True)
    common = {
        "revision": "revision-1",
        "cache_location": str(model_dir),
        "physical_size_bytes": 1024,
        "download_state": "installed-untested",
        "runnable": False,
        "runnable_reason": "Untested",
    }
    catalogue = [
        {
            **common,
            "model_id": "google/gemma-4-local",
            "generation_family_hint": "vision-language",
            "configuration_support": "scenechat-gemma4",
            "configuration_support_reason": "Supported by SceneChat.",
        },
        {
            **common,
            "model_id": "google/diffusiongemma-local",
            "generation_family_hint": "text-diffusion",
            "configuration_support": "diffusiongemma-transformers",
            "configuration_support_reason": "Supported by DiffusionGemma.",
        },
    ]
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: catalogue)
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            vision = await client.post(
                "/api/profiles",
                json={
                    "model_id": "google/gemma-4-local",
                    "revision": "revision-1",
                    "alias": "local-vision",
                    "dtype": "bfloat16",
                    "context_length": 8192,
                    "maximum_new_tokens": 512,
                },
            )
            diffusion = await client.post(
                "/api/profiles",
                json={
                    "model_id": "google/diffusiongemma-local",
                    "revision": "revision-1",
                    "alias": "local-diffusion",
                    "dtype": "bfloat16",
                    "lifecycle": "exclusive",
                    "maximum_new_tokens": 256,
                    "maximum_denoising_steps": 24,
                },
            )

    assert vision.status_code == 201
    assert vision.json()["preferred_runtime"] == "vision-language-transformers-rocm"
    assert diffusion.status_code == 201
    assert diffusion.json()["preferred_runtime"] == "text-diffusion-transformers-rocm"
    assert diffusion.json()["lifecycle"] == "exclusive"


@pytest.mark.asyncio
async def test_gateway_status_is_same_origin_and_structured(monkeypatch, tmp_path: Path) -> None:
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        payloads = {
            "/v1/health": {"status": "ok", "ready_providers": 1},
            "/v1/models": {"data": [{"id": "fast-chat", "ready": True}]},
            "/v1/providers": {"providers": [{"id": "qwen-small-rocm", "ready": True}]},
        }
        return httpx.Response(200, json=payloads[request.url.path])

    monkeypatch.setattr(
        main_module.httpx,
        "AsyncClient",
        lambda **_kwargs: original_client(transport=httpx.MockTransport(handler)),
    )
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with original_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/gateway/status")

    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["health"]["ready_providers"] == 1
    assert response.json()["providers"]["providers"][0]["id"] == "qwen-small-rocm"


@pytest.mark.asyncio
async def test_gateway_status_reports_local_unavailable_without_exception(monkeypatch) -> None:
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"}, request=request)

    monkeypatch.setattr(
        main_module.httpx,
        "AsyncClient",
        lambda **_kwargs: original_client(transport=httpx.MockTransport(handler)),
    )

    result = await main_module._gateway_status(Settings())

    assert result == {
        "available": False,
        "health": None,
        "models": None,
        "providers": None,
        "error": "The local ModelDeck gateway is unavailable.",
    }


@pytest.mark.asyncio
async def test_operator_console_assets_and_spa_routes_are_served(monkeypatch, tmp_path: Path) -> None:
    static = tmp_path / "static"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>Operator console</title>", encoding="utf-8")
    (assets / "app.js").write_text("export {};", encoding="utf-8")
    monkeypatch.setattr(main_module, "FRONTEND_ROOT", static)
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            root = await client.get("/")
            workers = await client.get("/workers")
            asset = await client.get("/assets/app.js")
            missing_api = await client.get("/api/not-real")

    assert root.status_code == 200
    assert workers.status_code == 200
    assert "Operator console" in workers.text
    assert asset.text == "export {};"
    assert "default-src 'self'" in root.headers["content-security-policy"]
    assert "default-src 'self'" in workers.headers["content-security-policy"]
    assert missing_api.status_code == 404
