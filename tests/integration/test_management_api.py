from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import RoutingProfile, WorkerDefinition, routing_snapshot
from modeldeck.main import create_app


def worker_definition(*, name: str = "Qwen trace", port: int = 8630) -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name=name,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="transformers-rocm",
        runtime_template_id="autoregressive-transformers",
        runtime_template_version="2",
        lifecycle="on-demand",
        port=port,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


def profile_document(worker_id: str, *, name: str = "Local applications") -> dict:
    return {
        "id": str(uuid4()),
        "name": name,
        "description": "Token Trail and SprintBot",
        "qualification": "compatible",
        "capabilities": [
            {
                "id": str(uuid4()),
                "display_name": "Token trace",
                "public_name": "qwen-0-5b",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [worker_id],
            }
        ],
    }


@pytest.mark.asyncio
async def test_management_starts_empty_with_routing_profiles(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/api/health")
            workers = await client.get("/api/workers")
            profiles = await client.get("/api/routing-profiles")
            live = await client.get("/api/live")

    assert health.json()["schema_version"] == 3
    assert workers.json() == []
    assert profiles.json() == {"profiles": []}
    assert live.json() == {"active_profile": None, "capabilities": []}


@pytest.mark.asyncio
async def test_management_has_no_public_event_or_mock_worker_api(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            event_api = await client.get("/api/events")
            templates = await client.get("/api/mock-worker-templates")
            create_mock = await client.post(
                "/api/workers/mocks", json={"protocol_contract": "openai-chat-v1"}
            )

    assert event_api.status_code == 404
    assert templates.status_code == 404
    assert create_mock.status_code in {404, 405}


@pytest.mark.asyncio
async def test_profile_publish_rollback_and_live_capabilities(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    profile = profile_document(worker.id)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post("/api/routing-profiles", json=profile)
            validation = await client.post(f"/api/routing-profiles/{profile['id']}/validate")
            first = await client.post(f"/api/routing-profiles/{profile['id']}/publish")
            changed = {**profile, "description": "Updated profile only"}
            assert (
                await client.put(f"/api/routing-profiles/{profile['id']}/draft", json=changed)
            ).status_code == 200
            second = await client.post(f"/api/routing-profiles/{profile['id']}/publish")
            rollback = await client.post(f"/api/routing-profiles/{profile['id']}/revisions/1/publish")
            live = await client.get("/api/live")
            revisions = await client.get(f"/api/routing-profiles/{profile['id']}/revisions")

    assert created.status_code == 201
    assert validation.json()["valid"] is True
    assert first.json()["revision"] == 1
    assert second.json()["revision"] == 2
    assert rollback.json()["revision"] == 1
    assert live.json()["active_profile"] == {"id": profile["id"], "name": profile["name"], "revision": 1}
    assert live.json()["capabilities"][0]["public_name"] == "qwen-0-5b"
    assert [item["revision"] for item in revisions.json()["revisions"]] == [2, 1]
    assert revisions.json()["revisions"][1]["active"] is True


@pytest.mark.asyncio
async def test_profile_publish_rejects_incompatible_worker(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    profile = profile_document(worker.id)
    profile["capabilities"][0]["protocol_contract"] = "text-diffusion-v1"
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/api/routing-profiles", json=profile)).status_code == 201
            response = await client.post(f"/api/routing-profiles/{profile['id']}/publish")

    assert response.status_code == 409
    assert response.json()["detail"]["validation"]["valid"] is False


def test_replacement_rebinds_profile_drafts_but_not_published_revisions(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    old = worker_definition()
    new = worker_definition(name="Replacement", port=8631)
    store.save_worker_definition(old.model_dump(mode="json"))
    store.save_worker_definition(new.model_dump(mode="json"))
    profile = RoutingProfile.model_validate(profile_document(old.id))
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))

    changed = store.rebind_routing_profile_drafts(old.id, new.id)

    assert changed == [profile.id]
    assert store.get_routing_profile(profile.id)["definition"]["capabilities"][0]["worker_ids"] == [new.id]
    assert store.get_routing_profile_revision(profile.id, 1)["definition"]["capabilities"][0][
        "worker_ids"
    ] == [old.id]


@pytest.mark.asyncio
async def test_open_day_mode_locks_profile_mutation_but_keeps_reads_available(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs", open_day=True))
    profile = profile_document(str(uuid4()), name="Locked")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            readable = await client.get("/api/routing-profiles")
            blocked = await client.post("/api/routing-profiles", json=profile)

    assert readable.status_code == 200
    assert blocked.status_code == 423
