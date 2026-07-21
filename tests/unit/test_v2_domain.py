import sqlite3
from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore, LegacyDatabaseError
from modeldeck.config import Settings
from modeldeck.domain import EventDefinition, WorkerDefinition, routing_snapshot, validate_event
from modeldeck.gateway.app import create_gateway_app
from modeldeck.main import create_app
from modeldeck.v2_api import _worker_smoke_request


def worker_definition() -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name="Qwen trace Worker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="mock",
        runtime_template_id="mock-autoregressive",
        runtime_template_version="2",
        lifecycle="on-demand",
        port=8630,
        dtype="float16",
        capabilities={"chat": True, "top_k_trace": True},
        settings={},
    )


def event_definition(worker_id: str, *, qualification: str = "compatible") -> EventDefinition:
    route_id = str(uuid4())
    return EventDefinition(
        id=str(uuid4()),
        name="2026 Open Day",
        description="Token Trails only",
        qualification=qualification,
        routes=[
            {
                "id": route_id,
                "display_name": "Token trace",
                "public_name": "qwen-0-5b",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [worker_id],
            }
        ],
        demos=[{"id": str(uuid4()), "name": "Token Trails", "route_ids": [route_id]}],
    )


def test_v2_store_starts_empty_and_refuses_legacy_database(tmp_path):
    path = tmp_path / "modeldeck.sqlite3"
    store = CompatibilityStore(path)
    store.initialise_v2()
    assert store.list_workers() == []
    assert store.list_events() == []
    assert store.active_routing_snapshot() is None

    legacy_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy_path) as database:
        database.execute("CREATE TABLE model_profiles (id TEXT PRIMARY KEY)")
    legacy = CompatibilityStore(legacy_path)
    with pytest.raises(LegacyDatabaseError, match="cutover_v2"):
        legacy.initialise_v2()


def test_event_routes_share_workers_and_preserve_explicit_order():
    primary = worker_definition()
    backup = worker_definition().model_copy(
        update={"id": str(uuid4()), "name": "Backup Worker", "port": 8631}
    )
    event = event_definition(primary.id)
    event.routes[0].worker_ids.append(backup.id)

    validation = validate_event(event, [primary, backup], [])

    assert validation["valid"] is True
    assert [worker["role"] for worker in validation["routes"][0]["workers"]] == [
        "primary",
        "backup",
    ]
    assert routing_snapshot(event, 4)["routes"][0]["worker_ids"] == [primary.id, backup.id]


def test_event_duplicate_api_model_ids_name_conflicting_routes():
    worker = worker_definition()
    event = event_definition(worker.id)
    duplicate = event.routes[0].model_copy(update={"id": str(uuid4()), "display_name": "Qwen3.5 vision"})

    with pytest.raises(
        ValueError,
        match=r"API Model IDs must be unique.*Token trace.*qwen-0-5b.*Qwen3.5 vision",
    ):
        EventDefinition.model_validate(
            {
                **event.model_dump(mode="json"),
                "routes": [
                    event.routes[0].model_dump(mode="json"),
                    duplicate.model_dump(mode="json"),
                ],
            }
        )


def test_tested_working_event_requires_matching_evidence():
    worker = worker_definition()
    event = event_definition(worker.id, qualification="tested-working")
    assert validate_event(event, [worker], [])["valid"] is False
    evidence = {
        "result": "tested-working",
        "evidence": {
            "model_id": worker.model_id,
            "model_revision": worker.revision,
            "runtime": worker.runtime,
        },
    }
    assert validate_event(event, [worker], [evidence])["valid"] is True


def test_worker_smoke_requests_generate_for_each_supported_engine():
    autoregressive = worker_definition()
    path, body, headers = _worker_smoke_request(autoregressive)
    assert path == "/native/autoregressive/trace"
    assert body["max_tokens"] == 4
    assert headers is None

    diffusion = autoregressive.model_copy(
        update={
            "generation_family": "text-diffusion",
            "runtime": "text-diffusion-transformers-rocm",
            "capabilities": {"iterative_refinement": True, "intermediate_frames": True},
        }
    )
    path, body, headers = _worker_smoke_request(diffusion)
    assert path == "/v1/refine"
    assert body["denoising_steps"] == 4
    assert headers is None


@pytest.mark.asyncio
async def test_management_and_gateway_use_only_published_v2_event(tmp_path):
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    event = event_definition(worker.id)
    store.save_event_draft(event.model_dump(mode="json"))

    management_app = create_app(settings)
    async with management_app.router.lifespan_context(management_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=management_app), base_url="http://test"
        ) as management:
            assert (await management.get("/api/workers")).json()[0]["name"] == worker.name
            assert (await management.get("/api/live")).json() == {
                "active_event": None,
                "routes": [],
            }
            publish = await management.post(f"/api/events/{event.id}/publish")
            assert publish.status_code == 201
            live = (await management.get("/api/live")).json()
    assert live["active_event"]["name"] == event.name
    assert live["routes"][0]["id"] == event.routes[0].id
    assert live["routes"][0]["public_name"] == "qwen-0-5b"
    assert live["routes"][0]["ready"] is False

    gateway_app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app), base_url="http://test"
    ) as gateway:
        advertised = (await gateway.get("/v1/models")).json()["data"]
    assert advertised == [
        {"id": "qwen-0-5b", "object": "model", "owned_by": "modeldeck-local", "ready": False}
    ]
