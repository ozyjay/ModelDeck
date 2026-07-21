from __future__ import annotations

from uuid import UUID, uuid4

import httpx
import modeldeck.main as main_module
import modeldeck.v2_api as v2_api
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.main import create_app


def worker_definition(*, name: str = "Qwen trace") -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name=name,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="mock",
        runtime_template_id="mock-autoregressive",
        runtime_template_version="2",
        lifecycle="on-demand",
        port=8630,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


def event_document(worker_id: str) -> dict:
    route_id = str(uuid4())
    return {
        "id": str(uuid4()),
        "name": "2026 Open Day",
        "description": "Token Trails",
        "qualification": "compatible",
        "routes": [
            {
                "id": route_id,
                "display_name": "Token trace",
                "public_name": "qwen-0-5b",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [worker_id],
            }
        ],
        "demos": [{"id": str(uuid4()), "name": "Token Trails", "route_ids": [route_id]}],
    }


@pytest.mark.asyncio
async def test_management_starts_empty_without_packaged_workers_or_routes(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = await client.get("/api/health")
            workers = await client.get("/api/workers")
            events = await client.get("/api/events")
            live = await client.get("/api/live")

    assert health.json()["schema_version"] == 2
    assert workers.json() == []
    assert events.json() == {"events": []}
    assert live.json() == {"active_event": None, "routes": []}


@pytest.mark.asyncio
async def test_management_creates_distinct_scenechat_mock_workers_on_demand(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post("/api/workers/mock-scenechat", json={"visual_token_budget": 70})
            second = await client.post("/api/workers/mock-scenechat", json={"visual_token_budget": 70})
            workers = await client.get("/api/workers")
            started = await client.post(f"/api/workers/{first.json()['id']}/start")
            smoked = await client.post(f"/api/workers/{first.json()['id']}/smoke")
            stopped = await client.post(f"/api/workers/{first.json()['id']}/stop")

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert first.json()["name"] == "SceneChat mock 70"
    assert second.json()["name"] == "SceneChat mock 70 (2)"
    assert first.json()["id"] != second.json()["id"]
    assert first.json()["model_id"] == "modeldeck/mock-scenechat-vision"
    assert first.json()["generation_family"] == "vision-language"
    assert first.json()["runtime"] == "mock"
    assert first.json()["capabilities"]["image_input"] is True
    assert first.json()["capabilities"]["structured_output"] is True
    assert first.json()["settings"]["visual_token_budget"] == 70
    assert len(workers.json()) == 2
    assert started.json()["state"] == "ready"
    assert smoked.json()["ok"] is True
    assert stopped.json()["state"] == "stopped"

    restarted = create_app(settings)
    assert len(restarted.state.worker_definitions) == 2


@pytest.mark.asyncio
async def test_management_lists_and_creates_contract_driven_mock_workers(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            templates = await client.get("/api/mock-worker-templates")
            delayed = await client.post(
                "/api/workers/mocks",
                json={
                    "protocol_contract": "openai-completions-v1",
                    "scenario": "delayed",
                    "delay_ms": 25,
                },
            )
            invalid_option = await client.post(
                "/api/workers/mocks",
                json={"protocol_contract": "openai-chat-v1", "visual_token_budget": 70},
            )
            missing_delay = await client.post(
                "/api/workers/mocks",
                json={"protocol_contract": "openai-chat-v1", "scenario": "delayed"},
            )
            arbitrary_setting = await client.post(
                "/api/workers/mocks",
                json={"protocol_contract": "openai-chat-v1", "command": "anything"},
            )

    assert len(templates.json()["templates"]) == 6
    assert delayed.status_code == 201, delayed.text
    assert delayed.json()["capabilities"]["completions"] is True
    assert delayed.json()["settings"] == {
        "mock_contract_id": "openai-completions-v1",
        "mock_scenario": "delayed",
        "mock_delay_ms": 25,
    }
    assert invalid_option.status_code == 422
    assert missing_delay.status_code == 422
    assert arbitrary_setting.status_code == 422


@pytest.mark.asyncio
async def test_worker_name_is_editable_and_persists_without_changing_identity(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            renamed = await client.patch(f"/api/workers/{worker.id}", json={"name": "Visitor Qwen"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Visitor Qwen"
    assert renamed.json()["id"] == worker.id
    assert renamed.json()["model_id"] == worker.model_id

    restarted = create_app(settings)
    assert restarted.state.worker_definitions[worker.id].name == "Visitor Qwen"


@pytest.mark.asyncio
async def test_model_creates_worker_from_trusted_runtime_without_public_alias(tmp_path, monkeypatch) -> None:
    cached = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "revision-1",
        "download_state": "installed-untested",
        "configuration_support": "autoregressive-transformers",
        "cache_location": str(tmp_path / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct"),
        "snapshot_location": str(tmp_path / "snapshot"),
        "base_model_id": None,
        "base_model_revision": None,
        "artifacts": [],
    }
    monkeypatch.setattr(v2_api, "discover_huggingface_models", lambda: [cached])
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [cached])
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            catalogue = await client.get("/api/catalogue")
            created = await client.post(
                "/api/workers",
                json={
                    "name": "Small Qwen",
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "runtime_template_id": "autoregressive-transformers",
                },
            )
    assert catalogue.json()["models"][0]["runnable"] is True
    assert created.status_code == 201, created.text
    payload = created.json()
    UUID(payload["id"])
    assert payload["name"] == "Small Qwen"
    assert "alias" not in payload


@pytest.mark.asyncio
async def test_worker_replacement_preserves_identity_and_only_rebinds_drafts(tmp_path, monkeypatch) -> None:
    cached = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "revision-1",
        "download_state": "installed-untested",
        "configuration_support": "autoregressive-transformers",
        "cache_location": str(tmp_path / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct"),
        "snapshot_location": str(tmp_path / "snapshot"),
        "base_model_id": None,
        "base_model_revision": None,
        "artifacts": [],
    }
    monkeypatch.setattr(v2_api, "discover_huggingface_models", lambda: [cached])
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [cached])
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            created = await client.post(
                "/api/workers",
                json={
                    "name": "Original Qwen",
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "runtime_template_id": "autoregressive-transformers",
                },
            )
            original = created.json()
            event = event_document(original["id"])
            assert (await client.post("/api/events", json=event)).status_code == 201
            assert (await client.post(f"/api/events/{event['id']}/publish")).status_code == 201

            replaced = await client.post(
                f"/api/workers/{original['id']}/replacement",
                json={
                    "name": "Revised Qwen",
                    "dtype": "bfloat16",
                    "lifecycle": "resident",
                    "context_length": 4096,
                    "maximum_new_tokens": 256,
                    "rebind_drafts": True,
                },
            )
            rejected_identity_change = await client.post(
                f"/api/workers/{original['id']}/replacement",
                json={"name": "Different Model", "model_id": "Other/Model"},
            )
            draft = (await client.get("/api/events")).json()["events"][0]
            published = (await client.get(f"/api/events/{event['id']}/revisions")).json()["revisions"][0]

    assert replaced.status_code == 201, replaced.text
    payload = replaced.json()
    replacement = payload["replacement"]
    assert replacement["model_id"] == original["model_id"]
    assert replacement["revision"] == original["revision"]
    assert replacement["runtime_template_id"] == original["runtime_template_id"]
    assert replacement["dtype"] == "bfloat16"
    assert replacement["lifecycle"] == "resident"
    assert replacement["settings"]["context_length"] == 4096
    assert replacement["settings"]["maximum_new_tokens"] == 256
    assert payload["rebound_event_drafts"] == [event["id"]]
    assert draft["definition"]["routes"][0]["worker_ids"] == [replacement["id"]]
    assert published["definition"]["routes"][0]["worker_ids"] == [original["id"]]
    assert rejected_identity_change.status_code == 422


@pytest.mark.asyncio
async def test_scenechat_worker_uses_trusted_runtime_creation_defaults(tmp_path, monkeypatch) -> None:
    cached = {
        "model_id": "google/gemma-4-E2B-it",
        "revision": "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf",
        "download_state": "installed-untested",
        "configuration_support": "scenechat-gemma4",
        "cache_location": str(tmp_path / "hub" / "models--google--gemma-4-E2B-it"),
        "snapshot_location": str(tmp_path / "snapshot"),
        "base_model_id": None,
        "base_model_revision": None,
        "artifacts": [],
    }
    monkeypatch.setattr(v2_api, "discover_huggingface_models", lambda: [cached])
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [cached])
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            templates = await client.get("/api/runtime-templates")
            created = await client.post(
                "/api/workers",
                json={
                    "name": "SceneChat Gemma 4 E2B",
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "runtime_template_id": "scenechat-gemma4",
                },
            )
            created_140 = await client.post(
                "/api/workers",
                json={
                    "name": "SceneChat Gemma 4 E2B 140",
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "runtime_template_id": "scenechat-gemma4",
                    "visual_token_budget": 140,
                },
            )
            rejected_budget = await client.post(
                "/api/workers",
                json={
                    "name": "SceneChat invalid budget",
                    "model_id": cached["model_id"],
                    "revision": cached["revision"],
                    "runtime_template_id": "scenechat-gemma4",
                    "visual_token_budget": 141,
                },
            )

    template = next(item for item in templates.json()["templates"] if item["id"] == "scenechat-gemma4")
    assert template["dtype"] == "bfloat16"
    assert template["settings"]["context_length"] == 8192
    assert template["settings"]["maximum_new_tokens"] == 512
    assert template["settings"]["visual_token_budget"] == 280
    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["dtype"] == "bfloat16"
    assert payload["settings"]["context_length"] == 8192
    assert payload["settings"]["maximum_new_tokens"] == 512
    assert payload["settings"]["visual_token_budget"] == 280
    assert created_140.status_code == 201, created_140.text
    assert created_140.json()["settings"]["visual_token_budget"] == 140
    assert rejected_budget.status_code == 422


@pytest.mark.asyncio
async def test_event_draft_publish_discard_and_reactivate_are_separate(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    event = event_document(worker.id)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/api/events", json=event)).status_code == 201
            first = await client.post(f"/api/events/{event['id']}/publish")
            assert first.status_code == 201
            changed = {**event, "description": "Changed draft only"}
            assert (await client.put(f"/api/events/{event['id']}/draft", json=changed)).status_code == 200
            live = (await client.get("/api/live")).json()
            assert live["active_event"]["revision"] == 1
            discarded = await client.delete(f"/api/events/{event['id']}/draft")
            assert discarded.json()["definition"]["description"] == event["description"]
            changed["description"] = "Second published revision"
            await client.put(f"/api/events/{event['id']}/draft", json=changed)
            second = await client.post(f"/api/events/{event['id']}/publish")
            assert second.json()["revision"] == 2
            restored = await client.post(f"/api/events/{event['id']}/revisions/1/publish")
            assert restored.json()["revision"] == 1


@pytest.mark.asyncio
async def test_replacement_rebinds_drafts_but_not_published_revisions(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    old = worker_definition()
    new = worker_definition(name="Replacement")
    store.save_worker_definition(old.model_dump(mode="json"))
    store.save_worker_definition(new.model_dump(mode="json"))
    event = event_document(old.id)
    store.save_event_draft(event)
    store.publish_event(
        event,
        {
            "format": "modeldeck-event-routing",
            "version": 2,
            "event_id": event["id"],
            "event_name": event["name"],
            "revision": 0,
            "routes": [],
        },
    )

    changed = store.rebind_event_drafts(old.id, new.id)

    assert changed == [event["id"]]
    assert store.get_event(event["id"])["definition"]["routes"][0]["worker_ids"] == [new.id]
    assert store.get_event_revision(event["id"], 1)["definition"]["routes"][0]["worker_ids"] == [old.id]


@pytest.mark.asyncio
async def test_open_day_mode_locks_configuration_but_not_reads(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs", open_day=True))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/events")).status_code == 200
            blocked = await client.post(
                "/api/events",
                json={
                    "id": str(uuid4()),
                    "name": "Locked",
                    "description": "",
                    "qualification": "compatible",
                    "demos": [],
                    "routes": [],
                },
            )
            blocked_mock = await client.post(
                "/api/workers/mocks",
                json={"protocol_contract": "openai-chat-v1"},
            )
    assert blocked.status_code == 423
    assert blocked_mock.status_code == 423


@pytest.mark.asyncio
async def test_unknown_worker_is_not_interpreted_as_a_command(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/workers/not-a-command/start")
    assert response.status_code == 404
