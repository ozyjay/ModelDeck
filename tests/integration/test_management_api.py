from __future__ import annotations

from uuid import uuid4

import httpx
import modeldeck.main as main_module
import modeldeck.v2_api as v2_api_module
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


def test_allowlisted_wayfinder_worker_normalises_persisted_cache_capability() -> None:
    worker = worker_definition()

    assert worker.capabilities["prefix_caching"] == "application-managed"
    assert worker.capabilities["prefix_cache_enabled"] is False


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


def discovered_model(*, runtime: str | None = None) -> dict:
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "revision-1",
        "cache_location": "/cache/model",
        "snapshot_location": "/cache/model/snapshots/revision-1",
        "physical_size_bytes": 1,
        "download_state": "installed-untested",
        "generation_family_hint": "autoregressive",
        "capability_hints": ["text-generation", "chat"],
        "configuration_support": runtime,
        "configuration_support_reason": "Local test candidate",
        "base_model_id": None,
        "base_model_revision": None,
        "runnable": False,
        "runnable_reason": "Not tested",
        "artifacts": [],
        "potential_capabilities": [
            {
                "id": "autoregressive-trace",
                "display_name": "Autoregressive trace",
                "description": "Native token traces.",
                "protocol_contract_id": "native-ar-trace-v1",
                "traits": ["text-input", "text-output", "token-trace"],
                "evidence": [
                    {
                        "kind": "detected",
                        "confidence": "direct",
                        "source": "test",
                        "detail": "Exact local test fixture.",
                    }
                ],
                "runtime_template_ids": [runtime] if runtime else [],
            }
        ],
    }


def discovered_qwen35_model() -> dict:
    return {
        **discovered_model(runtime="scenechat-qwen35"),
        "model_id": "Qwen/Qwen3.5-0.8B",
        "generation_family_hint": "vision-language",
        "capability_hints": ["text-generation", "chat", "image-input"],
        "potential_capabilities": [
            {
                "id": "general-chat",
                "display_name": "General chat",
                "description": "Conversational text generation.",
                "protocol_contract_id": "openai-chat-v1",
                "traits": ["text-input", "text-output", "chat"],
                "evidence": [
                    {"kind": "asserted", "confidence": "direct", "source": "test", "detail": "Test."}
                ],
                "runtime_template_ids": [],
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

    assert health.json()["schema_version"] == 4
    assert health.json()["configuration_locked"] is False
    assert health.json()["offline_only"] is True
    assert workers.json() == []
    assert profiles.json() == {"profiles": []}
    assert live.json() == {"active_profile": None, "active_profiles": [], "capabilities": []}


@pytest.mark.asyncio
async def test_worker_check_is_diagnostic_and_does_not_record_compatibility_evidence(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))

    class ProbeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url: str):
            payload = {
                "/health": {"ready": True, "device_name": "Test GPU"},
                "/model": {"model_id": worker.model_id, "revision": worker.revision},
                "/metrics": {"load_seconds": 1.25},
            }[next(path for path in ("/health", "/model", "/metrics") if url.endswith(path))]
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        async def post(self, url: str, *, json=None, headers=None):
            assert url.endswith("/v1/chat/completions")
            assert json["model"] == worker.to_profile().alias
            assert headers is None
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ready"}}]},
                request=httpx.Request("POST", url),
            )

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(v2_api_module.httpx, "AsyncClient", lambda *args, **kwargs: ProbeClient())
    monkeypatch.setattr(
        v2_api_module,
        "probe_environment",
        lambda: {
            "configured": {"profile_id": "test", "gpu_architecture": "test-gpu"},
            "detected": {"fedora_release": "Test Fedora", "kernel": "test-kernel"},
        },
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        monkeypatch.setattr(
            app.state.supervisor,
            "get_worker",
            lambda _worker_id: {"state": "ready", "endpoint": "http://127.0.0.1:8630"},
        )
        monkeypatch.setattr(
            app.state.supervisor,
            "worker_environment",
            lambda _worker_id: {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LD_PRELOAD": None,
            },
        )
        async with real_async_client(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(f"/api/workers/{worker.id}/smoke")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["diagnostic"]["result"] == "diagnostic-passed"
    assert payload["diagnostic"]["evidence"]["environment_overrides"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LD_PRELOAD": None,
    }
    assert store.list_tests() == []


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
async def test_capability_policy_records_missing_runtime_intent_and_master_denial(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [discovered_model()])
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            allowed = await client.post(
                "/api/catalogue/capabilities/policy",
                json={
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "revision-1",
                    "capability_id": "autoregressive-trace",
                    "allowed": True,
                },
            )
            await client.post(
                "/api/catalogue/policy",
                json={
                    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "revision-1",
                    "allowed": False,
                },
            )
            catalogue = (await client.get("/api/catalogue")).json()["models"][0]

    capability = catalogue["potential_capabilities"][0]
    assert allowed.status_code == 200
    assert capability["policy_allowed"] is True
    assert capability["effective_allowed"] is False
    assert capability["runtime_status"] == "missing"
    assert capability["reason"] == "This cached Model is disallowed in ModelDeck."


@pytest.mark.asyncio
async def test_qwen35_chat_capability_creates_only_its_dedicated_worker(tmp_path, monkeypatch) -> None:
    model = discovered_qwen35_model()
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [model])
    monkeypatch.setattr(v2_api_module, "discover_huggingface_models", lambda: [model])
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            allowed = await client.post(
                "/api/catalogue/capabilities/policy",
                json={
                    "model_id": model["model_id"],
                    "revision": model["revision"],
                    "capability_id": "general-chat",
                    "allowed": True,
                },
            )
            catalogue = (await client.get("/api/catalogue")).json()["models"][0]
            created = await client.post(
                "/api/workers",
                json={
                    "name": "Qwen chat",
                    "model_id": model["model_id"],
                    "revision": model["revision"],
                    "capability_id": "general-chat",
                    "runtime_template_id": "qwen35-chat-transformers-rocm",
                },
            )

    assert allowed.status_code == 200
    assert catalogue["potential_capabilities"][0]["available_runtime_template_ids"] == [
        "qwen35-chat-transformers-rocm"
    ]
    assert created.status_code == 201
    worker = created.json()
    assert worker["runtime"] == "qwen35-chat-transformers-rocm"
    assert worker["generation_family"] == "autoregressive"
    assert worker["settings"]["hardware_verification_required"] is True


@pytest.mark.asyncio
async def test_disallowing_capability_reports_current_profile_references(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "discover_huggingface_models",
        lambda: [discovered_model(runtime="autoregressive-transformers")],
    )
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v4()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    profile = profile_document(worker.id)
    store.save_routing_profile_draft(profile)
    store.set_model_capability_allowed(worker.model_id, worker.revision, "autoregressive-trace", allowed=True)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/catalogue/capabilities/policy",
                json={
                    "model_id": worker.model_id,
                    "revision": worker.revision,
                    "capability_id": "autoregressive-trace",
                    "allowed": False,
                },
            )

    assert response.status_code == 409
    assert response.json()["detail"]["references"][0]["kind"] == "draft"


@pytest.mark.asyncio
async def test_new_worker_requires_an_allowed_concrete_capability(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "model" / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    model = {
        **discovered_model(runtime="autoregressive-transformers"),
        "cache_location": str(cache_root / "model"),
        "snapshot_location": str(snapshot),
    }
    monkeypatch.setattr(main_module, "discover_huggingface_models", lambda: [model])
    monkeypatch.setattr(v2_api_module, "discover_huggingface_models", lambda: [model])
    app = create_app(Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs"))
    request = {
        "name": "Allowed trace Worker",
        "model_id": model["model_id"],
        "revision": model["revision"],
        "runtime_template_id": "autoregressive-transformers",
        "capability_id": "autoregressive-trace",
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            denied = await client.post("/api/workers", json=request)
            await client.post(
                "/api/catalogue/capabilities/policy",
                json={
                    "model_id": model["model_id"],
                    "revision": model["revision"],
                    "capability_id": "autoregressive-trace",
                    "allowed": True,
                },
            )
            created = await client.post("/api/workers", json=request)

    assert denied.status_code == 409
    assert denied.json()["detail"] == "Allow this capability before creating a Worker"
    assert created.status_code == 201
    assert created.json()["capability_policy_version"] == 4


@pytest.mark.asyncio
async def test_management_ignores_persisted_untrusted_workers(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    legacy_worker = worker_definition(name="Legacy mock").model_copy(
        update={"runtime": "retired-test-runtime"}
    )
    store.save_worker_definition(legacy_worker.model_dump(mode="json"))

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            workers = await client.get("/api/workers")
            direct = await client.get(f"/api/workers/{legacy_worker.id}")

    assert workers.json() == []
    assert direct.status_code == 404


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


@pytest.mark.asyncio
async def test_profiles_are_active_together_and_reject_model_id_collisions(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    first_worker = worker_definition(name="Fast", port=8630)
    second_worker = worker_definition(name="Deep", port=8631)
    store.save_worker_definition(first_worker.model_dump(mode="json"))
    store.save_worker_definition(second_worker.model_dump(mode="json"))
    first = profile_document(first_worker.id, name="Existing")
    first["capabilities"][0].update({"public_name": "existing-local", "protocol_contract": "openai-chat-v1"})
    second = profile_document(second_worker.id, name="wayfinder-gate0")
    second["capabilities"][0].update({"public_name": "fast-local", "protocol_contract": "openai-chat-v1"})
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for profile in (first, second):
                assert (await client.post("/api/routing-profiles", json=profile)).status_code == 201
                assert (
                    await client.post(f"/api/routing-profiles/{profile['id']}/publish")
                ).status_code == 201
            live = (await client.get("/api/live")).json()
            duplicate = {**second, "id": str(uuid4()), "name": "collision"}
            assert (await client.post("/api/routing-profiles", json=duplicate)).status_code == 201
            rejected = await client.post(f"/api/routing-profiles/{duplicate['id']}/publish")

    assert {profile["name"] for profile in live["active_profiles"]} == {"Existing", "wayfinder-gate0"}
    assert {capability["public_name"] for capability in live["capabilities"]} == {
        "existing-local",
        "fast-local",
    }
    assert rejected.status_code == 409
    assert "unique API Model IDs" in rejected.json()["detail"]


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
async def test_management_prefix_cache_clear_returns_only_safe_counts(monkeypatch, tmp_path) -> None:
    import modeldeck.v2_api as v2_api

    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    worker = worker_definition().model_copy(
        update={
            "capabilities": {
                "chat": True,
                "completions": True,
                "top_k_trace": True,
                "prefix_caching": "application-managed",
                "prefix_cache_enabled": True,
            },
            "settings": {"prefix_cache_enabled": True},
        }
    )
    store.save_worker_definition(worker.model_dump(mode="json"))

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url: str):
            assert url == f"http://127.0.0.1:{worker.port}/prefix-cache/clear"
            return httpx.Response(
                200,
                json={"ok": True, "cleared_entries": 1, "released_bytes": 4096},
                request=httpx.Request("POST", url),
            )

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(v2_api.httpx, "AsyncClient", lambda *args, **kwargs: Client())
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        monkeypatch.setattr(
            app.state.supervisor,
            "get_worker",
            lambda _worker_id: {"state": "ready"},
        )
        async with real_async_client(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(f"/api/workers/{worker.id}/prefix-cache/clear")

    assert response.json() == {
        "ok": True,
        "worker_id": worker.id,
        "cleared_entries": 1,
        "released_bytes": 4096,
    }


@pytest.mark.asyncio
async def test_configuration_lock_blocks_profile_mutation_but_keeps_reads_available(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path, log_dir=tmp_path / "logs", configuration_locked=True))
    profile = profile_document(str(uuid4()), name="Locked")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            readable = await client.get("/api/routing-profiles")
            blocked = await client.post("/api/routing-profiles", json=profile)

    assert readable.status_code == 200
    assert blocked.status_code == 423
