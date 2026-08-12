import json
import sqlite3
from uuid import uuid4

import httpx
import pytest
from modeldeck.capabilities import worker_configuration_fingerprint
from modeldeck.compatibility import CompatibilityStore, LegacyDatabaseError
from modeldeck.config import Settings
from modeldeck.domain import RoutingProfile, WorkerDefinition, routing_snapshot, validate_routing_profile
from modeldeck.gateway.app import create_gateway_app
from modeldeck.main import create_app
from modeldeck.migrate_v2_to_v3 import migrate
from modeldeck.migrate_v3_to_v4 import migrate as migrate_v3_to_v4
from modeldeck.v2_api import _has_smoke_evidence, _worker_smoke_request


def worker_definition() -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name="Qwen trace Worker",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="transformers-rocm",
        runtime_template_id="autoregressive-transformers",
        runtime_template_version="2",
        lifecycle="on-demand",
        port=65535,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


def routing_profile(worker_id: str, *, qualification: str = "compatible") -> RoutingProfile:
    return RoutingProfile(
        id=str(uuid4()),
        name="Local applications",
        description="Token Trails and SprintBot",
        qualification=qualification,
        capabilities=[
            {
                "id": str(uuid4()),
                "display_name": "Token trace",
                "public_name": "qwen-0-5b",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [worker_id],
            }
        ],
    )


def test_v3_store_starts_empty_and_refuses_unmigrated_databases(tmp_path) -> None:
    path = tmp_path / "modeldeck.sqlite3"
    store = CompatibilityStore(path)
    store.initialise_v3()
    assert store.list_workers() == []
    assert store.list_routing_profiles() == []
    assert store.active_routing_snapshot() is None

    legacy_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(legacy_path) as database:
        database.execute("CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        database.execute("INSERT INTO schema_metadata VALUES ('schema_version', '2')")
    with pytest.raises(LegacyDatabaseError, match="migrate_v2_to_v3"):
        CompatibilityStore(legacy_path).initialise_v3()


def test_routing_profile_preserves_capability_worker_order() -> None:
    primary = worker_definition()
    backup = worker_definition().model_copy(
        update={"id": str(uuid4()), "name": "Backup Worker", "port": 8631}
    )
    profile = routing_profile(primary.id)
    profile.capabilities[0].worker_ids.append(backup.id)

    validation = validate_routing_profile(profile, [primary, backup], [])

    assert validation["valid"] is True
    assert [worker["role"] for worker in validation["capabilities"][0]["workers"]] == [
        "primary",
        "backup",
    ]
    assert routing_snapshot(profile, 4)["capabilities"][0]["worker_ids"] == [
        primary.id,
        backup.id,
    ]


def test_routing_profile_rejects_duplicate_public_model_names() -> None:
    worker = worker_definition()
    profile = routing_profile(worker.id)
    duplicate = profile.capabilities[0].model_copy(update={"id": str(uuid4())})

    with pytest.raises(ValueError, match="Routing Profile"):
        RoutingProfile.model_validate(
            {
                **profile.model_dump(mode="json"),
                "capabilities": [
                    profile.capabilities[0].model_dump(mode="json"),
                    duplicate.model_dump(mode="json"),
                ],
            }
        )


def test_embedding_contract_rejects_autoregressive_chat_workers() -> None:
    chat_worker = worker_definition()
    profile = RoutingProfile(
        id=str(uuid4()),
        name="Local applications",
        capabilities=[
            {
                "id": str(uuid4()),
                "display_name": "SprintBot embedding",
                "public_name": "sprintbot-embedding",
                "protocol_contract": "openai-embeddings-v1",
                "worker_ids": [chat_worker.id],
            }
        ],
    )

    validation = validate_routing_profile(profile, [chat_worker], [])

    assert validation["valid"] is False
    messages = [error["message"] for error in validation["errors"]]
    assert "Requires embedding, got autoregressive" in messages
    assert "Missing capabilities: embeddings" in messages


def test_tested_working_profile_requires_matching_evidence() -> None:
    worker = worker_definition()
    profile = routing_profile(worker.id, qualification="tested-working")
    assert validate_routing_profile(profile, [worker], [])["valid"] is False
    evidence = {
        "result": "tested-working",
        "evidence": {
            "model_id": worker.model_id,
            "model_revision": worker.revision,
            "runtime": worker.runtime,
        },
    }
    assert validate_routing_profile(profile, [worker], [evidence])["valid"] is True


def test_v4_profile_requires_capability_permission_and_exact_evidence() -> None:
    worker = worker_definition().model_copy(update={"capability_policy_version": 4})
    profile = routing_profile(worker.id, qualification="tested-working")
    identity = (worker.model_id, worker.revision, "autoregressive-trace")
    evidence = {
        "id": 7,
        "result": "tested-working",
        "evidence": {
            "worker_id": worker.id,
            "capability_id": "autoregressive-trace",
            "model_id": worker.model_id,
            "model_revision": worker.revision,
            "runtime": worker.runtime,
            "worker_configuration_fingerprint": worker_configuration_fingerprint(
                worker.model_dump(mode="json")
            ),
        },
    }

    denied = validate_routing_profile(profile, [worker], [evidence], {})
    allowed = validate_routing_profile(profile, [worker], [evidence], {identity: True})
    changed_worker = worker.model_copy(update={"dtype": "bfloat16"})
    stale = validate_routing_profile(profile, [changed_worker], [evidence], {identity: True})

    assert denied["valid"] is False
    assert "Allow the autoregressive-trace capability" in denied["errors"][0]["message"]
    assert allowed["valid"] is True
    assert stale["valid"] is False
    assert any("tested-working evidence" in error["message"] for error in stale["errors"])


def test_worker_smoke_requests_use_worker_protocols() -> None:
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

    embedding = autoregressive.model_copy(
        update={
            "generation_family": "embedding",
            "capabilities": {"embeddings": True},
        }
    )
    path, body, headers = _worker_smoke_request(embedding)
    assert path == "/v1/embeddings"
    assert body["input"] == ["The local Worker is ready."]
    assert headers is None


def test_embedding_smoke_requires_ordered_1024_dimension_vectors() -> None:
    embedding = worker_definition().model_copy(
        update={"generation_family": "embedding", "capabilities": {"embeddings": True}}
    )
    valid = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.0] * 1024},
            {"object": "embedding", "index": 1, "embedding": [1.0] * 1024},
        ],
    }

    assert _has_smoke_evidence(embedding, valid) is True
    assert _has_smoke_evidence(embedding, {**valid, "data": valid["data"][:1]}) is True
    assert (
        _has_smoke_evidence(
            embedding,
            {**valid, "data": [{"object": "embedding", "index": 1, "embedding": [0.0] * 1024}]},
        )
        is False
    )
    assert (
        _has_smoke_evidence(
            embedding,
            {**valid, "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 1023}]},
        )
        is False
    )


@pytest.mark.asyncio
async def test_management_and_gateway_use_only_a_published_profile(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, log_dir=tmp_path / "logs")
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    worker = worker_definition()
    store.save_worker_definition(worker.model_dump(mode="json"))
    profile = routing_profile(worker.id)
    store.save_routing_profile_draft(profile.model_dump(mode="json"))

    management_app = create_app(settings)
    async with management_app.router.lifespan_context(management_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=management_app), base_url="http://test"
        ) as management:
            assert (await management.get("/api/live")).json() == {
                "active_profile": None,
                "active_profiles": [],
                "capabilities": [],
            }
            publish = await management.post(f"/api/routing-profiles/{profile.id}/publish")
            assert publish.status_code == 201
            live = (await management.get("/api/live")).json()
    assert live["active_profile"]["name"] == profile.name
    assert live["active_profiles"] == [{"id": profile.id, "name": profile.name, "revision": 1}]
    assert live["capabilities"][0]["id"] == profile.capabilities[0].id
    assert live["capabilities"][0]["ready"] is False

    gateway_app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app), base_url="http://test"
    ) as gateway:
        assert (await gateway.get("/v1/models")).json() == {"object": "list", "data": []}
        native = (await gateway.get("/native/v1/capabilities")).json()["capabilities"]
    assert native == [
        {
            "id": profile.capabilities[0].id,
            "display_name": "Token trace",
            "public_name": "qwen-0-5b",
            "protocol_contract": "native-ar-trace-v1",
            "surfaces": ["POST /native/v1/autoregressive/traces"],
            "ready": False,
            "metadata": {"generation_family": "autoregressive", "worker_count": 1},
        }
    ]


def test_migration_converts_event_revisions_and_drops_demo_membership(tmp_path) -> None:
    path = tmp_path / "modeldeck.sqlite3"
    worker = worker_definition()
    event_id, route_id, demo_id = str(uuid4()), str(uuid4()), str(uuid4())
    event = {
        "id": event_id,
        "name": "Open Day",
        "description": "Token Trail",
        "qualification": "compatible",
        "routes": [
            {
                "id": route_id,
                "display_name": "Token trace",
                "public_name": "token-trail",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [worker.id],
            }
        ],
        "demos": [{"id": demo_id, "name": "Token Trail", "route_ids": [route_id]}],
    }
    with sqlite3.connect(path) as database:
        database.executescript(
            """
            CREATE TABLE schema_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE events (
                id TEXT PRIMARY KEY, draft_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE event_revisions (
                event_id TEXT NOT NULL, revision INTEGER NOT NULL,
                document_json TEXT NOT NULL, published_at TEXT NOT NULL,
                PRIMARY KEY (event_id, revision)
            );
            CREATE TABLE active_event (
                singleton_id INTEGER PRIMARY KEY, event_id TEXT NOT NULL,
                revision INTEGER NOT NULL, routing_json TEXT NOT NULL,
                published_at TEXT NOT NULL
            );
            """
        )
        database.execute("INSERT INTO schema_metadata VALUES ('schema_version', '2', 'now')")
        database.execute("INSERT INTO events VALUES (?, ?, 'now', 'now')", (event_id, json.dumps(event)))
        database.execute("INSERT INTO event_revisions VALUES (?, 1, ?, 'now')", (event_id, json.dumps(event)))
        database.execute("INSERT INTO active_event VALUES (1, ?, 1, '{}', 'now')", (event_id,))

    migrate(path)
    migrate_v3_to_v4(path)
    store = CompatibilityStore(path)
    store.initialise_v4()
    profile = store.get_routing_profile(event_id)

    assert profile is not None
    assert profile["definition"]["capabilities"][0]["id"] == route_id
    assert "demos" not in profile["definition"]
    assert store.active_routing_snapshot()["profile_id"] == event_id
