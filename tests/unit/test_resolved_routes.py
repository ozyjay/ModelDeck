from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import RoutingProfile, routing_snapshot
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import model_discovery_record, route_role, worker_discovery_identity, worker_health
from modeldeck.gateway.resolution import ResolvedRoute
from modeldeck.persistence import connect_database

from tests.contract.test_gateway import worker


@pytest.mark.parametrize("condition", ["missing", "archived", "corrupt", "retired"])
def test_backup_never_becomes_configured_primary(condition):
    primary = worker().model_dump(mode="json")
    backup = worker().model_dump(mode="json")
    records = [{"definition": backup}]
    if condition != "missing":
        document = primary.copy()
        if condition == "corrupt":
            document = {}
        elif condition == "retired":
            document["runtime"] = "retired-runtime"
        records.append(
            {
                "id": primary["id"],
                "definition": document,
                "archived_at": "yesterday" if condition == "archived" else None,
            }
        )
    route = ResolvedRoute.resolve("visitor-chat", [primary["id"], backup["id"]], records)
    assert [profile.id for profile in route.candidates] == [backup["id"]]
    assert route_role(route.candidates[0], route.candidates) == "backup"
    states = {backup["id"]: {"ready": True, "health": {}}}
    result = model_discovery_record("visitor-chat", route.candidates, states, route=route.metadata())
    assert result["modeldeck"]["selection_reason"] == "backup_ready"
    assert result["modeldeck"]["primary_worker"] is None
    assert result["modeldeck"]["route"]["configured_worker_ids"][0] == primary["id"]


def test_no_resolved_candidates_remains_discoverable_without_substitution():
    primary_id = str(uuid4())
    route = ResolvedRoute.resolve("visitor-chat", [primary_id], [])
    record = model_discovery_record("visitor-chat", route.candidates, {}, route=route.metadata())
    assert not record["ready"]
    assert record["modeldeck"]["selected_worker"] is None
    assert record["modeldeck"]["route"]["configured_workers"][0]["status"] == "missing"


def test_requested_and_reported_execution_identity_remain_separate():
    profile = (
        worker()
        .to_profile()
        .model_copy(
            update={
                "dtype": "float16",
                "artifact_model_id": "org/artifact",
                "artifact_revision": "artifact-revision",
                "settings": {
                    "backend": "requested-backend",
                    "device": "requested-device",
                    "context_length": 2048,
                    "kv_cache_dtype": "float16",
                },
            }
        )
    )
    observed = {
        "ready": True,
        "health": {
            "model_id": "reported-model",
            "model_revision": "reported-revision",
            "runtime": "reported-runtime",
            "artifact_model_id": "reported-artifact",
            "artifact_revision": "reported-artifact-revision",
            "backend": "reported-backend",
            "device": "reported-device",
            "dtype": "bfloat16",
            "context_length": 1024,
            "kv_cache_dtype": "bfloat16",
        },
    }
    identity = worker_discovery_identity(profile, observed)
    requested, resolved = identity["requested"], identity["resolved"]
    assert requested["model_id"] == profile.model_id
    assert requested["artifact_model_id"] == "org/artifact"
    assert requested["artifact_revision"] == "artifact-revision"
    assert requested["backend"] == "requested-backend"
    assert requested["device"] == "requested-device"
    assert requested["precision"] == "float16"
    assert requested["context_length"] == 2048
    assert requested["kv_cache"]["kv_cache_dtype"] == "float16"
    assert resolved["model_id"] == "reported-model"
    assert resolved["model_revision"] == "reported-revision"
    assert resolved["runtime"] == "reported-runtime"
    assert resolved["backend"] == "reported-backend"
    assert resolved["device"] == "reported-device"
    assert resolved["precision"] == "bfloat16"
    assert resolved["context_length"] == 1024
    assert resolved["kv_cache"]["kv_cache_dtype"] == "bfloat16"
    assert worker_discovery_identity(profile, {"ready": False, "health": {}})["resolved"] is None


@pytest.mark.parametrize("payload", [[], None, "invalid", {"worker_id": "another-worker", "ready": True}])
async def test_malformed_health_is_unavailable(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        _, ready = await worker_health(client, worker().to_profile())
    assert ready is False


@pytest.mark.parametrize("condition", ["missing", "archived", "corrupt", "retired", "unavailable"])
async def test_api_discovery_survives_unusable_worker(tmp_path, condition, monkeypatch):
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v5()
    definition = worker()
    store.save_worker_definition(definition.model_dump(mode="json"))
    profile = RoutingProfile(
        id=str(uuid4()),
        name="Discovery",
        capabilities=[
            {
                "id": str(uuid4()),
                "display_name": "Chat",
                "public_name": "chat-route",
                "protocol_contract": "openai-chat-v1",
                "worker_ids": [definition.id],
            },
            {
                "id": str(uuid4()),
                "display_name": "Trace",
                "public_name": "trace-route",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [definition.id],
            },
        ],
    )
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))
    with connect_database(store.path) as database:
        if condition == "missing":
            database.execute("DELETE FROM workers")
        elif condition == "archived":
            database.execute("UPDATE workers SET archived_at='yesterday'")
        elif condition == "corrupt":
            database.execute("UPDATE workers SET document_json='broken JSON'")
        elif condition == "retired":
            database.execute(
                "UPDATE workers SET document_json=json_set(document_json, '$.runtime', 'retired')"
            )

    async def unavailable(*args):
        return None, False

    monkeypatch.setattr("modeldeck.gateway.app.worker_health", unavailable)
    app = create_gateway_app(settings=Settings(data_dir=tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        models = await client.get("/v1/models")
        assert models.status_code == 200
        model = models.json()["data"][0]
        assert model["ready"] is False
        assert model["modeldeck"]["route"]["configured_worker_ids"] == [definition.id]
        native = await client.get("/native/v1/capabilities")
        assert native.status_code == 200
        assert native.json()["capabilities"][0]["ready"] is False
        assert (await client.get("/v1/capabilities")).status_code == 200
