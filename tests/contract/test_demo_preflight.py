from __future__ import annotations

import argparse
from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.demo_preflight import DEMO_ROUTES, local_url, preflight
from modeldeck.domain import RoutingProfile, routing_snapshot
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.adapters import PROTOCOL_ADAPTERS
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS

from tests.contract.test_gateway import worker


def test_management_and_gateway_publish_identical_exact_surfaces():
    assert PROTOCOL_CONTRACTS.keys() == PROTOCOL_ADAPTERS.keys()
    for name, contract in PROTOCOL_CONTRACTS.items():
        assert contract.surfaces == PROTOCOL_ADAPTERS[name].public_surfaces


@pytest.mark.asyncio
async def test_all_demo_discovery_preserves_other_profiles_and_ordered_backup(tmp_path, monkeypatch):
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    primary, backup = worker(), worker()
    backup = backup.model_copy(update={"name": "Backup Worker"})
    for definition in (primary, backup):
        store.save_worker_definition(definition.model_dump(mode="json"))
    requirements = {name: contract for routes in DEMO_ROUTES.values() for name, contract in routes.items()}
    for names in (requirements, {"unrelated-chat": "openai-chat-v1"}):
        profile = RoutingProfile(
            id=str(uuid4()),
            name="Applications " + str(uuid4()),
            capabilities=[
                {
                    "id": str(uuid4()),
                    "display_name": name,
                    "public_name": name,
                    "protocol_contract": contract,
                    "worker_ids": [primary.id, backup.id],
                }
                for name, contract in names.items()
            ],
        )
        store.save_routing_profile_draft(profile.model_dump(mode="json"))
        store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))
    before = store.active_routing_snapshots()

    async def health(_client, profile):
        return {"runtime": "mock"}, profile.id == backup.id

    monkeypatch.setattr("modeldeck.gateway.app.worker_health", health)
    app = create_gateway_app(settings=Settings(data_dir=tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as client:
        routes = (await client.get("/v1/routes")).json()
        models = (await client.get("/v1/models")).json()["data"]
        native = (await client.get("/native/v1/capabilities")).json()["capabilities"]
    assert store.active_routing_snapshots() == before
    assert len(routes["routes"]) == len(requirements) + 1
    for route in routes["routes"]:
        assert route["ready"] is True
        assert route["surfaces"] == list(PROTOCOL_CONTRACTS[route["protocol_contract"]].surfaces)
        resolution = routes["resolution"][route["public_name"]]
        assert resolution["configured_worker_ids"] == [primary.id, backup.id]
    assert {m["id"] for m in models} == {"speechshift-stt", "speechshift-voice", "unrelated-chat"}
    assert {r["public_name"] for r in native} == {
        "qwen-0-5b",
        "qwen-1-5b",
        "qwen-3b",
        "text-diffusion-lab-q4",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy,canonical,method",
    [
        ("/native/autoregressive/trace", "/native/v1/autoregressive/traces", "POST"),
        ("/v1/refine", "/native/v1/text-diffusion/refine", "POST"),
        ("/v1/diffuse", "/native/v1/text-diffusion/jobs", "POST"),
        ("/v1/jobs/missing", "/native/v1/text-diffusion/jobs/missing", "GET"),
        ("/v1/jobs/missing/events", "/native/v1/text-diffusion/jobs/missing/events", "GET"),
        ("/v1/jobs/missing/cancel", "/native/v1/text-diffusion/jobs/missing/cancel", "POST"),
    ],
)
async def test_legacy_alias_error_contracts(legacy, canonical, method, tmp_path):
    app = create_gateway_app({}, settings=Settings(data_dir=tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://127.0.0.1") as client:
        old = await client.request(method, legacy, json={"model": "missing"})
        new = await client.request(method, canonical, json={"model": "missing"})
    assert old.status_code == new.status_code
    assert old.status_code in {404, 503}
    assert old.json() == new.json()
    assert old.headers["deprecation"] == "true"
    assert old.headers["link"] == f'<{canonical}>; rel="successor-version"'
    assert "deprecation" not in new.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,temperature,assessment",
    [
        ("normal", 65, "snapshot_permits_rehearsal_review"),
        ("normal", 86, "pause_or_reduce"),
        ("normal", 90, "terminate"),
        ("hot", 70, "pause_or_reduce"),
        ("telemetry_degraded", None, "unknown"),
        ("critical", None, "terminate"),
    ],
)
async def test_preflight_separates_readiness_protocol_and_thermal(state, temperature, assessment):
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path))
        bodies = {
            "/api/health": {"status": "ok", "state_store": {"kind": "desktop-standalone"}},
            "/v1/health": {"status": "ok"},
            "/api/live": {"active_profiles": [{"id": "other", "revision": 7}]},
            "/v1/routes": {
                "routes": [
                    {"public_name": "ready", "protocol_contract": "native-ar-trace-v1", "ready": True},
                    {"public_name": "wrong", "protocol_contract": "openai-chat-v1", "ready": True},
                    {"public_name": "stopped", "protocol_contract": "native-ar-trace-v1", "ready": False},
                    {"public_name": "old-build", "ready": True},
                ]
            },
            "/v1/thermal": {"enabled": True, "state": state, "temperature_c": temperature},
        }
        return httpx.Response(200, json=bodies[request.url.path])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        report = await preflight(
            client,
            "http://127.0.0.1:3600",
            "http://127.0.0.1:8600",
            {
                "TokenTrail": dict.fromkeys(
                    ["ready", "wrong", "stopped", "missing", "old-build"], "native-ar-trace-v1"
                )
            },
        )
    assert len(calls) == 5
    assert all(method == "GET" for method, _ in calls)
    ready, wrong, stopped, missing, old = report["routes"]
    assert ready["protocol_compatible"] and ready["worker_ready"]
    assert wrong["protocol_compatible"] is False and wrong["worker_ready"] is True
    assert stopped["worker_ready"] is False
    assert missing["present"] is False and "Publish 'missing'" in missing["diagnostics"][0]
    assert old["protocol_compatible"] is None
    assert report["active_profiles"] == [{"id": "other", "revision": 7}]
    assert report["thermal_admission"]["assessment"] == assessment
    assert report["hardware_qualified"] is False
    assert report["checks_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["offline", "malformed", "degraded", "redirect"])
async def test_discovery_failure_is_unknown_not_missing(failure):
    def handle(request):
        if failure == "offline":
            raise httpx.ConnectError("offline", request=request)
        if failure == "degraded":
            return httpx.Response(503, json={"error": {"code": "schema_upgrade_required"}})
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "http://example.com"})
        return httpx.Response(200, json={"routes": [None]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        report = await preflight(client, "http://127.0.0.1:3600", "http://127.0.0.1:8600")
    assert all(route["present"] is None for route in report["routes"])
    assert report["checks_passed"] is False


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost:8600",
        "http://example.com:8600",
        "http://user:pass@localhost:8600",
        "http://localhost:8600/v1",
    ],
)
def test_preflight_rejects_nonlocal_or_ambiguous_base_urls(url):
    with pytest.raises(argparse.ArgumentTypeError):
        local_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_preflight_success_requires_enabled_thermal_observation(enabled):
    def handle(request):
        bodies = {
            "/api/health": {"status": "ok"},
            "/v1/health": {"status": "ok"},
            "/api/live": {"active_profiles": [{"id": "demo", "revision": 1}]},
            "/v1/routes": {
                "routes": [
                    {
                        "public_name": "scenechat-vision",
                        "protocol_contract": "scene-analysis-v1",
                        "ready": True,
                    }
                ]
            },
            "/v1/thermal": {"enabled": enabled, "state": "normal", "temperature_c": 60},
        }
        return httpx.Response(200, json=bodies[request.url.path])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        report = await preflight(
            client, "http://127.0.0.1:3600", "http://127.0.0.1:8600", {"SceneChat": DEMO_ROUTES["SceneChat"]}
        )
    assert report["checks_passed"] is enabled
    assert report["hardware_qualified"] is False
