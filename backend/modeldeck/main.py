from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from modeldeck.catalogue import discover_huggingface_models
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.demo_config import (
    DEMO_ADAPTERS,
    DemoRouteContract,
    DemoSetDefinition,
    default_demo_set,
    routing_snapshot,
    validate_demo_set,
)
from modeldeck.hardware import probe_environment
from modeldeck.profile_registry import (
    load_local_profiles,
    profile_allowed,
    profile_cache_identity,
    profile_uses_huggingface_cache,
    profile_verified,
)
from modeldeck.profiles import (
    LOCAL_PORT_RANGE,
    RESERVED_GATEWAY_ALIASES,
    LocalProfileRequest,
    create_local_profile,
    default_model_profiles,
)
from modeldeck.provider_selection import provider_compatible, selectable_aliases
from modeldeck.q4_release import Q4ReleaseError, verify_modeldeck_q4_release
from modeldeck.registry import ReservedAlias
from modeldeck.supervisor import WorkerSupervisor

FRONTEND_ROOT = Path(__file__).parent / "api/static"
FRONTEND_FALLBACK = """<!doctype html><html lang="en-AU"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ModelDeck</title></head>
<body><main><h1>ModelDeck operator console is not built</h1>
<p>Run <code>pwsh -NoProfile -File scripts/build_frontend.ps1</code> and restart ModelDeck.</p>
</main></body></html>"""


async def _run_blocking[BlockingResult](
    function: Callable[..., BlockingResult],
    *arguments: object,
) -> BlockingResult:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modeldeck-management")
    try:
        return await asyncio.get_running_loop().run_in_executor(executor, function, *arguments)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class LifecycleEvidence(BaseModel):
    shutdown_result: Literal["success", "failed"]
    memory_recovery_result: Literal[
        "not-measured-process-exit-confirmed",
        "measured-recovered",
        "measured-not-recovered",
    ]
    stability_duration_seconds: float | None = Field(default=None, ge=0)
    stability_request_count: int | None = Field(default=None, ge=0)
    stability_failures: int | None = Field(default=None, ge=0)


class ModelCachePolicyRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    allowed: bool


class GatewayProviderSelectionRequest(BaseModel):
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    built_in_profiles = default_model_profiles()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured.data_dir.mkdir(parents=True, exist_ok=True)
        store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")
        store.initialise()
        app.state.compatibility_store = store
        policy = store.list_model_cache_policy()
        for profile in built_in_profiles:
            if not profile_allowed(profile, policy):
                await app.state.supervisor.remove_profile(profile.id)
        for profile in load_local_profiles(configured.data_dir):
            app.state.profiles.append(profile)
            if profile_allowed(profile, policy):
                app.state.supervisor.register_profile(profile)
        if not store.list_demo_sets():
            store.save_demo_set(default_demo_set(app.state.profiles).model_dump(mode="json"))
        yield
        await app.state.supervisor.stop_all()

    app = FastAPI(
        title="ModelDeck management API",
        version="0.1.0",
        description="Local-only management for isolated model workers.",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.supervisor = WorkerSupervisor(built_in_profiles, log_dir=configured.log_dir)
    app.state.profiles = list(built_in_profiles)
    app.state.built_in_profile_ids = {profile.id for profile in built_in_profiles}
    assets = FRONTEND_ROOT / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.middleware("http")
    async def browser_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(("/api", "/docs", "/redoc", "/openapi.json")):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            )
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        return _frontend_index()

    @app.get("/api/health")
    async def health(request: Request):
        settings = request.app.state.settings
        return {
            "status": "ok",
            "service": "modeldeck-management",
            "open_day": settings.open_day,
            "downloads_allowed": settings.allow_downloads,
            "gateway_url": f"http://{settings.host}:{settings.gateway_port}",
        }

    @app.get("/api/gateway/status")
    async def gateway_status(request: Request):
        return await _gateway_status(request.app.state.settings)

    async def provider_selection_status(
        alias: str,
        contract: ReservedAlias,
        request: Request,
    ) -> dict:
        store = request.app.state.compatibility_store
        stored_selection = store.gateway_provider_selection(alias)
        selected_provider = stored_selection or contract.default_provider
        policy = store.list_model_cache_policy()
        supervisor = request.app.state.supervisor
        gateway = await _gateway_status(request.app.state.settings)
        gateway_model = next(
            (model for model in (gateway.get("models") or {}).get("data", []) if model.get("id") == alias),
            {},
        )
        effective_provider = gateway_model.get("effective_provider")
        candidates = []
        compatibility_tests = store.list_tests()
        for profile in request.app.state.profiles:
            if (
                profile.id not in supervisor.workers
                or not provider_compatible(contract, profile)
                or not profile_allowed(profile, policy)
            ):
                continue
            worker = supervisor.get_worker(profile.id)
            candidates.append(
                {
                    "profile_id": profile.id,
                    "profile_alias": profile.alias,
                    "model_id": profile.model_id,
                    "selected": profile.id == selected_provider,
                    "worker_state": worker["state"],
                    "gateway_ready": profile.id == effective_provider,
                    "verified": profile_verified(profile, compatibility_tests),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                candidate["profile_id"] != contract.default_provider,
                candidate["profile_id"],
            )
        )
        return {
            "alias": alias,
            "display_name": contract.display_name,
            "default_provider": contract.default_provider,
            "explicit_selection": stored_selection is not None,
            "selected_provider": selected_provider,
            "effective_provider": effective_provider,
            "gateway_ready": gateway_model.get("ready") is True,
            "candidates": candidates,
        }

    @app.get("/api/gateway/provider-selections")
    async def list_provider_selections(request: Request):
        return {
            "selections": [
                await provider_selection_status(alias, contract, request)
                for alias, contract in selectable_aliases().items()
            ]
        }

    @app.get("/api/gateway/provider-selections/{alias}")
    async def get_provider_selection(alias: str, request: Request):
        contract = selectable_aliases().get(alias)
        if contract is None:
            raise HTTPException(404, "Unknown selectable reserved gateway alias")
        return await provider_selection_status(alias, contract, request)

    @app.post("/api/gateway/provider-selections/{alias}")
    async def set_provider_selection(
        alias: str,
        payload: GatewayProviderSelectionRequest,
        request: Request,
    ):
        _require_configuration_mutable(request)
        contract = selectable_aliases().get(alias)
        if contract is None:
            raise HTTPException(404, "Unknown selectable reserved gateway alias")
        profile = next(
            (candidate for candidate in request.app.state.profiles if candidate.id == payload.profile_id),
            None,
        )
        if profile is None or profile.id not in request.app.state.supervisor.workers:
            raise HTTPException(409, "Select an existing registered ModelDeck runtime profile")
        if not provider_compatible(contract, profile):
            raise HTTPException(
                409,
                "The selected profile does not satisfy this reserved alias contract",
            )
        if not profile_verified(profile, request.app.state.compatibility_store.list_tests()):
            raise HTTPException(409, "Record successful hardware compatibility evidence first")
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        if not profile_allowed(profile, policy):
            raise HTTPException(409, "Allow the selected cached model in ModelDeck first")
        request.app.state.compatibility_store.set_gateway_provider_selection(
            alias,
            profile.id,
        )
        return await provider_selection_status(alias, contract, request)

    @app.get("/api/hardware")
    async def hardware():
        return await asyncio.to_thread(probe_environment)

    @app.get("/api/telemetry")
    async def telemetry():
        probe = await asyncio.to_thread(probe_environment)
        detected = probe["detected"]
        return {
            key: detected[key]
            for key in ("memory", "swap", "filesystems", "temperatures", "fans", "active_model_processes")
        }

    @app.get("/api/catalogue")
    async def catalogue(request: Request):
        models = await asyncio.to_thread(discover_huggingface_models)
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        return {
            "models": [
                {
                    **model,
                    "modeldeck_allowed": policy.get((model["model_id"], model["revision"]), True),
                }
                for model in models
            ],
            "downloads_started": False,
        }

    @app.post("/api/catalogue/policy")
    async def set_catalogue_policy(payload: ModelCachePolicyRequest, request: Request):
        _require_configuration_mutable(request)
        cached = next(
            (
                model
                for model in discover_huggingface_models()
                if model["model_id"] == payload.model_id and model["revision"] == payload.revision
            ),
            None,
        )
        if cached is None:
            raise HTTPException(404, "The exact cached model revision was not discovered")
        matching_profiles = [
            profile
            for profile in request.app.state.profiles
            if profile_cache_identity(profile) == (payload.model_id, payload.revision)
            and profile_uses_huggingface_cache(profile)
        ]
        selected_providers = {
            request.app.state.compatibility_store.gateway_provider_selection(alias)
            or contract.default_provider
            for alias, contract in selectable_aliases().items()
        }
        if not payload.allowed and any(profile.id in selected_providers for profile in matching_profiles):
            raise HTTPException(
                409,
                "Select a different reserved-alias provider before disallowing this model",
            )
        supervisor = request.app.state.supervisor
        if not payload.allowed:
            for profile in matching_profiles:
                if profile.id not in supervisor.workers:
                    continue
                state = supervisor.get_worker(profile.id)["state"]
                if state not in {"stopped", "failed"}:
                    raise HTTPException(
                        409,
                        f"Stop worker {profile.id} before disallowing this cached model",
                    )
            removed_profiles = []
            try:
                for profile in matching_profiles:
                    if profile.id in supervisor.workers:
                        await supervisor.remove_profile(profile.id)
                        removed_profiles.append(profile)
            except RuntimeError as error:
                for removed in removed_profiles:
                    supervisor.register_profile(removed)
                raise HTTPException(409, str(error)) from error
        else:
            registered_profiles = []
            try:
                for profile in matching_profiles:
                    if profile.id not in supervisor.workers:
                        supervisor.register_profile(profile)
                        registered_profiles.append(profile)
            except ValueError as error:
                for registered in registered_profiles:
                    await supervisor.remove_profile(registered.id)
                raise HTTPException(409, str(error)) from error
        request.app.state.compatibility_store.set_model_cache_allowed(
            payload.model_id,
            payload.revision,
            allowed=payload.allowed,
        )
        return {
            "ok": True,
            "model_id": payload.model_id,
            "revision": payload.revision,
            "allowed": payload.allowed,
            "cache_removed": False,
            "affected_profiles": [profile.id for profile in matching_profiles],
        }

    @app.get("/api/profiles")
    async def list_profiles(request: Request):
        built_in_ids = request.app.state.built_in_profile_ids
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        return [
            {
                **profile.model_dump(mode="json"),
                "source": "built-in" if profile.id in built_in_ids else "local",
                "modeldeck_allowed": profile_allowed(profile, policy),
            }
            for profile in request.app.state.profiles
        ]

    @app.get("/api/deployments")
    async def list_deployments(request: Request):
        built_in_ids = request.app.state.built_in_profile_ids
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        supervisor = request.app.state.supervisor
        return [
            {
                "id": profile.id,
                "source": "packaged" if profile.id in built_in_ids else "local",
                "model": {
                    "model_id": profile.model_id,
                    "revision": profile.revision,
                    "artifact_model_id": profile.artifact_model_id,
                    "artifact_revision": profile.artifact_revision,
                },
                "runtime": profile.preferred_runtime,
                "generation_family": profile.generation_family,
                "lifecycle": profile.lifecycle,
                "capabilities": profile.capabilities.model_dump(mode="json"),
                "allowed": profile_allowed(profile, policy),
                "registered": profile.id in supervisor.workers,
                "worker": (supervisor.get_worker(profile.id) if profile.id in supervisor.workers else None),
            }
            for profile in request.app.state.profiles
        ]

    @app.get("/api/demo-adapters")
    async def list_demo_adapters():
        return {"adapters": [adapter.model_dump(mode="json") for adapter in DEMO_ADAPTERS.values()]}

    @app.get("/api/demo-sets")
    async def list_demo_sets(request: Request):
        return {
            "demo_sets": [
                _demo_set_response(record)
                for record in request.app.state.compatibility_store.list_demo_sets()
            ]
        }

    @app.get("/api/demo-sets/{demo_set_id}")
    async def get_demo_set(demo_set_id: str, request: Request):
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id)
        if record is None:
            raise HTTPException(404, "Unknown demo set")
        return _demo_set_response(record)

    @app.get("/api/demo-sets/{demo_set_id}/revisions")
    async def list_demo_set_revisions(demo_set_id: str, request: Request):
        revisions = request.app.state.compatibility_store.list_demo_set_revisions(demo_set_id)
        if not revisions:
            raise HTTPException(404, "Unknown demo set")
        return {"revisions": [_demo_set_response(record) for record in revisions]}

    @app.get("/api/demo-sets/{demo_set_id}/revisions/{revision}")
    async def get_demo_set_revision(demo_set_id: str, revision: int, request: Request):
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id, revision)
        if record is None:
            raise HTTPException(404, "Unknown demo set revision")
        return _demo_set_response(record)

    @app.post("/api/demo-sets", status_code=201)
    async def create_demo_set(payload: DemoSetDefinition, request: Request):
        _require_configuration_mutable(request)
        store = request.app.state.compatibility_store
        if store.get_demo_set(payload.id) is not None:
            raise HTTPException(409, "That demo set already exists")
        return _demo_set_response(store.save_demo_set(payload.model_dump(mode="json")))

    @app.put("/api/demo-sets/{demo_set_id}")
    async def update_demo_set(demo_set_id: str, payload: DemoSetDefinition, request: Request):
        _require_configuration_mutable(request)
        if payload.id != demo_set_id:
            raise HTTPException(409, "The demo set identifier cannot be changed")
        store = request.app.state.compatibility_store
        if store.get_demo_set(demo_set_id) is None:
            raise HTTPException(404, "Unknown demo set")
        return _demo_set_response(store.save_demo_set(payload.model_dump(mode="json")))

    @app.delete("/api/demo-sets/{demo_set_id}")
    async def delete_demo_set(demo_set_id: str, request: Request):
        _require_configuration_mutable(request)
        try:
            removed = request.app.state.compatibility_store.delete_demo_set(demo_set_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        if not removed:
            raise HTTPException(404, "Unknown demo set")
        return {"ok": True, "demo_set_id": demo_set_id}

    @app.post("/api/demo-sets/{demo_set_id}/validate")
    async def validate_stored_demo_set(demo_set_id: str, request: Request):
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id)
        if record is None:
            raise HTTPException(404, "Unknown demo set")
        definition = DemoSetDefinition.model_validate(record["definition"])
        return {
            "demo_set_id": demo_set_id,
            "revision": record["revision"],
            **_validate_demo_definition(definition, request),
        }

    @app.post("/api/demo-sets/{demo_set_id}/plan")
    async def plan_demo_set(demo_set_id: str, request: Request):
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id)
        if record is None:
            raise HTTPException(404, "Unknown demo set")
        definition = DemoSetDefinition.model_validate(record["definition"])
        validation = _validate_demo_definition(definition, request)
        return {
            "demo_set_id": demo_set_id,
            "revision": record["revision"],
            "validation": validation,
            **_demo_set_plan(definition, request),
        }

    @app.post("/api/demo-sets/{demo_set_id}/activate")
    async def activate_demo_set(demo_set_id: str, request: Request):
        _require_configuration_mutable(request)
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id)
        if record is None:
            raise HTTPException(404, "Unknown demo set")
        return _activate_demo_set_record(demo_set_id, record, request)

    @app.post("/api/demo-sets/{demo_set_id}/revisions/{revision}/activate")
    async def activate_demo_set_revision(demo_set_id: str, revision: int, request: Request):
        _require_configuration_mutable(request)
        record = request.app.state.compatibility_store.get_demo_set(demo_set_id, revision)
        if record is None:
            raise HTTPException(404, "Unknown demo set revision")
        return _activate_demo_set_record(demo_set_id, record, request)

    @app.post("/api/demo-sets/{demo_set_id}/revisions/{revision}/restore", status_code=201)
    async def restore_demo_set_revision(demo_set_id: str, revision: int, request: Request):
        _require_configuration_mutable(request)
        store = request.app.state.compatibility_store
        record = store.get_demo_set(demo_set_id, revision)
        if record is None:
            raise HTTPException(404, "Unknown demo set revision")
        restored = store.save_demo_set(record["definition"])
        return _demo_set_response(restored)

    @app.get("/api/demo-sets/{demo_set_id}/routes/{route_id}/status")
    async def demo_route_status(demo_set_id: str, route_id: str, request: Request):
        record, route = _demo_route_record(demo_set_id, route_id, request)
        gateway = await _gateway_status(request.app.state.settings)
        gateway_model = next(
            (
                model
                for model in (gateway.get("models") or {}).get("data", [])
                if model.get("id") == route.public_model
            ),
            {},
        )
        supervisor = request.app.state.supervisor
        providers = []
        for binding in sorted(route.providers, key=lambda item: (item.priority, item.deployment_id)):
            worker = (
                supervisor.get_worker(binding.deployment_id)
                if binding.deployment_id in supervisor.workers
                else None
            )
            providers.append(
                {
                    "deployment_id": binding.deployment_id,
                    "priority": binding.priority,
                    "worker_state": worker.get("state") if worker else "unregistered",
                }
            )
        active = record["active_revision"] == record["revision"]
        return {
            "demo_set_id": demo_set_id,
            "revision": record["revision"],
            "route_id": route.id,
            "public_model": route.public_model,
            "adapter_id": route.adapter_id,
            "active": active,
            "gateway_available": gateway["available"],
            "advertised": active and bool(gateway_model),
            "ready": active and gateway_model.get("ready") is True,
            "selected_provider": gateway_model.get("selected_provider") if active else None,
            "effective_provider": gateway_model.get("effective_provider") if active else None,
            "providers": providers,
            "smoke_supported": route.adapter_id != "speech-conversation-v1",
            "smoke_unavailable_reason": (
                "Speech conversation rehearsal requires an interactive WebSocket client"
                if route.adapter_id == "speech-conversation-v1"
                else None
            ),
        }

    @app.post("/api/demo-sets/{demo_set_id}/routes/{route_id}/smoke")
    async def smoke_demo_route(demo_set_id: str, route_id: str, request: Request):
        record, route = _demo_route_record(demo_set_id, route_id, request)
        if record["active_revision"] != record["revision"]:
            raise HTTPException(409, "Activate this revision before rehearsing its gateway routes")
        return await _smoke_demo_route(route, request.app.state.settings)

    @app.post("/api/profiles", status_code=201)
    async def create_profile(payload: LocalProfileRequest, request: Request):
        _require_configuration_mutable(request)
        catalogue = discover_huggingface_models()
        cached = next(
            (
                model
                for model in catalogue
                if model["model_id"] == payload.model_id
                and model["revision"] == payload.revision
                and model["download_state"] == "installed-untested"
            ),
            None,
        )
        if cached is None:
            raise HTTPException(409, "The requested pinned snapshot is not complete in the local cache")
        if not request.app.state.compatibility_store.model_cache_allowed(payload.model_id, payload.revision):
            raise HTTPException(409, "Allow this cached model in ModelDeck before configuring it")
        configuration_support = cached["configuration_support"]
        if configuration_support is None:
            raise HTTPException(
                409,
                cached["configuration_support_reason"],
            )
        checkpoint_dir = (
            Path(cached["snapshot_location"])
            if configuration_support == "diffusiongemma-modeldeck-q4"
            else None
        )
        artifact_path = None
        if configuration_support == "gpt-oss-llama-vulkan":
            artifact = next(
                (
                    candidate
                    for candidate in cached.get("artifacts", [])
                    if candidate["artifact_id"] == payload.artifact_id
                ),
                None,
            )
            if artifact is None:
                raise HTTPException(409, "Select a discovered GPT-OSS MXFP4 artefact")
            artifact_path = Path(cached["snapshot_location"]) / artifact["filenames"][0]
        if checkpoint_dir is not None:
            try:
                await _run_blocking(verify_modeldeck_q4_release, checkpoint_dir)
            except (OSError, Q4ReleaseError) as error:
                raise HTTPException(409, f"ModelDeck Q4 release verification failed: {error}") from error

        profiles = request.app.state.profiles
        profile_id = f"local-{payload.profile_name or payload.alias}"
        reserved_contract = selectable_aliases().get(payload.alias)
        dynamic_reserved_alias = reserved_contract is not None and not reserved_contract.providers
        if (payload.alias in RESERVED_GATEWAY_ALIASES and not dynamic_reserved_alias) or any(
            profile.alias == payload.alias for profile in profiles
        ):
            raise HTTPException(409, "That gateway alias is already reserved or configured")
        if any(profile.id == profile_id for profile in profiles):
            raise HTTPException(409, "That local runtime configuration already exists")
        used_ports = {profile.port for profile in profiles}
        port = next((candidate for candidate in LOCAL_PORT_RANGE if candidate not in used_ports), None)
        if port is None:
            raise HTTPException(409, "No local ModelDeck worker ports are available")

        cache_root = Path(cached["cache_location"]).parent
        profile = create_local_profile(
            payload,
            cache_root=cache_root,
            port=port,
            configuration_support=configuration_support,
            checkpoint_dir=checkpoint_dir,
            base_model_id=cached.get("base_model_id"),
            base_model_revision=cached.get("base_model_revision"),
            artifact_path=artifact_path,
        )
        if reserved_contract is not None and not provider_compatible(reserved_contract, profile):
            raise HTTPException(409, "The runtime does not satisfy that reserved gateway alias contract")
        store = request.app.state.compatibility_store
        store.save_model_profile(profile.model_dump(mode="json"))
        try:
            request.app.state.supervisor.register_profile(profile)
        except ValueError as error:
            store.delete_model_profile(profile.id)
            raise HTTPException(409, str(error)) from error
        profiles.append(profile)
        return {**profile.model_dump(mode="json"), "source": "local"}

    @app.delete("/api/profiles/{profile_id}")
    async def delete_profile(profile_id: str, request: Request):
        _require_configuration_mutable(request)
        if profile_id in request.app.state.built_in_profile_ids:
            raise HTTPException(409, "Built-in runtime configurations cannot be removed")
        profile = next(
            (candidate for candidate in request.app.state.profiles if candidate.id == profile_id),
            None,
        )
        if profile is None:
            raise HTTPException(404, "Unknown local runtime configuration")
        selected_alias = next(
            (
                alias
                for alias in selectable_aliases()
                if request.app.state.compatibility_store.gateway_provider_selection(alias) == profile_id
            ),
            None,
        )
        if selected_alias is not None:
            raise HTTPException(
                409,
                f"Select a different {selected_alias} provider before removing this runtime configuration",
            )
        if profile_id in request.app.state.supervisor.workers:
            try:
                await request.app.state.supervisor.remove_profile(profile_id)
            except RuntimeError as error:
                raise HTTPException(409, str(error)) from error
        request.app.state.compatibility_store.delete_model_profile(profile_id)
        request.app.state.profiles.remove(profile)
        return {
            "ok": True,
            "profile_id": profile_id,
            "cache_removed": False,
        }

    @app.get("/api/workers")
    async def list_workers(request: Request):
        return request.app.state.supervisor.list_workers()

    @app.get("/api/workers/{worker_id}")
    async def get_worker(worker_id: str, request: Request):
        return _worker_call(request, "get_worker", worker_id)

    @app.post("/api/workers/{worker_id}/start")
    async def start_worker(worker_id: str, request: Request):
        return await _worker_async_call(request, "start", worker_id)

    @app.post("/api/workers/{worker_id}/stop")
    async def stop_worker(worker_id: str, request: Request):
        return await _worker_async_call(request, "stop", worker_id)

    @app.post("/api/workers/{worker_id}/restart")
    async def restart_worker(worker_id: str, request: Request):
        return await _worker_async_call(request, "restart", worker_id)

    @app.post("/api/workers/{worker_id}/warmup")
    async def warmup_worker(worker_id: str, request: Request):
        worker = _worker_call(request, "get_worker", worker_id)
        if worker["state"] != "ready":
            raise HTTPException(409, "Worker must be ready before warmup")
        return {"ok": True, "worker_id": worker_id}

    @app.post("/api/workers/{worker_id}/smoke")
    async def smoke_worker(worker_id: str, request: Request):
        worker = _worker_call(request, "get_worker", worker_id)
        if worker["state"] != "ready":
            raise HTTPException(409, "Worker must be ready before smoke testing")
        endpoint = worker["endpoint"]
        started = asyncio.get_running_loop().time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                health_response, model_response, metrics_response = await asyncio.gather(
                    client.get(f"{endpoint}/health"),
                    client.get(f"{endpoint}/model"),
                    client.get(f"{endpoint}/metrics"),
                )
                if worker["runtime"] == "llama-vulkan":
                    trace_response = await client.post(
                        f"{endpoint}/v1/chat/completions",
                        json={
                            "model": worker["alias"],
                            "messages": [{"role": "user", "content": "Reply with the word ready."}],
                            "max_tokens": 4,
                            "temperature": 0,
                            "stream": False,
                        },
                    )
                elif worker["generation_family"] == "autoregressive":
                    trace_response = await client.post(
                        f"{endpoint}/native/autoregressive/trace",
                        json={
                            "model": worker["alias"],
                            "prompt": "Reply with the word ready.",
                            "max_tokens": 4,
                            "temperature": 0,
                            "top_k": 3,
                            "seed": 7,
                        },
                    )
                elif worker["generation_family"] == "vision-language":
                    trace_response = await client.post(
                        f"{endpoint}/native/vision-language/smoke",
                        headers={
                            "Authorization": (
                                "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")
                            )
                        },
                    )
                elif worker["generation_family"] == "speech-conversation":
                    trace_response = await client.post(f"{endpoint}/smoke")
                else:
                    trace_response = await client.post(
                        f"{endpoint}/v1/refine",
                        json={
                            "model": worker["alias"],
                            "prompt": "A local worker is ready.",
                            "denoising_steps": 4,
                            "seed": 7,
                        },
                    )
                for response in (health_response, model_response, metrics_response, trace_response):
                    response.raise_for_status()
            health_payload = health_response.json()
            model_payload = model_response.json()
            metrics_payload = metrics_response.json()
            trace_payload = trace_response.json()
            smoke_evidence = (
                trace_payload.get("events")
                or trace_payload.get("frames")
                or trace_payload.get("ok")
                or trace_payload.get("choices")
            )
            if not smoke_evidence:
                raise RuntimeError("Smoke request returned no generation evidence")
            result = "tested-working"
            failure_class = None
            error_summary = None
        except Exception as error:
            health_payload = {}
            model_payload = {}
            metrics_payload = {}
            trace_payload = {}
            result = "transient-failure"
            failure_class = "smoke-failure"
            error_summary = f"{type(error).__name__}: {error}"
        probe = await asyncio.to_thread(probe_environment)
        detected = probe["detected"]
        evidence = {
            "hardware_profile": probe["configured"]["profile_id"],
            "fedora_version": detected.get("fedora_release"),
            "kernel": detected.get("kernel"),
            "gpu": health_payload.get("device_name"),
            "gpu_architecture": probe["configured"].get("gpu_architecture"),
            "rocm_version": health_payload.get("rocm_version"),
            "torch_version": metrics_payload.get("torch_version"),
            "transformers_version": metrics_payload.get("transformers_version"),
            "vllm_version": None,
            "model_id": model_payload.get("model_id", worker["model_id"]),
            "model_revision": model_payload.get("revision"),
            "quantisation": model_payload.get("quantization", "none"),
            "dtype": model_payload.get("dtype"),
            "runtime": worker["runtime"],
            "environment_overrides": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LD_PRELOAD": None,
            },
            "load_result": "success" if health_payload.get("ready") else "not-confirmed",
            "warmup_result": "success" if health_payload.get("ready") else "not-confirmed",
            "smoke_result": "success" if result == "tested-working" else "failed",
            "cold_load_seconds": metrics_payload.get("load_seconds"),
            "first_output_seconds": (
                trace_payload.get("metrics", {}).get("first_token_seconds")
                or trace_payload.get("metrics", {}).get("first_output_seconds")
            ),
            "throughput_tokens_per_second": trace_payload.get("metrics", {}).get("tokens_per_second"),
            "peak_memory_bytes": metrics_payload.get("peak_memory_allocated_bytes"),
            "steady_memory_bytes": metrics_payload.get("memory_allocated_bytes"),
            "shutdown_result": "not-tested",
            "memory_recovery_result": "not-tested",
            "test_duration_seconds": round(asyncio.get_running_loop().time() - started, 4),
            "error_summary": error_summary,
        }
        record = request.app.state.compatibility_store.record_test(
            evidence,
            result=result,
            failure_class=failure_class,
        )
        return {"ok": result == "tested-working", "worker_id": worker_id, "test": record}

    @app.get("/api/workers/{worker_id}/logs")
    async def worker_logs(worker_id: str, request: Request):
        return {"logs": _worker_call(request, "logs", worker_id)}

    @app.get("/api/events")
    async def events(request: Request):
        supervisor = request.app.state.supervisor

        async def stream() -> AsyncIterator[str]:
            for event in supervisor.event_history():
                yield f"event: worker\ndata: {json.dumps(event)}\n\n"
            while True:
                event = await supervisor.next_event()
                yield f"event: worker\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/workers/{worker_id}/logs/stream")
    async def worker_log_stream(worker_id: str, request: Request):
        _worker_call(request, "get_worker", worker_id)

        async def stream() -> AsyncIterator[str]:
            sent = 0
            session_id = None
            while True:
                logs = request.app.state.supervisor.logs(worker_id)
                current_session_id = logs[0].get("session_id") if logs else None
                if current_session_id != session_id or sent > len(logs):
                    sent = 0
                    session_id = current_session_id
                for item in logs[sent:]:
                    yield f"event: log\ndata: {json.dumps(item)}\n\n"
                sent = len(logs)
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/compatibility")
    async def compatibility(request: Request):
        return {"tests": request.app.state.compatibility_store.list_tests()}

    @app.put("/api/compatibility/tests/{test_id}/lifecycle")
    async def compatibility_lifecycle(test_id: int, payload: LifecycleEvidence, request: Request):
        _require_configuration_mutable(request)
        try:
            return request.app.state.compatibility_store.update_test_evidence(
                test_id, payload.model_dump(exclude_none=True)
            )
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/presets")
    async def presets():
        return [
            {"id": "open-day-minimum", "workers": ["mock-ar", "mock-diffusion"], "fallback": "mock"},
            {"id": "stop-all", "workers": [], "fallback": "structured-unavailable"},
        ]

    @app.post("/api/presets/stop-all")
    async def stop_all(request: Request):
        await request.app.state.supervisor.stop_all()
        return {"ok": True}

    @app.get("/{client_path:path}", include_in_schema=False)
    async def frontend_route(client_path: str):
        if client_path == "api" or client_path.startswith("api/"):
            raise HTTPException(404, "Unknown management API route")
        return _frontend_index()

    return app


def _require_configuration_mutable(request: Request) -> None:
    if request.app.state.settings.open_day:
        raise HTTPException(
            409,
            "Open Day mode locks deployment, route, provider and compatibility configuration",
        )


def _demo_set_response(record: dict) -> dict:
    return {
        **record["definition"],
        "revision": record["revision"],
        "updated_at": record["updated_at"],
        "active": record["active"],
        "active_revision": record["active_revision"],
    }


def _activate_demo_set_record(demo_set_id: str, record: dict, request: Request) -> dict:
    definition = DemoSetDefinition.model_validate(record["definition"])
    validation = _validate_demo_definition(definition, request)
    if not validation["valid"]:
        raise HTTPException(409, {"message": "Demo set validation failed", **validation})
    activated = request.app.state.compatibility_store.activate_demo_set(
        demo_set_id,
        record["revision"],
        routing_snapshot(definition, record["revision"]),
    )
    return {**activated, "plan": _demo_set_plan(definition, request)}


def _demo_route_record(demo_set_id: str, route_id: str, request: Request) -> tuple[dict, DemoRouteContract]:
    record = request.app.state.compatibility_store.get_demo_set(demo_set_id)
    if record is None:
        raise HTTPException(404, "Unknown demo set")
    definition = DemoSetDefinition.model_validate(record["definition"])
    route = next((candidate for candidate in definition.routes if candidate.id == route_id), None)
    if route is None:
        raise HTTPException(404, "Unknown demo route")
    return record, route


def _demo_route_smoke_request(route: DemoRouteContract) -> tuple[str, dict]:
    common_model = {"model": route.public_model}
    if route.adapter_id == "openai-chat-v1":
        return "/v1/chat/completions", {
            **common_model,
            "messages": [{"role": "user", "content": "Reply with the word ready."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        }
    if route.adapter_id == "openai-completions-v1":
        return "/v1/completions", {
            **common_model,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        }
    if route.adapter_id == "native-ar-trace-v1":
        return "/native/autoregressive/trace", {
            **common_model,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "top_k": 3,
            "seed": 7,
        }
    if route.adapter_id == "text-diffusion-v1":
        return "/v1/refine", {
            **common_model,
            "prompt": "A local worker is ready.",
            "denoising_steps": 4,
            "seed": 7,
        }
    if route.adapter_id == "scene-analysis-v1":
        from modeldeck.contracts.scenechat import external_prompt

        pixel = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
            "AAAAASUVORK5CYII="
        )
        return "/v1/vision/analyse", {
            **common_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": pixel}},
                        {"type": "text", "text": external_prompt("Describe the scene.")},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
    raise HTTPException(
        409,
        "Speech conversation rehearsal requires an interactive WebSocket client",
    )


async def _smoke_demo_route(route: DemoRouteContract, settings: Settings) -> dict:
    path, payload = _demo_route_smoke_request(route)
    started = asyncio.get_running_loop().time()
    timeout_seconds = max(60.0, settings.scenechat_timeout_seconds)
    if route.adapter_id == "text-diffusion-v1":
        timeout_seconds = settings.diffusion_timeout_seconds
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"http://{settings.host}:{settings.gateway_port}{path}", json=payload
            )
        if not response.is_success:
            try:
                body = response.json()
                code = body.get("error", {}).get("code") or body.get("detail")
            except ValueError:
                code = None
            raise HTTPException(
                response.status_code,
                f"Gateway route rehearsal failed{f': {code}' if code else ''}",
            )
        body = response.json()
    except httpx.HTTPError as error:
        raise HTTPException(503, "The local ModelDeck gateway is unavailable") from error
    evidence_fields = {
        "openai-chat-v1": "choices",
        "openai-completions-v1": "choices",
        "native-ar-trace-v1": "events",
        "text-diffusion-v1": "frames",
        "scene-analysis-v1": "choices",
    }
    evidence_field = evidence_fields[route.adapter_id]
    if not body.get(evidence_field):
        raise HTTPException(502, "Gateway route rehearsal returned no generation evidence")
    return {
        "ok": True,
        "route_id": route.id,
        "public_model": route.public_model,
        "adapter_id": route.adapter_id,
        "provider": response.headers.get("x-modeldeck-provider"),
        "evidence": evidence_field,
        "duration_seconds": round(asyncio.get_running_loop().time() - started, 4),
    }


def _validate_demo_definition(definition: DemoSetDefinition, request: Request) -> dict:
    profiles = request.app.state.profiles
    supervisor = request.app.state.supervisor
    policy = request.app.state.compatibility_store.list_model_cache_policy()
    return validate_demo_set(
        definition,
        profiles,
        registered_ids=set(supervisor.workers),
        allowed_ids={profile.id for profile in profiles if profile_allowed(profile, policy)},
        compatibility_tests=request.app.state.compatibility_store.list_tests(),
    )


def _demo_set_plan(definition: DemoSetDefinition, request: Request) -> dict:
    supervisor = request.app.state.supervisor
    profiles = {profile.id: profile for profile in request.app.state.profiles}
    desired = []
    for route in definition.routes:
        if not route.providers:
            continue
        provider_id = min(route.providers, key=lambda item: (item.priority, item.deployment_id)).deployment_id
        if provider_id not in desired:
            desired.append(provider_id)
    states = {
        deployment_id: supervisor.get_worker(deployment_id)["state"]
        for deployment_id in desired
        if deployment_id in supervisor.workers
    }
    start_required = [
        deployment_id for deployment_id in desired if states.get(deployment_id) not in {"ready", "busy"}
    ]
    desired_exclusive = [
        deployment_id
        for deployment_id in desired
        if deployment_id in profiles and profiles[deployment_id].lifecycle.value == "exclusive"
    ]
    active_exclusive = [
        worker["id"]
        for worker in supervisor.list_workers()
        if worker["lifecycle"] == "exclusive" and worker["state"] not in {"stopped", "failed", "incompatible"}
    ]
    warnings = []
    if len(desired_exclusive) > 1:
        warnings.append(
            "The demo set selects multiple exclusive primary deployments; they cannot all run concurrently"
        )
    return {
        "desired_primary_deployments": desired,
        "start_required": start_required,
        "stop_required": [
            deployment_id for deployment_id in active_exclusive if deployment_id not in desired_exclusive
        ],
        "warnings": warnings,
        "applies_process_changes": False,
    }


def _frontend_index() -> FileResponse | HTMLResponse:
    index = FRONTEND_ROOT / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(FRONTEND_FALLBACK, status_code=503)


async def _gateway_status(settings: Settings) -> dict:
    base_url = f"http://{settings.host}:{settings.gateway_port}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.4)) as client:
            health_response, models_response, providers_response = await asyncio.gather(
                client.get(f"{base_url}/v1/health"),
                client.get(f"{base_url}/v1/models"),
                client.get(f"{base_url}/v1/providers"),
            )
            for response in (health_response, models_response, providers_response):
                response.raise_for_status()
        return {
            "available": True,
            "health": health_response.json(),
            "models": models_response.json(),
            "providers": providers_response.json(),
            "error": None,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "available": False,
            "health": None,
            "models": None,
            "providers": None,
            "error": "The local ModelDeck gateway is unavailable.",
        }


def _worker_call(request: Request, method: str, worker_id: str):
    try:
        return getattr(request.app.state.supervisor, method)(worker_id)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error


async def _worker_async_call(request: Request, method: str, worker_id: str):
    try:
        return await getattr(request.app.state.supervisor, method)(worker_id)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


app = create_app()
