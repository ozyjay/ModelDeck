from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from modeldeck import __version__
from modeldeck.async_execution import run_in_isolated_thread
from modeldeck.benchmark_history import read_benchmark_history
from modeldeck.build_info import BUILD_ID
from modeldeck.capabilities import (
    capability_evidence_status,
    compatible_runtime_template_ids,
    worker_cache_identity,
)
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.compatibility import CompatibilityStore, LegacyDatabaseError
from modeldeck.config import Settings, gateway_base_url, state_store_metadata
from modeldeck.domain import WorkerDefinition
from modeldeck.hardware import probe_environment
from modeldeck.qwen_candidates import approve_candidate
from modeldeck.registry import runtime_template_registrations
from modeldeck.supervisor import WorkerSupervisor
from modeldeck.thermal import ThermalPolicyManager
from modeldeck.v2_api import create_v3_router

FRONTEND_ROOT = Path(__file__).parent / "api/static"
LOGGER = logging.getLogger(__name__)
FRONTEND_FALLBACK = """<!doctype html><html lang="en-AU"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ModelDeck</title></head>
<body><main><h1>ModelDeck operator console is not built</h1>
<p>Run <code>pwsh -NoProfile -File scripts/operations/build_frontend.ps1</code> and restart ModelDeck.</p>
</main></body></html>"""


class ModelCachePolicyRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    allowed: bool


class ModelCapabilityPolicyRequest(ModelCachePolicyRequest):
    capability_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")


class CandidateApprovalRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")


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


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()

    def discover_models():
        return discover_huggingface_models(data_dir=configured.data_dir)

    configured.data_dir.mkdir(parents=True, exist_ok=True)
    store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")
    store.initialise_v5()
    definitions: dict[str, WorkerDefinition] = {}
    worker_profiles = []
    for record in store.list_workers():
        try:
            definition = WorkerDefinition.model_validate(record["definition"])
            profile = definition.to_profile()
        except ValueError as error:
            # Keep historical Worker and evidence records intact, but never expose or
            # launch a runtime that is no longer trusted by the production gateway.
            LOGGER.warning(
                "Ignoring persisted Worker %s because its runtime is not trusted: %s",
                record["definition"].get("id", "unknown"),
                error,
            )
            continue
        definitions[definition.id] = definition
        worker_profiles.append(profile)
    thermal_manager = ThermalPolicyManager(
        configured.thermal_throttling,
        data_dir=configured.data_dir,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.thermal_manager.start()
        await app.state.reconcile_capability_setups(app)
        try:
            yield
        finally:
            await app.state.shutdown_capability_setups()
            await app.state.supervisor.stop_all()
            await app.state.thermal_manager.stop()

    app = FastAPI(
        title="ModelDeck management API",
        version=__version__,
        description="Local-only management for Routing Profiles, capabilities and isolated model Workers.",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.thermal_manager = thermal_manager
    app.state.compatibility_store = store
    app.state.worker_definitions = definitions
    app.state.supervisor = WorkerSupervisor(
        worker_profiles,
        log_dir=configured.log_dir,
        thermal_manager=thermal_manager,
        data_dir=configured.data_dir,
    )
    thermal_manager.critical_handler = app.state.supervisor.critical_stop_all
    app.state.runtime_registrations = runtime_template_registrations(configured.data_dir)
    app.state.discover_models = discover_models

    assets = FRONTEND_ROOT / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")
    management_router = create_v3_router()
    app.state.reconcile_capability_setups = management_router.reconcile_capability_setups
    app.state.shutdown_capability_setups = management_router.shutdown_capability_setups
    app.include_router(management_router)

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
    async def health():
        return {
            "status": "ok",
            "service": "modeldeck-management",
            "version": __version__,
            "build_id": BUILD_ID,
            "schema_version": 5,
            "configuration_locked": configured.configuration_locked,
            "offline_only": True,
            "gateway_url": gateway_base_url(configured.gateway_host, configured.gateway_port),
            "state_store": state_store_metadata(configured.data_dir),
        }

    @app.get("/api/gateway/status")
    async def gateway_status():
        return await _gateway_status(configured)

    @app.get("/api/hardware")
    async def hardware():
        return await run_in_isolated_thread(probe_environment)

    @app.get("/api/telemetry")
    async def telemetry():
        probe = await run_in_isolated_thread(probe_environment)
        detected = probe["detected"]
        return {
            key: detected[key]
            for key in (
                "memory",
                "swap",
                "filesystems",
                "temperatures",
                "fans",
                "active_model_processes",
            )
        }

    @app.get("/api/benchmark-history")
    async def benchmark_history():
        return await run_in_isolated_thread(read_benchmark_history, Path("var/benchmarks"))

    @app.get("/api/thermal")
    async def thermal_status(request: Request):
        return {
            **request.app.state.thermal_manager.status(),
            "policy": asdict(request.app.state.settings.thermal_throttling),
        }

    @app.get("/api/catalogue")
    async def catalogue(request: Request):
        models = await run_in_isolated_thread(discover_models)
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        capability_policy = request.app.state.compatibility_store.list_model_capability_policy()
        tests = request.app.state.compatibility_store.list_tests()
        active_snapshots = request.app.state.compatibility_store.active_routing_snapshots()

        def response(model):
            allowed = policy.get((model["model_id"], model["revision"]), True)
            model_workers = [
                definition
                for definition in request.app.state.worker_definitions.values()
                if worker_cache_identity(definition.model_dump(mode="json"))
                == (model["model_id"], model["revision"])
            ]
            potential_capabilities = []
            for candidate in model.get("potential_capabilities", []):
                stored_allowed = capability_policy.get(
                    (model["model_id"], model["revision"], candidate["id"]), False
                )
                effective_allowed = allowed and stored_allowed
                available_templates = compatible_runtime_template_ids(
                    candidate["id"],
                    model.get("configuration_support"),
                    request.app.state.runtime_registrations,
                )
                qualifying_workers = []
                statuses = []
                for worker in model_workers:
                    if worker.runtime_template_id not in available_templates:
                        continue
                    status, evidence_id = capability_evidence_status(
                        worker.model_dump(mode="json"), candidate["id"], tests
                    )
                    statuses.append(status)
                    qualifying_workers.append(
                        {
                            "worker_id": worker.id,
                            "worker_name": worker.name,
                            "evidence_id": evidence_id,
                            "status": status,
                        }
                    )
                qualification_status = next(
                    (status for status in ("qualified", "legacy", "failed", "stale") if status in statuses),
                    "not-tested",
                )
                published = any(
                    route.get("protocol_contract") == candidate.get("protocol_contract_id")
                    and any(worker.id in route.get("worker_ids", []) for worker in model_workers)
                    for snapshot in active_snapshots
                    for route in snapshot.get("capabilities", [])
                )
                creatable = bool(
                    model["download_state"] == "installed-untested"
                    and effective_allowed
                    and available_templates
                )
                reason = (
                    "Ready to create a Worker with an installed trusted runtime."
                    if creatable
                    else "This cached Model is disallowed in ModelDeck."
                    if not allowed
                    else "Allow this capability before creating a Worker or publishing a route."
                    if not stored_allowed
                    else "Allowed; a trusted runtime is required."
                    if not available_templates
                    else "Finish the local snapshot before creating a Worker."
                )
                potential_capabilities.append(
                    {
                        **candidate,
                        "runtime_template_ids": available_templates,
                        "policy_allowed": stored_allowed,
                        "effective_allowed": effective_allowed,
                        "available_runtime_template_ids": available_templates,
                        "runtime_status": "available" if available_templates else "missing",
                        "qualification_status": qualification_status,
                        "qualifying_workers": qualifying_workers,
                        "published": published,
                        "creatable": creatable,
                        "reason": reason,
                    }
                )
            runnable = any(item["creatable"] for item in potential_capabilities)
            return {
                **model,
                "potential_capabilities": potential_capabilities,
                "modeldeck_allowed": allowed,
                "runnable": runnable,
                "runnable_reason": (
                    "Ready to create a Worker with an installed trusted runtime."
                    if runnable
                    else "This cached Model is disallowed in ModelDeck."
                    if not allowed
                    else "Allow at least one runnable capability before creating a Worker."
                    if any(item["runtime_status"] == "available" for item in potential_capabilities)
                    else model.get("configuration_support_reason")
                    or "No installed trusted runtime recognises this Model."
                ),
                "worker_count": len(model_workers),
            }

        return {
            "models": [response(model) for model in models],
            "downloads_started": False,
        }

    @app.post("/api/catalogue/capabilities/policy")
    async def set_catalogue_capability_policy(payload: ModelCapabilityPolicyRequest, request: Request):
        _require_mutable(request)
        cached = next(
            (
                model
                for model in discover_models()
                if model["model_id"] == payload.model_id
                and model["revision"] == payload.revision
                and model["download_state"] == "installed-untested"
            ),
            None,
        )
        if cached is None:
            raise HTTPException(404, "The exact complete cached Model revision was not discovered")
        candidate = next(
            (
                item
                for item in cached.get("potential_capabilities", [])
                if item["id"] == payload.capability_id
            ),
            None,
        )
        if candidate is None:
            raise HTTPException(409, "That capability is not a recognised candidate for this Model")
        references = (
            _capability_policy_references(request, payload.model_id, payload.revision, candidate)
            if not payload.allowed
            else []
        )
        if references:
            raise HTTPException(
                409,
                {
                    "message": "Remove this capability from current Routing Profiles before disallowing it",
                    "references": references,
                },
            )
        request.app.state.compatibility_store.set_model_capability_allowed(
            payload.model_id,
            payload.revision,
            payload.capability_id,
            allowed=payload.allowed,
        )
        return {
            "ok": True,
            "model_id": payload.model_id,
            "revision": payload.revision,
            "capability_id": payload.capability_id,
            "allowed": payload.allowed,
        }

    @app.post("/api/catalogue/candidates/approve")
    async def approve_catalogue_candidate(payload: CandidateApprovalRequest, request: Request):
        _require_mutable(request)
        cached = next(
            (
                model
                for model in discover_models()
                if model["model_id"] == payload.model_id
                and model["revision"] == payload.revision
                and model["download_state"] == "installed-untested"
            ),
            None,
        )
        if cached is None or cached.get("snapshot_location") is None:
            raise HTTPException(404, "The exact complete cached Model revision was not discovered")
        registration = cached.get("candidate_registration")
        if not isinstance(registration, dict) or not registration.get("eligible"):
            reason = registration.get("reason") if isinstance(registration, dict) else None
            raise HTTPException(409, reason or "This Model is not eligible for local candidate approval")
        try:
            manifest = await run_in_isolated_thread(
                approve_candidate,
                payload.model_id,
                payload.revision,
                Path(cached["snapshot_location"]),
                data_dir=configured.data_dir,
            )
        except (OSError, ValueError) as error:
            raise HTTPException(409, str(error)) from error
        return {
            "ok": True,
            "candidate_id": manifest.id,
            "model_id": manifest.artefact_model_id,
            "revision": manifest.artefact_revision,
            "sha256": manifest.model.sha256,
        }

    @app.post("/api/catalogue/policy")
    async def set_catalogue_policy(payload: ModelCachePolicyRequest, request: Request):
        _require_mutable(request)
        cached = next(
            (
                model
                for model in discover_models()
                if model["model_id"] == payload.model_id and model["revision"] == payload.revision
            ),
            None,
        )
        if cached is None:
            raise HTTPException(404, "The exact cached Model revision was not discovered")
        configured_workers = [
            definition
            for definition in request.app.state.worker_definitions.values()
            if (definition.model_id, definition.revision) == (payload.model_id, payload.revision)
            or (definition.artifact_model_id, definition.artifact_revision)
            == (payload.model_id, payload.revision)
        ]
        if not payload.allowed and configured_workers:
            raise HTTPException(
                409,
                {
                    "message": "Archive this Model's Workers before disallowing it",
                    "workers": [{"id": worker.id, "name": worker.name} for worker in configured_workers],
                },
            )
        request.app.state.compatibility_store.set_model_cache_allowed(
            payload.model_id, payload.revision, allowed=payload.allowed
        )
        return {
            "ok": True,
            "model_id": payload.model_id,
            "revision": payload.revision,
            "allowed": payload.allowed,
            "cache_removed": False,
        }

    @app.get("/api/runtime-templates")
    async def runtime_templates(request: Request):
        return {
            "templates": [
                {
                    "id": registration.template.id,
                    "display_name": registration.template.display_name,
                    "implementation": registration.template.runtime,
                    "generation_family": registration.template.generation_family,
                    "cache_setting": registration.template.cache_setting,
                    "uses_base_model_identity": registration.template.uses_base_model_identity,
                    "lifecycle": registration.template.lifecycle,
                    "dtype": registration.template.dtype,
                    "settings": registration.template.settings,
                    "package_id": registration.package.id,
                    "package_version": registration.package.version,
                    "package_display_name": registration.package.display_name,
                    "publisher": registration.package.publisher,
                    "source": registration.source,
                    "digest": registration.digest,
                }
                for registration in request.app.state.runtime_registrations.values()
            ],
            "installation": "local-admin-only",
        }

    @app.get("/api/compatibility")
    async def compatibility(request: Request):
        return {"tests": request.app.state.compatibility_store.list_tests()}

    @app.post("/api/compatibility/tests/{test_id}/observations", status_code=201)
    async def compatibility_observation(test_id: int, payload: LifecycleEvidence, request: Request):
        _require_mutable(request)
        try:
            return request.app.state.compatibility_store.record_test_observation(
                test_id, payload.model_dump(exclude_none=True), kind="lifecycle"
            )
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/workers/{worker_id}/logs")
    async def worker_logs(worker_id: str, request: Request):
        try:
            return {"logs": request.app.state.supervisor.logs(worker_id)}
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/api/workers/{worker_id}/logs/stream")
    async def worker_log_stream(worker_id: str, request: Request):
        try:
            request.app.state.supervisor.get_worker(worker_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

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

    @app.get("/api/worker-events")
    async def worker_events(request: Request):
        supervisor = request.app.state.supervisor

        async def stream() -> AsyncIterator[str]:
            for event in supervisor.event_history():
                yield f"event: worker\ndata: {json.dumps(event)}\n\n"
            while True:
                event = await supervisor.next_event()
                yield f"event: worker\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/workers/stop-all")
    async def stop_all(request: Request):
        await request.app.state.supervisor.stop_all()
        return {"ok": True}

    @app.get("/{client_path:path}", include_in_schema=False)
    async def frontend_route(client_path: str):
        if client_path == "api" or client_path.startswith("api/"):
            raise HTTPException(404, "Unknown management API route")
        return _frontend_index()

    return app


def _require_mutable(request: Request) -> None:
    if request.app.state.settings.configuration_locked:
        raise HTTPException(423, "Configuration is locked by the local deployment policy")


def _capability_policy_references(
    request: Request,
    model_id: str,
    revision: str,
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    contract = candidate.get("protocol_contract_id")
    if not isinstance(contract, str):
        return []
    matching_worker_ids = {
        worker.id
        for worker in request.app.state.worker_definitions.values()
        if worker_cache_identity(worker.model_dump(mode="json")) == (model_id, revision)
    }
    references: list[dict[str, object]] = []
    store = request.app.state.compatibility_store
    for profile in store.list_routing_profiles():
        definition = profile["definition"]
        for binding in definition.get("capabilities", []):
            if binding.get("protocol_contract") == contract and matching_worker_ids.intersection(
                binding.get("worker_ids", [])
            ):
                references.append(
                    {
                        "profile_id": definition["id"],
                        "profile_name": definition["name"],
                        "capability_id": binding["id"],
                        "capability_name": binding["display_name"],
                        "kind": "draft",
                    }
                )
        if not profile["active"] or profile["active_revision"] is None:
            continue
        active = store.get_routing_profile_revision(definition["id"], profile["active_revision"])
        if active is None:
            continue
        for binding in active["definition"].get("capabilities", []):
            if binding.get("protocol_contract") == contract and matching_worker_ids.intersection(
                binding.get("worker_ids", [])
            ):
                references.append(
                    {
                        "profile_id": definition["id"],
                        "profile_name": definition["name"],
                        "capability_id": binding["id"],
                        "capability_name": binding["display_name"],
                        "kind": "active",
                        "revision": profile["active_revision"],
                    }
                )
    unique = {
        (reference["profile_id"], reference["capability_id"], reference["kind"]): reference
        for reference in references
    }
    return list(unique.values())


def _frontend_index() -> FileResponse | HTMLResponse:
    index = FRONTEND_ROOT / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(FRONTEND_FALLBACK, status_code=503)


async def _gateway_status(settings: Settings) -> dict:
    base_url = gateway_base_url(settings.gateway_host, settings.gateway_port)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.4)) as client:
            health_response, models_response, routes_response = await asyncio.gather(
                client.get(f"{base_url}/v1/health"),
                client.get(f"{base_url}/v1/models"),
                client.get(f"{base_url}/v1/routes"),
            )
            for response in (health_response, models_response, routes_response):
                response.raise_for_status()
        return {
            "available": True,
            "health": health_response.json(),
            "models": models_response.json(),
            "routes": routes_response.json(),
            "error": None,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "available": False,
            "health": None,
            "models": None,
            "routes": None,
            "error": "The local ModelDeck gateway is unavailable.",
        }


try:
    app = create_app()
except LegacyDatabaseError as startup_error:
    startup_error_message = str(startup_error)
    app = FastAPI(title="ModelDeck database upgrade required", version=__version__)

    @app.get("/api/health", status_code=503)
    async def database_upgrade_required():
        return {"status": "upgrade-required", "detail": startup_error_message}
