from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from modeldeck.catalogue import discover_huggingface_models
from modeldeck.compatibility import CompatibilityStore, LegacyDatabaseError
from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.hardware import probe_environment
from modeldeck.registry import runtime_template_registrations
from modeldeck.supervisor import WorkerSupervisor
from modeldeck.v2_api import create_v2_router

FRONTEND_ROOT = Path(__file__).parent / "api/static"
FRONTEND_FALLBACK = """<!doctype html><html lang="en-AU"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ModelDeck</title></head>
<body><main><h1>ModelDeck operator console is not built</h1>
<p>Run <code>pwsh -NoProfile -File scripts/build_frontend.ps1</code> and restart ModelDeck.</p>
</main></body></html>"""


class ModelCachePolicyRequest(BaseModel):
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    allowed: bool


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
    configured.data_dir.mkdir(parents=True, exist_ok=True)
    store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")
    store.initialise_v2()
    definitions = {
        worker.id: worker
        for worker in (
            WorkerDefinition.model_validate(record["definition"]) for record in store.list_workers()
        )
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.supervisor.stop_all()

    app = FastAPI(
        title="ModelDeck management API",
        version="0.2.0",
        description="Local-only management for Events, Routes and isolated model Workers.",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.compatibility_store = store
    app.state.worker_definitions = definitions
    app.state.supervisor = WorkerSupervisor(
        [definition.to_profile() for definition in definitions.values()],
        log_dir=configured.log_dir,
    )
    app.state.runtime_registrations = runtime_template_registrations(configured.data_dir)

    assets = FRONTEND_ROOT / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")
    app.include_router(create_v2_router())

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
            "schema_version": 2,
            "open_day": configured.open_day,
            "downloads_allowed": configured.allow_downloads,
            "gateway_url": f"http://{configured.host}:{configured.gateway_port}",
        }

    @app.get("/api/gateway/status")
    async def gateway_status():
        return await _gateway_status(configured)

    @app.get("/api/hardware")
    async def hardware():
        return await asyncio.to_thread(probe_environment)

    @app.get("/api/telemetry")
    async def telemetry():
        probe = await asyncio.to_thread(probe_environment)
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

    @app.get("/api/catalogue")
    async def catalogue(request: Request):
        models = await asyncio.to_thread(discover_huggingface_models)
        policy = request.app.state.compatibility_store.list_model_cache_policy()

        def response(model):
            allowed = policy.get((model["model_id"], model["revision"]), True)
            supported = bool(model.get("configuration_support"))
            return {
                **model,
                "modeldeck_allowed": allowed,
                "runnable": allowed and supported,
                "runnable_reason": (
                    "Ready to create a Worker with an installed trusted runtime."
                    if allowed and supported
                    else "This cached Model is disallowed in ModelDeck."
                    if not allowed
                    else model.get("configuration_support_reason")
                    or "No installed trusted runtime recognises this Model."
                ),
                "worker_count": sum(
                    definition.model_id == model["model_id"] and definition.revision == model["revision"]
                    for definition in request.app.state.worker_definitions.values()
                ),
            }

        return {
            "models": [response(model) for model in models],
            "downloads_started": False,
        }

    @app.post("/api/catalogue/policy")
    async def set_catalogue_policy(payload: ModelCachePolicyRequest, request: Request):
        _require_mutable(request)
        cached = next(
            (
                model
                for model in discover_huggingface_models()
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

    @app.put("/api/compatibility/tests/{test_id}/lifecycle")
    async def compatibility_lifecycle(test_id: int, payload: LifecycleEvidence, request: Request):
        _require_mutable(request)
        try:
            return request.app.state.compatibility_store.update_test_evidence(
                test_id, payload.model_dump(exclude_none=True)
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
    if request.app.state.settings.open_day:
        raise HTTPException(423, "Configuration is locked while Open Day mode is active")


def _frontend_index() -> FileResponse | HTMLResponse:
    index = FRONTEND_ROOT / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(FRONTEND_FALLBACK, status_code=503)


async def _gateway_status(settings: Settings) -> dict:
    base_url = f"http://{settings.host}:{settings.gateway_port}"
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
    app = FastAPI(title="ModelDeck database upgrade required", version="0.2.0")

    @app.get("/api/health", status_code=503)
    async def database_upgrade_required():
        return {"status": "upgrade-required", "detail": startup_error_message}
