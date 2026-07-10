from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from modeldeck.api.dashboard import DASHBOARD_HTML
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.hardware import probe_environment
from modeldeck.profiles import default_model_profiles
from modeldeck.supervisor import WorkerSupervisor


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    profiles = default_model_profiles()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured.data_dir.mkdir(parents=True, exist_ok=True)
        store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")
        store.initialise()
        app.state.compatibility_store = store
        yield
        await app.state.supervisor.stop_all()

    app = FastAPI(
        title="ModelDeck management API",
        version="0.1.0",
        description="Local-only management for isolated model workers.",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.supervisor = WorkerSupervisor(profiles)
    app.state.profiles = profiles

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> str:
        return DASHBOARD_HTML

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
    async def catalogue():
        return {"models": await asyncio.to_thread(discover_huggingface_models), "downloads_started": False}

    @app.get("/api/profiles")
    async def list_profiles(request: Request):
        return [profile.model_dump(mode="json") for profile in request.app.state.profiles]

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
        return {"ok": True, "worker_id": worker_id, "result": "mock-smoke-passed"}

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
            while True:
                logs = request.app.state.supervisor.logs(worker_id)
                for item in logs[sent:]:
                    yield f"event: log\ndata: {json.dumps(item)}\n\n"
                sent = len(logs)
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/compatibility")
    async def compatibility(request: Request):
        return {"tests": request.app.state.compatibility_store.list_tests()}

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

    return app


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
