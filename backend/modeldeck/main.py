from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from modeldeck.api.dashboard import DASHBOARD_HTML
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.hardware import probe_environment
from modeldeck.profiles import default_model_profiles
from modeldeck.supervisor import WorkerSupervisor


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
        endpoint = worker["endpoint"]
        started = asyncio.get_running_loop().time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                health_response, model_response, metrics_response = await asyncio.gather(
                    client.get(f"{endpoint}/health"),
                    client.get(f"{endpoint}/model"),
                    client.get(f"{endpoint}/metrics"),
                )
                if worker["generation_family"] == "autoregressive":
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
            smoke_events = trace_payload.get("events") or trace_payload.get("frames")
            if not smoke_events:
                raise RuntimeError("Smoke request returned no generation events")
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
            "quantisation": "none",
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
            "first_output_seconds": trace_payload.get("metrics", {}).get("first_token_seconds"),
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

    @app.put("/api/compatibility/tests/{test_id}/lifecycle")
    async def compatibility_lifecycle(test_id: int, payload: LifecycleEvidence, request: Request):
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
