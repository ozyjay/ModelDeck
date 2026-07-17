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
from modeldeck.hardware import probe_environment
from modeldeck.profile_registry import (
    load_local_profiles,
    profile_allowed,
    profile_cache_identity,
    profile_uses_huggingface_cache,
)
from modeldeck.profiles import (
    LOCAL_PORT_RANGE,
    RESERVED_GATEWAY_ALIASES,
    LocalProfileRequest,
    create_local_profile,
    default_model_profiles,
)
from modeldeck.provider_selection import (
    DEFAULT_SCENECHAT_PROVIDER_ID,
    SCENECHAT_ALIAS,
    scenechat_provider_compatible,
)
from modeldeck.q4_release import Q4ReleaseError, verify_modeldeck_q4_release
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

    async def scenechat_selection_status(request: Request) -> dict:
        store = request.app.state.compatibility_store
        stored_selection = store.gateway_provider_selection(SCENECHAT_ALIAS)
        selected_provider = stored_selection or DEFAULT_SCENECHAT_PROVIDER_ID
        policy = store.list_model_cache_policy()
        supervisor = request.app.state.supervisor
        gateway = await _gateway_status(request.app.state.settings)
        gateway_model = next(
            (
                model
                for model in (gateway.get("models") or {}).get("data", [])
                if model.get("id") == SCENECHAT_ALIAS
            ),
            {},
        )
        effective_provider = gateway_model.get("effective_provider")
        candidates = []
        for profile in request.app.state.profiles:
            if (
                profile.id not in supervisor.workers
                or not scenechat_provider_compatible(profile)
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
                }
            )
        candidates.sort(
            key=lambda candidate: (
                candidate["profile_id"] != DEFAULT_SCENECHAT_PROVIDER_ID,
                candidate["profile_id"],
            )
        )
        return {
            "alias": SCENECHAT_ALIAS,
            "default_provider": DEFAULT_SCENECHAT_PROVIDER_ID,
            "explicit_selection": stored_selection is not None,
            "selected_provider": selected_provider,
            "effective_provider": effective_provider,
            "gateway_ready": gateway_model.get("ready") is True,
            "candidates": candidates,
        }

    @app.get("/api/gateway/provider-selections/scenechat-vision")
    async def get_scenechat_selection(request: Request):
        return await scenechat_selection_status(request)

    @app.post("/api/gateway/provider-selections/scenechat-vision")
    async def set_scenechat_selection(
        payload: GatewayProviderSelectionRequest,
        request: Request,
    ):
        profile = next(
            (candidate for candidate in request.app.state.profiles if candidate.id == payload.profile_id),
            None,
        )
        if profile is None or profile.id not in request.app.state.supervisor.workers:
            raise HTTPException(409, "Select an existing registered ModelDeck runtime profile")
        if not scenechat_provider_compatible(profile):
            raise HTTPException(
                409,
                "The selected profile must be a vision-language runtime with image input "
                "and structured output",
            )
        policy = request.app.state.compatibility_store.list_model_cache_policy()
        if not profile_allowed(profile, policy):
            raise HTTPException(409, "Allow the selected cached model in ModelDeck first")
        request.app.state.compatibility_store.set_gateway_provider_selection(
            SCENECHAT_ALIAS,
            profile.id,
        )
        return await scenechat_selection_status(request)

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
        selected_scenechat = (
            request.app.state.compatibility_store.gateway_provider_selection(SCENECHAT_ALIAS)
            or DEFAULT_SCENECHAT_PROVIDER_ID
        )
        if not payload.allowed and any(profile.id == selected_scenechat for profile in matching_profiles):
            raise HTTPException(
                409,
                "Select a different scenechat-vision provider before disallowing this model",
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

    @app.post("/api/profiles", status_code=201)
    async def create_profile(payload: LocalProfileRequest, request: Request):
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
        if checkpoint_dir is not None:
            try:
                await _run_blocking(verify_modeldeck_q4_release, checkpoint_dir)
            except (OSError, Q4ReleaseError) as error:
                raise HTTPException(409, f"ModelDeck Q4 release verification failed: {error}") from error

        profiles = request.app.state.profiles
        profile_id = f"local-{payload.alias}"
        if payload.alias in RESERVED_GATEWAY_ALIASES or any(
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
        )
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
        if profile_id in request.app.state.built_in_profile_ids:
            raise HTTPException(409, "Built-in runtime configurations cannot be removed")
        profile = next(
            (candidate for candidate in request.app.state.profiles if candidate.id == profile_id),
            None,
        )
        if profile is None:
            raise HTTPException(404, "Unknown local runtime configuration")
        if request.app.state.compatibility_store.gateway_provider_selection(SCENECHAT_ALIAS) == profile_id:
            raise HTTPException(
                409,
                "Select a different scenechat-vision provider before removing this runtime configuration",
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
                elif worker["generation_family"] == "vision-language":
                    trace_response = await client.post(
                        f"{endpoint}/native/vision-language/smoke",
                        headers={
                            "Authorization": (
                                "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")
                            )
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
            smoke_evidence = (
                trace_payload.get("events") or trace_payload.get("frames") or trace_payload.get("ok")
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
