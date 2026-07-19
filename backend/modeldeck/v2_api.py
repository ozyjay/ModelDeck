from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.catalogue import discover_huggingface_models
from modeldeck.domain import EventDefinition, WorkerDefinition, routing_snapshot, validate_event
from modeldeck.hardware import probe_environment
from modeldeck.profiles import LOCAL_PORT_RANGE, LocalProfileRequest, create_local_profile
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS
from modeldeck.q4_release import Q4ReleaseError, verify_modeldeck_q4_release


class WorkerCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    dtype: Literal["float16", "bfloat16"] = "float16"
    lifecycle: Literal["resident", "on-demand", "exclusive"] = "on-demand"
    context_length: int = Field(default=2048, ge=256, le=32768)
    maximum_new_tokens: int = Field(default=128, ge=1, le=512)
    maximum_denoising_steps: int = Field(default=24, ge=1, le=48)
    artifact_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")
    runtime_template_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")


class WorkerRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)


class WorkerReplacementRequest(WorkerCreateRequest):
    rebind_drafts: bool = True


def create_v2_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/protocol-contracts")
    async def protocol_contracts():
        return {
            "contracts": [
                {
                    "id": contract.id,
                    "display_name": contract.display_name,
                    "generation_family": contract.generation_family,
                    "required_capabilities": list(contract.required_capabilities),
                    "surfaces": list(contract.surfaces),
                }
                for contract in PROTOCOL_CONTRACTS.values()
            ]
        }

    @router.get("/workers")
    async def list_workers(request: Request):
        return [_worker_response(request, record) for record in _worker_records(request)]

    @router.post("/workers", status_code=201)
    async def create_worker(payload: WorkerCreateRequest, request: Request):
        _require_mutable(request)
        clean_name = " ".join(payload.name.split())
        if any(
            record["definition"]["name"].casefold() == clean_name.casefold()
            for record in _worker_records(request)
        ):
            raise HTTPException(409, "A Worker with that name already exists")
        cached = next(
            (
                model
                for model in discover_huggingface_models()
                if model["model_id"] == payload.model_id
                and model["revision"] == payload.revision
                and model["download_state"] == "installed-untested"
            ),
            None,
        )
        if cached is None:
            raise HTTPException(409, "The requested pinned snapshot is not complete in the local cache")
        store = request.app.state.compatibility_store
        if not store.model_cache_allowed(payload.model_id, payload.revision):
            raise HTTPException(409, "Allow this cached Model before creating a Worker")
        support = cached.get("configuration_support")
        template_id = payload.runtime_template_id or support
        registrations = request.app.state.runtime_registrations
        baseline = registrations.get(support) if support else None
        selected = registrations.get(template_id) if template_id else None
        if baseline is None or selected is None:
            raise HTTPException(409, "Select an installed trusted runtime")
        if (
            selected.template.generation_family != baseline.template.generation_family
            or selected.template.cache_setting != baseline.template.cache_setting
            or selected.template.uses_base_model_identity != baseline.template.uses_base_model_identity
        ):
            raise HTTPException(409, "The selected trusted runtime is incompatible with this Model")
        checkpoint_dir = (
            Path(cached["snapshot_location"])
            if selected.template.cache_setting == "q4_checkpoint_dir"
            else None
        )
        artefact_path = None
        if selected.template.cache_setting == "artifact_path":
            artefact = next(
                (item for item in cached.get("artifacts", []) if item["artifact_id"] == payload.artifact_id),
                None,
            )
            if artefact is None:
                raise HTTPException(409, "Select a discovered allowlisted artefact")
            artefact_path = Path(cached["snapshot_location"]) / artefact["filenames"][0]
        if checkpoint_dir is not None:
            try:
                await asyncio.to_thread(verify_modeldeck_q4_release, checkpoint_dir)
            except (OSError, Q4ReleaseError) as error:
                raise HTTPException(409, f"ModelDeck Q4 release verification failed: {error}") from error
        used_ports = {
            int(record["definition"]["port"]) for record in _worker_records(request, include_archived=True)
        }
        port = next((candidate for candidate in LOCAL_PORT_RANGE if candidate not in used_ports), None)
        if port is None:
            raise HTTPException(409, "No local ModelDeck Worker ports are available")
        worker_id = str(uuid4())
        internal_name = f"worker-{worker_id[:8]}"
        profile_request = LocalProfileRequest(
            model_id=payload.model_id,
            revision=payload.revision,
            alias=internal_name,
            profile_name=internal_name,
            dtype=payload.dtype,
            lifecycle=payload.lifecycle,
            context_length=payload.context_length,
            maximum_new_tokens=payload.maximum_new_tokens,
            maximum_denoising_steps=payload.maximum_denoising_steps,
            artifact_id=payload.artifact_id,
            runtime_template_id=payload.runtime_template_id,
        )
        cache_root = Path(cached["cache_location"]).parent
        profile = create_local_profile(
            profile_request,
            cache_root=cache_root,
            port=port,
            configuration_support=template_id,
            checkpoint_dir=checkpoint_dir,
            base_model_id=cached.get("base_model_id"),
            base_model_revision=cached.get("base_model_revision"),
            artifact_path=artefact_path,
            template_registrations=registrations,
        ).model_copy(update={"id": worker_id})
        definition = WorkerDefinition.from_profile(profile, name=clean_name)
        store.save_worker_definition(definition.model_dump(mode="json"))
        try:
            request.app.state.supervisor.register_profile(definition.to_profile())
        except ValueError as error:
            store.delete_worker_definition(worker_id)
            raise HTTPException(409, str(error)) from error
        request.app.state.worker_definitions[worker_id] = definition
        return _worker_response(request, store.get_worker_definition(worker_id))

    @router.get("/workers/{worker_id}")
    async def get_worker(worker_id: str, request: Request):
        record = request.app.state.compatibility_store.get_worker_definition(worker_id)
        if record is None:
            raise HTTPException(404, "Unknown Worker")
        return _worker_response(request, record)

    @router.patch("/workers/{worker_id}")
    async def rename_worker(worker_id: str, payload: WorkerRenameRequest, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        record = store.get_worker_definition(worker_id)
        if record is None:
            raise HTTPException(404, "Unknown Worker")
        definition = WorkerDefinition.model_validate(record["definition"])
        renamed = definition.model_copy(update={"name": " ".join(payload.name.split())})
        try:
            saved = store.save_worker_definition(renamed.model_dump(mode="json"))
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        request.app.state.worker_definitions[worker_id] = renamed
        return _worker_response(request, saved)

    @router.get("/workers/{worker_id}/usage")
    async def worker_usage(worker_id: str, request: Request):
        if request.app.state.compatibility_store.get_worker_definition(worker_id) is None:
            raise HTTPException(404, "Unknown Worker")
        return _worker_usage(worker_id, request)

    @router.post("/workers/{worker_id}/replacement", status_code=201)
    async def replace_worker(worker_id: str, payload: WorkerReplacementRequest, request: Request):
        _require_worker(request, worker_id)
        replacement = await create_worker(
            WorkerCreateRequest.model_validate(payload.model_dump(exclude={"rebind_drafts"})), request
        )
        rebound_events = []
        if payload.rebind_drafts:
            rebound_events = request.app.state.compatibility_store.rebind_event_drafts(
                worker_id, replacement["id"]
            )
        return {"replacement": replacement, "rebound_event_drafts": rebound_events}

    @router.delete("/workers/{worker_id}")
    async def archive_worker(worker_id: str, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        record = store.get_worker_definition(worker_id)
        if record is None:
            raise HTTPException(404, "Unknown Worker")
        usage = _worker_usage(worker_id, request)
        if not usage["archivable"]:
            raise HTTPException(409, {"message": "Reassign this Worker before archiving it", **usage})
        snapshot = request.app.state.supervisor.get_worker(worker_id)
        if snapshot["state"] not in {"stopped", "failed"}:
            raise HTTPException(409, "Stop the Worker before archiving it")
        await request.app.state.supervisor.remove_profile(worker_id)
        store.archive_worker(worker_id)
        request.app.state.worker_definitions.pop(worker_id, None)
        return {"ok": True, "worker_id": worker_id, "cache_removed": False}

    for operation in ("start", "stop", "restart"):
        _add_lifecycle_route(router, operation)

    @router.post("/workers/{worker_id}/smoke")
    async def smoke_worker(worker_id: str, request: Request):
        definition = _require_worker(request, worker_id)
        worker = request.app.state.supervisor.get_worker(worker_id)
        if worker["state"] != "ready":
            raise HTTPException(409, "Worker must be ready before smoke testing")
        started = asyncio.get_running_loop().time()
        health_payload = {}
        model_payload = {}
        metrics_payload = {}
        generation_payload = {}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                health_response, model_response, metrics_response = await asyncio.gather(
                    client.get(f"{worker['endpoint']}/health"),
                    client.get(f"{worker['endpoint']}/model"),
                    client.get(f"{worker['endpoint']}/metrics"),
                )
                path, body, headers = _worker_smoke_request(definition)
                generation_response = await client.post(
                    f"{worker['endpoint']}{path}", json=body, headers=headers
                )
                for response in (
                    health_response,
                    model_response,
                    metrics_response,
                    generation_response,
                ):
                    response.raise_for_status()
                health_payload = health_response.json()
                model_payload = model_response.json()
                metrics_payload = metrics_response.json()
                generation_payload = generation_response.json()
                if health_payload.get("ready") is not True:
                    raise RuntimeError("Worker health did not report ready")
                smoke_evidence = (
                    generation_payload.get("events")
                    or generation_payload.get("frames")
                    or generation_payload.get("ok")
                    or generation_payload.get("choices")
                )
                if not smoke_evidence:
                    raise RuntimeError("Smoke request returned no generation evidence")
            result = "tested-working"
            failure_class = None
            error_summary = None
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            result = "transient-failure"
            failure_class = "smoke-failure"
            error_summary = f"{type(error).__name__}: {error}"
        probe = await asyncio.to_thread(probe_environment)
        detected = probe["detected"]
        evidence = {
            "worker_id": definition.id,
            "hardware_profile": probe["configured"]["profile_id"],
            "fedora_version": detected.get("fedora_release"),
            "kernel": detected.get("kernel"),
            "gpu": health_payload.get("device_name"),
            "gpu_architecture": probe["configured"].get("gpu_architecture"),
            "rocm_version": health_payload.get("rocm_version"),
            "torch_version": metrics_payload.get("torch_version"),
            "transformers_version": metrics_payload.get("transformers_version"),
            "vllm_version": metrics_payload.get("vllm_version"),
            "model_id": model_payload.get("model_id", definition.model_id),
            "model_revision": model_payload.get("revision", definition.revision),
            "quantisation": model_payload.get("quantization", "none"),
            "dtype": model_payload.get("dtype", definition.dtype),
            "runtime": definition.runtime,
            "environment_overrides": {
                key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "LD_PRELOAD")
            },
            "load_result": "success" if health_payload.get("ready") else "not-confirmed",
            "warmup_result": "success" if health_payload.get("ready") else "not-confirmed",
            "smoke_result": "success" if result == "tested-working" else "failed",
            "cold_load_seconds": metrics_payload.get("load_seconds"),
            "first_output_seconds": (
                generation_payload.get("metrics", {}).get("first_token_seconds")
                or generation_payload.get("metrics", {}).get("first_output_seconds")
            ),
            "throughput_tokens_per_second": generation_payload.get("metrics", {}).get("tokens_per_second"),
            "peak_memory_bytes": metrics_payload.get("peak_memory_allocated_bytes"),
            "steady_memory_bytes": metrics_payload.get("memory_allocated_bytes"),
            "shutdown_result": "not-tested",
            "memory_recovery_result": "not-tested",
            "test_duration_seconds": round(asyncio.get_running_loop().time() - started, 4),
            "error_summary": error_summary,
        }
        record = request.app.state.compatibility_store.record_test(
            evidence, result=result, failure_class=failure_class
        )
        return {"ok": result == "tested-working", "worker_id": worker_id, "test": record}

    @router.get("/events")
    async def list_events(request: Request):
        return {"events": request.app.state.compatibility_store.list_events()}

    @router.post("/events", status_code=201)
    async def create_event(payload: EventDefinition, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        if store.get_event(payload.id) is not None:
            raise HTTPException(409, "That Event already exists")
        return store.save_event_draft(payload.model_dump(mode="json"))

    @router.get("/events/{event_id}")
    async def get_event(event_id: str, request: Request):
        record = request.app.state.compatibility_store.get_event(event_id)
        if record is None:
            raise HTTPException(404, "Unknown Event")
        return record

    @router.put("/events/{event_id}/draft")
    async def save_event_draft(event_id: str, payload: EventDefinition, request: Request):
        _require_mutable(request)
        if payload.id != event_id:
            raise HTTPException(409, "The Event identifier cannot be changed")
        store = request.app.state.compatibility_store
        if store.get_event(event_id) is None:
            raise HTTPException(404, "Unknown Event")
        return store.save_event_draft(payload.model_dump(mode="json"))

    @router.delete("/events/{event_id}/draft")
    async def discard_event_draft(event_id: str, request: Request):
        _require_mutable(request)
        try:
            return request.app.state.compatibility_store.discard_event_draft(event_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.delete("/events/{event_id}")
    async def delete_event(event_id: str, request: Request):
        _require_mutable(request)
        try:
            removed = request.app.state.compatibility_store.delete_event(event_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        if not removed:
            raise HTTPException(404, "Unknown Event")
        return {"ok": True, "event_id": event_id}

    @router.post("/events/{event_id}/validate")
    async def validate_stored_event(event_id: str, request: Request):
        definition = _event_definition(event_id, request)
        return _validate(definition, request)

    @router.post("/events/{event_id}/publish", status_code=201)
    async def publish_event(event_id: str, request: Request):
        _require_mutable(request)
        definition = _event_definition(event_id, request)
        validation = _validate(definition, request)
        if not validation["valid"]:
            raise HTTPException(409, {"message": "Event validation failed", "validation": validation})
        snapshot = routing_snapshot(definition, 0)
        revision = request.app.state.compatibility_store.publish_event(
            definition.model_dump(mode="json"), snapshot
        )
        return {"event_id": event_id, "revision": revision["revision"], "active": True}

    @router.get("/events/{event_id}/revisions")
    async def event_revisions(event_id: str, request: Request):
        if request.app.state.compatibility_store.get_event(event_id) is None:
            raise HTTPException(404, "Unknown Event")
        return {"revisions": request.app.state.compatibility_store.list_event_revisions(event_id)}

    @router.post("/events/{event_id}/routes/{route_id}/smoke")
    async def smoke_event_route(event_id: str, route_id: str, request: Request):
        snapshot = request.app.state.compatibility_store.active_routing_snapshot()
        if snapshot is None or snapshot.get("event_id") != event_id:
            raise HTTPException(409, "Publish this Event before smoke-testing its Routes")
        route = next(
            (item for item in snapshot.get("routes", []) if item.get("route_id") == route_id),
            None,
        )
        if route is None:
            raise HTTPException(404, "The Route is not in the live Event revision")
        path, body = _route_smoke_request(route)
        settings = request.app.state.settings
        timeout = (
            settings.diffusion_timeout_seconds
            if route["protocol_contract"] == "text-diffusion-v1"
            else max(60.0, settings.scenechat_timeout_seconds)
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"http://{settings.host}:{settings.gateway_port}{path}", json=body
                )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, f"Gateway Route smoke test failed: {error}") from error
        return {
            "ok": True,
            "event_id": event_id,
            "route_id": route_id,
            "public_name": route["public_name"],
            "evidence": next(
                (name for name in ("choices", "events", "frames", "ok") if result.get(name)),
                "response",
            ),
        }

    @router.post("/events/{event_id}/revisions/{revision}/publish")
    async def reactivate_event_revision(event_id: str, revision: int, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        record = store.get_event_revision(event_id, revision)
        if record is None:
            raise HTTPException(404, "Unknown Event revision")
        definition = EventDefinition.model_validate(record["definition"])
        validation = _validate(definition, request)
        if not validation["valid"]:
            raise HTTPException(409, {"message": "Event validation failed", "validation": validation})
        store.activate_event_revision(event_id, revision, routing_snapshot(definition, revision))
        return {"event_id": event_id, "revision": revision, "active": True}

    @router.get("/live")
    async def live(request: Request):
        snapshot = request.app.state.compatibility_store.active_routing_snapshot()
        if snapshot is None:
            return {"active_event": None, "routes": []}
        workers = {item["id"]: item for item in await list_workers(request)}
        routes = []
        for route in snapshot.get("routes", []):
            chain = [workers.get(worker_id) for worker_id in route.get("worker_ids", [])]
            effective = next(
                (worker for worker in chain if worker and worker["state"] in {"ready", "busy"}),
                None,
            )
            routes.append(
                {
                    **route,
                    "id": route["route_id"],
                    "workers": [worker for worker in chain if worker],
                    "effective_worker": effective,
                    "ready": effective is not None,
                }
            )
        return {
            "active_event": {
                "id": snapshot["event_id"],
                "name": snapshot["event_name"],
                "revision": snapshot["revision"],
            },
            "routes": routes,
        }

    return router


def _add_lifecycle_route(router: APIRouter, operation: str) -> None:
    async def lifecycle(worker_id: str, request: Request):
        _require_worker(request, worker_id)
        try:
            method = getattr(request.app.state.supervisor, operation)
            return await method(worker_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    router.add_api_route(
        f"/workers/{{worker_id}}/{operation}", lifecycle, methods=["POST"], name=f"v2_{operation}_worker"
    )


def _worker_records(request: Request, *, include_archived: bool = False):
    return request.app.state.compatibility_store.list_workers(include_archived=include_archived)


def _worker_response(request: Request, record):
    definition = WorkerDefinition.model_validate(record["definition"])
    process = None
    if definition.id in request.app.state.supervisor.workers:
        process = request.app.state.supervisor.get_worker(definition.id)
    return {
        **definition.model_dump(mode="json"),
        "state": process["state"] if process else "archived",
        "endpoint": process["endpoint"] if process else None,
        "pid": process["pid"] if process else None,
        "started_at": process["started_at"] if process else None,
        "last_error": process["last_error"] if process else None,
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "archived_at": record["archived_at"],
    }


def _require_worker(request: Request, worker_id: str) -> WorkerDefinition:
    definition = request.app.state.worker_definitions.get(worker_id)
    if definition is None:
        raise HTTPException(404, "Unknown Worker")
    return definition


def _worker_usage(worker_id: str, request: Request):
    references = []
    store = request.app.state.compatibility_store
    for event in store.list_events():
        for route in event["definition"].get("routes", []):
            if worker_id in route.get("worker_ids", []):
                references.append(
                    {
                        "event_id": event["definition"]["id"],
                        "event_name": event["definition"]["name"],
                        "route_id": route["id"],
                        "route_name": route["display_name"],
                        "kind": "draft",
                    }
                )
        for revision in store.list_event_revisions(event["definition"]["id"]):
            for route in revision["definition"].get("routes", []):
                if worker_id in route.get("worker_ids", []):
                    references.append(
                        {
                            "event_id": event["definition"]["id"],
                            "event_name": event["definition"]["name"],
                            "route_id": route["id"],
                            "route_name": route["display_name"],
                            "kind": "active" if revision["active"] else "history",
                            "revision": revision["revision"],
                        }
                    )
    blocking = [reference for reference in references if reference["kind"] != "history"]
    return {
        "worker_id": worker_id,
        "references": references,
        "blocking_references": blocking,
        "archivable": not blocking,
    }


def _event_definition(event_id: str, request: Request) -> EventDefinition:
    record = request.app.state.compatibility_store.get_event(event_id)
    if record is None:
        raise HTTPException(404, "Unknown Event")
    return EventDefinition.model_validate(record["definition"])


def _validate(definition: EventDefinition, request: Request):
    workers = list(request.app.state.worker_definitions.values())
    return validate_event(definition, workers, request.app.state.compatibility_store.list_tests())


def _require_mutable(request: Request) -> None:
    if request.app.state.settings.open_day:
        raise HTTPException(423, "Configuration is locked while Open Day mode is active")


def _route_smoke_request(route):
    public_name = route["public_name"]
    contract = route["protocol_contract"]
    if contract == "openai-chat-v1":
        return "/v1/chat/completions", {
            "model": public_name,
            "messages": [{"role": "user", "content": "Reply with the word ready."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        }
    if contract == "openai-completions-v1":
        return "/v1/completions", {
            "model": public_name,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        }
    if contract == "native-ar-trace-v1":
        return "/native/autoregressive/trace", {
            "model": public_name,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "top_k": 3,
            "seed": 7,
        }
    if contract == "text-diffusion-v1":
        return "/v1/refine", {
            "model": public_name,
            "prompt": "A local Worker is ready.",
            "denoising_steps": 4,
            "seed": 7,
        }
    raise HTTPException(409, "This protocol requires an interactive smoke-test client")


def _worker_smoke_request(definition: WorkerDefinition):
    model = definition.to_profile().alias
    if definition.runtime == "llama-vulkan":
        return (
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the word ready."}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            },
            None,
        )
    if definition.generation_family == "autoregressive":
        return (
            "/native/autoregressive/trace",
            {
                "model": model,
                "prompt": "Reply with the word ready.",
                "max_tokens": 4,
                "temperature": 0,
                "top_k": 3,
                "seed": 7,
            },
            None,
        )
    if definition.generation_family == "vision-language":
        return (
            "/native/vision-language/smoke",
            None,
            {"Authorization": "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")},
        )
    if definition.generation_family == "speech-conversation":
        return "/smoke", None, None
    if definition.generation_family == "text-diffusion":
        return (
            "/v1/refine",
            {
                "model": model,
                "prompt": "A local Worker is ready.",
                "denoising_steps": 4,
                "seed": 7,
            },
            None,
        )
    raise HTTPException(409, "This Worker family does not support an automatic smoke test")
