from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.capabilities import (
    CAPABILITY_DEFINITIONS,
    capabilities_for_worker,
    compatible_runtime_template_ids,
    worker_cache_identity,
    worker_configuration_fingerprint,
)
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.config import gateway_base_url
from modeldeck.domain import (
    RoutingProfile,
    WorkerDefinition,
    routing_snapshot,
    validate_routing_profile,
)
from modeldeck.gemma4_settings import DEFAULT_VISUAL_TOKEN_BUDGET, VisualTokenBudget
from modeldeck.hardware import probe_environment
from modeldeck.prefix_cache import supports_wayfinder_prefix_cache
from modeldeck.profiles import LOCAL_PORT_RANGE, LocalProfileRequest, create_local_profile
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS
from modeldeck.q4_release import Q4ReleaseError, verify_modeldeck_q4_release
from modeldeck.registry import MAXIMUM_NEW_TOKENS_LIMIT
from modeldeck.thermal import ThermalAdmissionError


class WorkerCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    dtype: Literal["float16", "bfloat16", "float32"] | None = None
    lifecycle: Literal["resident", "on-demand", "exclusive"] | None = None
    context_length: int | None = Field(default=None, ge=256, le=32768)
    maximum_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAXIMUM_NEW_TOKENS_LIMIT,
    )
    maximum_denoising_steps: int | None = Field(default=None, ge=1, le=48)
    visual_token_budget: VisualTokenBudget | None = None
    artifact_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")
    runtime_template_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")
    prefix_cache_enabled: bool = False
    capability_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")


class WorkerRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)


class WorkerReplacementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    dtype: Literal["float16", "bfloat16", "float32"] | None = None
    lifecycle: Literal["resident", "on-demand", "exclusive"] | None = None
    context_length: int | None = Field(default=None, ge=256, le=32768)
    maximum_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAXIMUM_NEW_TOKENS_LIMIT,
    )
    maximum_denoising_steps: int | None = Field(default=None, ge=1, le=48)
    visual_token_budget: VisualTokenBudget | None = None
    prefix_cache_enabled: bool | None = None
    rebind_drafts: bool = True


def create_v3_router() -> APIRouter:
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
                    "required_worker_settings": contract.required_worker_settings,
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
        if payload.prefix_cache_enabled and not supports_wayfinder_prefix_cache(payload.model_id):
            raise HTTPException(
                409,
                "Prefix caching is allowlisted only for the dedicated WayFinder Qwen2.5 models",
            )
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
        candidates = cached.get("potential_capabilities", [])
        registrations = request.app.state.runtime_registrations
        candidate_templates = {
            item["id"]: compatible_runtime_template_ids(item["id"], support, registrations)
            for item in candidates
        }
        if payload.capability_id is not None:
            selected_capability = next(
                (item for item in candidates if item["id"] == payload.capability_id), None
            )
            if selected_capability is None:
                raise HTTPException(409, "That capability is not recognised for this Model")
            if not store.model_capability_allowed(payload.model_id, payload.revision, payload.capability_id):
                raise HTTPException(409, "Allow this capability before creating a Worker")
            if template_id not in candidate_templates[selected_capability["id"]]:
                raise HTTPException(409, "Select a trusted runtime listed for the requested capability")
        else:
            allowed_candidates = [
                item
                for item in candidates
                if template_id in candidate_templates[item["id"]]
                and store.model_capability_allowed(payload.model_id, payload.revision, item["id"])
            ]
            if not allowed_candidates:
                raise HTTPException(
                    409,
                    "Allow a capability exposed by the selected runtime, or supply capability_id",
                )
        # candidate_templates above is the authoritative compatibility gate. It supports
        # reviewed cross-family adapters such as Qwen3.5's text-only chat path while
        # retaining exact model matching in the catalogue resolver.
        selected = registrations.get(template_id) if template_id else None
        if selected is None:
            raise HTTPException(409, "Select an installed trusted runtime")
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
            dtype=payload.dtype or selected.template.dtype or "float16",
            lifecycle=payload.lifecycle or selected.template.lifecycle or "on-demand",
            context_length=payload.context_length
            or _integer_template_default(selected.template.settings, "context_length", 2048),
            maximum_new_tokens=payload.maximum_new_tokens
            or _integer_template_default(selected.template.settings, "maximum_new_tokens", 128),
            maximum_denoising_steps=payload.maximum_denoising_steps
            or _integer_template_default(selected.template.settings, "maximum_denoising_steps", 24),
            visual_token_budget=payload.visual_token_budget
            or _integer_template_default(
                selected.template.settings,
                "visual_token_budget",
                DEFAULT_VISUAL_TOKEN_BUDGET,
            ),
            artifact_id=payload.artifact_id,
            runtime_template_id=payload.runtime_template_id,
            prefix_cache_enabled=payload.prefix_cache_enabled,
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
        _require_worker(request, worker_id)
        record = request.app.state.compatibility_store.get_worker_definition(worker_id)
        if record is None:
            raise HTTPException(404, "Unknown Worker")
        return _worker_response(request, record)

    @router.patch("/workers/{worker_id}")
    async def rename_worker(worker_id: str, payload: WorkerRenameRequest, request: Request):
        _require_mutable(request)
        _require_worker(request, worker_id)
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
        _require_worker(request, worker_id)
        return _worker_usage(worker_id, request)

    @router.post("/workers/{worker_id}/prefix-cache/clear")
    async def clear_worker_prefix_cache(worker_id: str, request: Request):
        definition = _require_worker(request, worker_id)
        if definition.capabilities.get("prefix_caching") != "application-managed":
            raise HTTPException(409, "This Worker does not support application-managed prefix caching")
        snapshot = request.app.state.supervisor.get_worker(worker_id)
        if snapshot["state"] == "busy":
            raise HTTPException(409, "Wait for active generation before clearing the prefix cache")
        if snapshot["state"] != "ready":
            raise HTTPException(409, "Start the Worker before clearing its prefix cache")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=0.5)) as client:
                response = await client.post(f"http://127.0.0.1:{definition.port}/prefix-cache/clear")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, "The Worker prefix cache could not be cleared") from error
        return {
            "ok": payload.get("ok") is True,
            "worker_id": worker_id,
            "cleared_entries": int(payload.get("cleared_entries", 0)),
            "released_bytes": int(payload.get("released_bytes", 0)),
        }

    @router.post("/workers/{worker_id}/replacement", status_code=201)
    async def replace_worker(worker_id: str, payload: WorkerReplacementRequest, request: Request):
        definition = _require_worker(request, worker_id)
        if definition.runtime_template_id is None:
            raise HTTPException(409, "This Worker has no trusted runtime identity to replace")
        model_id = definition.artifact_model_id or definition.model_id
        revision = definition.artifact_revision or definition.revision
        artifact_id = _replacement_artifact_id(definition, model_id, revision, request)
        replacement = await create_worker(
            WorkerCreateRequest(
                name=payload.name,
                model_id=model_id,
                revision=revision,
                dtype=payload.dtype or definition.dtype,
                lifecycle=payload.lifecycle or definition.lifecycle,
                context_length=payload.context_length
                or _worker_integer_setting(definition, "context_length"),
                maximum_new_tokens=payload.maximum_new_tokens
                or _worker_integer_setting(definition, "maximum_new_tokens"),
                maximum_denoising_steps=payload.maximum_denoising_steps
                or _worker_integer_setting(definition, "maximum_denoising_steps"),
                visual_token_budget=payload.visual_token_budget
                or _worker_integer_setting(definition, "visual_token_budget"),
                artifact_id=artifact_id,
                runtime_template_id=definition.runtime_template_id,
                prefix_cache_enabled=(
                    payload.prefix_cache_enabled
                    if payload.prefix_cache_enabled is not None
                    else bool(definition.settings.get("prefix_cache_enabled", False))
                ),
            ),
            request,
        )
        rebound_profiles = []
        if payload.rebind_drafts:
            rebound_profiles = request.app.state.compatibility_store.rebind_routing_profile_drafts(
                worker_id, replacement["id"]
            )
        return {"replacement": replacement, "rebound_profile_drafts": rebound_profiles}

    @router.delete("/workers/{worker_id}")
    async def archive_worker(worker_id: str, request: Request):
        _require_mutable(request)
        _require_worker(request, worker_id)
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
                if not _has_smoke_evidence(definition, generation_payload):
                    raise RuntimeError("Smoke response contained no valid Worker evidence")
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

    @router.post("/workers/{worker_id}/capabilities/{capability_id}/qualify")
    async def qualify_worker_capability(worker_id: str, capability_id: str, request: Request):
        definition = _require_worker(request, worker_id)
        model_id, revision = worker_cache_identity(definition.model_dump(mode="json"))
        cached = next(
            (
                model
                for model in discover_huggingface_models()
                if model["model_id"] == model_id
                and model["revision"] == revision
                and model["download_state"] == "installed-untested"
            ),
            None,
        )
        if cached is None:
            raise HTTPException(409, "The Worker's exact cached Model revision is unavailable")
        candidate = next(
            (item for item in cached.get("potential_capabilities", []) if item["id"] == capability_id),
            None,
        )
        compatible_templates = (
            compatible_runtime_template_ids(
                capability_id,
                cached.get("configuration_support"),
                request.app.state.runtime_registrations,
            )
            if candidate is not None
            else []
        )
        if candidate is None or definition.runtime_template_id not in compatible_templates:
            raise HTTPException(409, "This Worker has no trusted adapter for that capability")
        store = request.app.state.compatibility_store
        if not store.model_cache_allowed(model_id, revision) or not store.model_capability_allowed(
            model_id, revision, capability_id
        ):
            raise HTTPException(409, "Allow this Model capability before qualifying it")
        worker = request.app.state.supervisor.get_worker(worker_id)
        if worker["state"] != "ready":
            raise HTTPException(409, "Worker must be ready before qualification")
        capability = CAPABILITY_DEFINITIONS[capability_id]
        if capability.protocol_contract_id is None:
            raise HTTPException(409, "This capability has no qualification contract")
        started = asyncio.get_running_loop().time()
        health_payload: dict[str, object] = {}
        model_payload: dict[str, object] = {}
        metrics_payload: dict[str, object] = {}
        generation_payload: dict[str, object] = {}
        try:
            path, body, headers = _worker_capability_request(definition, capability_id)
            async with httpx.AsyncClient(
                timeout=_capability_smoke_timeout(capability.protocol_contract_id, request)
            ) as client:
                health_response, model_response, metrics_response = await asyncio.gather(
                    client.get(f"{worker['endpoint']}/health"),
                    client.get(f"{worker['endpoint']}/model"),
                    client.get(f"{worker['endpoint']}/metrics"),
                )
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
                if not _has_smoke_evidence(definition, generation_payload):
                    raise RuntimeError("Qualification response contained no valid Worker evidence")
            result = "tested-working"
            failure_class = None
            error_summary = None
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            result = "transient-failure"
            failure_class = "capability-qualification-failure"
            error_summary = f"{type(error).__name__}: {error}"
        probe = await asyncio.to_thread(probe_environment)
        detected = probe["detected"]
        evidence = {
            "worker_id": definition.id,
            "capability_id": capability_id,
            "protocol_contract_id": capability.protocol_contract_id,
            "runtime_template_id": definition.runtime_template_id,
            "runtime_template_version": definition.runtime_template_version,
            "worker_configuration_fingerprint": worker_configuration_fingerprint(
                definition.model_dump(mode="json")
            ),
            "hardware_profile": probe["configured"]["profile_id"],
            "fedora_version": detected.get("fedora_release"),
            "kernel": detected.get("kernel"),
            "gpu": health_payload.get("device_name"),
            "gpu_architecture": probe["configured"].get("gpu_architecture"),
            "rocm_version": health_payload.get("rocm_version"),
            "torch_version": metrics_payload.get("torch_version"),
            "transformers_version": metrics_payload.get("transformers_version"),
            "vllm_version": metrics_payload.get("vllm_version"),
            "model_id": model_id,
            "model_revision": revision,
            "reported_model_id": model_payload.get("model_id", definition.model_id),
            "reported_model_revision": model_payload.get("revision", definition.revision),
            "quantisation": model_payload.get("quantization", "none"),
            "dtype": model_payload.get("dtype", definition.dtype),
            "runtime": definition.runtime,
            "environment_overrides": {
                key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "LD_PRELOAD")
            },
            "load_result": "success" if health_payload.get("ready") else "not-confirmed",
            "warmup_result": "success" if health_payload.get("ready") else "not-confirmed",
            "smoke_result": "success" if result == "tested-working" else "failed",
            "test_duration_seconds": round(asyncio.get_running_loop().time() - started, 4),
            "error_summary": error_summary,
        }
        record = store.record_test(evidence, result=result, failure_class=failure_class)
        return {"ok": result == "tested-working", "worker_id": worker_id, "test": record}

    @router.get("/routing-profiles")
    async def list_routing_profiles(request: Request):
        return {"profiles": request.app.state.compatibility_store.list_routing_profiles()}

    @router.post("/routing-profiles", status_code=201)
    async def create_routing_profile(payload: RoutingProfile, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        if store.get_routing_profile(payload.id) is not None:
            raise HTTPException(409, "That Routing Profile already exists")
        return store.save_routing_profile_draft(payload.model_dump(mode="json"))

    @router.get("/routing-profiles/{profile_id}")
    async def get_routing_profile(profile_id: str, request: Request):
        record = request.app.state.compatibility_store.get_routing_profile(profile_id)
        if record is None:
            raise HTTPException(404, "Unknown Routing Profile")
        return record

    @router.put("/routing-profiles/{profile_id}/draft")
    async def save_routing_profile_draft(profile_id: str, payload: RoutingProfile, request: Request):
        _require_mutable(request)
        if payload.id != profile_id:
            raise HTTPException(409, "The Routing Profile identifier cannot be changed")
        store = request.app.state.compatibility_store
        if store.get_routing_profile(profile_id) is None:
            raise HTTPException(404, "Unknown Routing Profile")
        return store.save_routing_profile_draft(payload.model_dump(mode="json"))

    @router.delete("/routing-profiles/{profile_id}/draft")
    async def discard_routing_profile_draft(profile_id: str, request: Request):
        _require_mutable(request)
        try:
            return request.app.state.compatibility_store.discard_routing_profile_draft(profile_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.delete("/routing-profiles/{profile_id}")
    async def delete_routing_profile(profile_id: str, request: Request):
        _require_mutable(request)
        try:
            removed = request.app.state.compatibility_store.delete_routing_profile(profile_id)
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        if not removed:
            raise HTTPException(404, "Unknown Routing Profile")
        return {"ok": True, "profile_id": profile_id}

    @router.post("/routing-profiles/{profile_id}/validate")
    async def validate_stored_routing_profile(profile_id: str, request: Request):
        return _validate(_routing_profile_definition(profile_id, request), request)

    @router.post("/routing-profiles/{profile_id}/publish", status_code=201)
    async def publish_routing_profile(profile_id: str, request: Request):
        _require_mutable(request)
        definition = _routing_profile_definition(profile_id, request)
        validation = _validate(definition, request)
        if not validation["valid"]:
            raise HTTPException(
                409, {"message": "Routing Profile validation failed", "validation": validation}
            )
        try:
            revision = request.app.state.compatibility_store.publish_routing_profile(
                definition.model_dump(mode="json"), routing_snapshot(definition, 0)
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"profile_id": profile_id, "revision": revision["revision"], "active": True}

    @router.get("/routing-profiles/{profile_id}/revisions")
    async def routing_profile_revisions(profile_id: str, request: Request):
        if request.app.state.compatibility_store.get_routing_profile(profile_id) is None:
            raise HTTPException(404, "Unknown Routing Profile")
        return {"revisions": request.app.state.compatibility_store.list_routing_profile_revisions(profile_id)}

    @router.post("/routing-profiles/{profile_id}/capabilities/{capability_id}/smoke")
    async def smoke_routing_profile_capability(profile_id: str, capability_id: str, request: Request):
        snapshot = next(
            (
                item
                for item in request.app.state.compatibility_store.active_routing_snapshots()
                if item.get("profile_id") == profile_id
            ),
            None,
        )
        if snapshot is None:
            raise HTTPException(409, "Publish this Routing Profile before smoke-testing capabilities")
        capability = next(
            (item for item in snapshot.get("capabilities", []) if item.get("capability_id") == capability_id),
            None,
        )
        if capability is None:
            raise HTTPException(404, "The capability is not in the live Routing Profile revision")
        if capability["protocol_contract"] == "openai-chat-v1":
            return await _rehearse_route_tool_calling(snapshot, capability, request)
        path, body = _capability_smoke_request(capability)
        timeout = _capability_smoke_timeout(capability["protocol_contract"], request)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                gateway_url = gateway_base_url(
                    request.app.state.settings.gateway_host,
                    request.app.state.settings.gateway_port,
                )
                response = await client.post(
                    f"{gateway_url}{path}",
                    json=body,
                )
            response.raise_for_status()
            result = (
                {"audio": True}
                if capability["protocol_contract"] == "speech-synthesis-v1"
                else response.json()
            )
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, f"Gateway capability smoke test failed: {error}") from error
        return {
            "ok": True,
            "profile_id": profile_id,
            "capability_id": capability_id,
            "public_name": capability["public_name"],
            "evidence": next(
                (
                    name
                    for name in ("choices", "events", "frames", "ok", "output_text", "text", "audio")
                    if result.get(name)
                ),
                "response",
            ),
        }

    @router.post("/routing-profiles/{profile_id}/revisions/{revision}/publish")
    async def reactivate_routing_profile_revision(profile_id: str, revision: int, request: Request):
        _require_mutable(request)
        store = request.app.state.compatibility_store
        record = store.get_routing_profile_revision(profile_id, revision)
        if record is None:
            raise HTTPException(404, "Unknown Routing Profile revision")
        definition = RoutingProfile.model_validate(record["definition"])
        validation = _validate(definition, request)
        if not validation["valid"]:
            raise HTTPException(
                409, {"message": "Routing Profile validation failed", "validation": validation}
            )
        try:
            store.activate_routing_profile_revision(
                profile_id, revision, routing_snapshot(definition, revision)
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"profile_id": profile_id, "revision": revision, "active": True}

    @router.delete("/routing-profiles/{profile_id}/active")
    async def deactivate_routing_profile(profile_id: str, request: Request):
        _require_mutable(request)
        if request.app.state.compatibility_store.get_routing_profile(profile_id) is None:
            raise HTTPException(404, "Unknown Routing Profile")
        return {
            "ok": request.app.state.compatibility_store.deactivate_routing_profile(profile_id),
            "profile_id": profile_id,
        }

    @router.get("/live")
    async def live(request: Request):
        snapshots = request.app.state.compatibility_store.active_routing_snapshots()
        if not snapshots:
            return {"active_profile": None, "active_profiles": [], "capabilities": []}
        workers = {item["id"]: item for item in await list_workers(request)}
        capabilities = []
        for snapshot in snapshots:
            for capability in snapshot.get("capabilities", []):
                chain = [workers.get(worker_id) for worker_id in capability.get("worker_ids", [])]
                effective = next(
                    (worker for worker in chain if worker and worker["state"] in {"ready", "busy"}),
                    None,
                )
                capabilities.append(
                    {
                        **capability,
                        "id": capability["capability_id"],
                        "profile_id": snapshot["profile_id"],
                        "tool_calling": request.app.state.compatibility_store.route_tool_calling_state(
                            str(snapshot["profile_id"]),
                            int(snapshot["revision"]),
                            str(capability["capability_id"]),
                        ),
                        "workers": [worker for worker in chain if worker],
                        "effective_worker": effective,
                        "ready": effective is not None,
                    }
                )
        active_profiles = [
            {"id": snapshot["profile_id"], "name": snapshot["profile_name"], "revision": snapshot["revision"]}
            for snapshot in snapshots
        ]
        return {
            "active_profile": active_profiles[0] if len(active_profiles) == 1 else None,
            "active_profiles": active_profiles,
            "capabilities": capabilities,
        }

    return router


async def _rehearse_route_tool_calling(
    snapshot: dict[str, object], capability: dict[str, object], request: Request
) -> dict[str, object]:
    """Exercise the public route without retaining prompts, arguments, or model output."""

    profile_id = str(snapshot["profile_id"])
    revision = int(snapshot["revision"])
    capability_id = str(capability["capability_id"])
    public_name = str(capability["public_name"])
    list_tool = {
        "type": "function",
        "function": {
            "name": "list_workspace_entries",
            "description": "List allowlisted workspace entries.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    read_tool = {
        "type": "function",
        "function": {
            "name": "read_workspace_text_file",
            "description": "Read one allowlisted workspace text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": (
                "Call list_workspace_entries exactly once. After its result, call "
                "read_workspace_text_file for Readme.md, then report the rehearsal marker "
                "from that file. Do not guess or answer before using the tools."
            ),
        }
    ]
    evidence: list[dict[str, object]] = []
    failure_code: str | None = None
    gateway_url = gateway_base_url(
        request.app.state.settings.gateway_host, request.app.state.settings.gateway_port
    )
    try:
        async with httpx.AsyncClient(timeout=_capability_smoke_timeout("openai-chat-v1", request)) as client:
            started = time.perf_counter()
            response = await client.post(
                f"{gateway_url}/v1/chat/completions",
                json={
                    "model": public_name,
                    "messages": messages,
                    "tools": [list_tool],
                    "tool_choice": "required",
                    "temperature": 0,
                    "max_tokens": 192,
                },
            )
            call, category = _rehearsal_tool_call(response, "list_workspace_entries", {})
            evidence.append(
                {
                    "tool_call_count": 1 if call else 0,
                    "result_category": category,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            if call is None:
                failure_code = category
            else:
                messages.extend(
                    (
                        {"role": "assistant", "content": None, "tool_calls": [call]},
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps({"entries": ["Readme.md"]}),
                        },
                    )
                )

            if failure_code is None:
                started = time.perf_counter()
                response = await client.post(
                    f"{gateway_url}/v1/chat/completions",
                    json={
                        "model": public_name,
                        "messages": messages,
                        "tools": [read_tool],
                        "tool_choice": "required",
                        "temperature": 0,
                        "max_tokens": 192,
                    },
                )
                call, category = _rehearsal_tool_call(
                    response, "read_workspace_text_file", {"path": "Readme.md"}
                )
                evidence.append(
                    {
                        "tool_call_count": 1 if call else 0,
                        "result_category": category,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    }
                )
                if call is None:
                    failure_code = category
                else:
                    messages.extend(
                        (
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    {
                                        "path": "Readme.md",
                                        "content": "MODELDECK_TOOL_REHEARSAL_OK",
                                    }
                                ),
                            },
                        )
                    )

            if failure_code is None:
                started = time.perf_counter()
                response = await client.post(
                    f"{gateway_url}/v1/chat/completions",
                    json={
                        "model": public_name,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": 96,
                    },
                )
                valid, category = _valid_rehearsal_final_text(response, "MODELDECK_TOOL_REHEARSAL_OK")
                evidence.append(
                    {
                        "tool_call_count": 0,
                        "result_category": category,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    }
                )
                if not valid:
                    failure_code = category
    except httpx.TimeoutException:
        failure_code = "gateway_timeout"
    except httpx.HTTPError:
        failure_code = "local_route_unavailable"
    supported = failure_code is None and len(evidence) == 3
    state = request.app.state.compatibility_store.save_route_tool_calling_rehearsal(
        profile_id,
        revision,
        capability_id,
        supported=supported,
        failure_code=failure_code,
        evidence={"probe_count": len(evidence), "probes": evidence},
    )
    return {
        "ok": supported,
        "profile_id": profile_id,
        "capability_id": capability_id,
        "public_name": public_name,
        "tool_calling": state,
    }


def _valid_rehearsal_tool_call(
    response: httpx.Response, expected_name: str, expected_arguments: dict[str, object]
) -> tuple[bool, str]:
    """Validate a response shape without recording any response content."""

    call, category = _rehearsal_tool_call(response, expected_name, expected_arguments)
    return call is not None, category


def _rehearsal_tool_call(
    response: httpx.Response, expected_name: str, expected_arguments: dict[str, object]
) -> tuple[dict[str, object] | None, str]:
    """Return one validated OpenAI tool call for bounded rehearsal continuation."""

    try:
        payload = response.json()
    except ValueError:
        return None, "invalid_worker_response"
    if not response.is_success:
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        return None, str(code) if isinstance(code, str) else "tool_calling_probe_failed"
    try:
        calls = payload["choices"][0]["message"]["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            return None, "tool_call_count_invalid"
        call = calls[0]
        if not isinstance(call, dict):
            return None, "tool_call_protocol_invalid"
        if not isinstance(call.get("id"), str) or not call["id"]:
            return None, "tool_call_id_invalid"
        if call.get("type") != "function" or call["function"]["name"] != expected_name:
            return None, "tool_call_name_invalid"
        arguments = call["function"]["arguments"]
        if not isinstance(arguments, str) or json.loads(arguments) != expected_arguments:
            return None, "tool_call_arguments_invalid"
    except (KeyError, TypeError, ValueError):
        return None, "tool_call_protocol_invalid"
    return call, "valid"


def _valid_rehearsal_final_text(response: httpx.Response, marker: str) -> tuple[bool, str]:
    """Validate grounded final text without retaining the model response."""

    try:
        payload = response.json()
    except ValueError:
        return False, "invalid_worker_response"
    if not response.is_success:
        error = payload.get("error") if isinstance(payload, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        return False, str(code) if isinstance(code, str) else "tool_calling_probe_failed"
    try:
        message = payload["choices"][0]["message"]
        if message.get("tool_calls"):
            return False, "final_tool_call_unexpected"
        content = message["content"]
        if not isinstance(content, str) or marker not in content:
            return False, "grounded_final_text_invalid"
    except (KeyError, TypeError):
        return False, "tool_call_protocol_invalid"
    return True, "valid"


def _add_lifecycle_route(router: APIRouter, operation: str) -> None:
    async def lifecycle(worker_id: str, request: Request):
        _require_worker(request, worker_id)
        try:
            method = getattr(request.app.state.supervisor, operation)
            return await method(worker_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ThermalAdmissionError as error:
            raise HTTPException(429, error.decision.as_dict()) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    router.add_api_route(
        f"/workers/{{worker_id}}/{operation}", lifecycle, methods=["POST"], name=f"v3_{operation}_worker"
    )


def _worker_records(request: Request, *, include_archived: bool = False):
    return [
        record
        for record in request.app.state.compatibility_store.list_workers(include_archived=include_archived)
        if record["definition"].get("id") in request.app.state.worker_definitions
    ]


def _worker_integer_setting(definition: WorkerDefinition, name: str) -> int | None:
    value = definition.settings.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _replacement_artifact_id(
    definition: WorkerDefinition, model_id: str, revision: str, request: Request
) -> str | None:
    registration = request.app.state.runtime_registrations.get(definition.runtime_template_id)
    if registration is None:
        raise HTTPException(409, "The Worker's trusted runtime is no longer installed")
    if registration.template.cache_setting != "artifact_path":
        return None
    artifact_path = definition.settings.get("artifact_path")
    cached = next(
        (
            model
            for model in discover_huggingface_models()
            if model["model_id"] == model_id and model["revision"] == revision
        ),
        None,
    )
    if cached is None or not isinstance(artifact_path, str):
        raise HTTPException(409, "The Worker's allowlisted Model artefact is no longer available")
    current_path = Path(artifact_path).resolve()
    snapshot_path = Path(cached["snapshot_location"])
    for artifact in cached.get("artifacts", []):
        if any((snapshot_path / filename).resolve() == current_path for filename in artifact["filenames"]):
            return artifact["artifact_id"]
    raise HTTPException(409, "The Worker's allowlisted Model artefact could not be identified")


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
    for profile in store.list_routing_profiles():
        for capability in profile["definition"].get("capabilities", []):
            if worker_id in capability.get("worker_ids", []):
                references.append(
                    {
                        "profile_id": profile["definition"]["id"],
                        "profile_name": profile["definition"]["name"],
                        "capability_id": capability["id"],
                        "capability_name": capability["display_name"],
                        "kind": "draft",
                    }
                )
        for revision in store.list_routing_profile_revisions(profile["definition"]["id"]):
            for capability in revision["definition"].get("capabilities", []):
                if worker_id in capability.get("worker_ids", []):
                    references.append(
                        {
                            "profile_id": profile["definition"]["id"],
                            "profile_name": profile["definition"]["name"],
                            "capability_id": capability["id"],
                            "capability_name": capability["display_name"],
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


def _routing_profile_definition(profile_id: str, request: Request) -> RoutingProfile:
    record = request.app.state.compatibility_store.get_routing_profile(profile_id)
    if record is None:
        raise HTTPException(404, "Unknown Routing Profile")
    return RoutingProfile.model_validate(record["definition"])


def _validate(definition: RoutingProfile, request: Request):
    workers = list(request.app.state.worker_definitions.values())
    store = request.app.state.compatibility_store
    model_policy = store.list_model_cache_policy()
    stored_policy = store.list_model_capability_policy()
    effective_policy = {
        key: value and model_policy.get((key[0], key[1]), True) for key, value in stored_policy.items()
    }
    # A pre-v4 Worker is grandfathered only when its model-level master policy
    # remains allowed. The v3→v4 migration persists the same grants explicitly;
    # this branch also preserves trusted legacy fixtures and recovery imports.
    for worker in workers:
        if worker.capability_policy_version is not None:
            continue
        model_id, revision = worker_cache_identity(worker.model_dump(mode="json"))
        if not model_policy.get((model_id, revision), True):
            continue
        for capability_id in capabilities_for_worker(worker.model_dump(mode="json")):
            effective_policy.setdefault((model_id, revision, capability_id), True)
    return validate_routing_profile(
        definition,
        workers,
        store.list_tests(),
        effective_policy,
    )


def _require_mutable(request: Request) -> None:
    if request.app.state.settings.configuration_locked:
        raise HTTPException(423, "Configuration is locked by the local deployment policy")


def _integer_template_default(settings: dict[str, object], name: str, fallback: int) -> int:
    value = settings.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _capability_smoke_request(capability):
    public_name = capability["public_name"]
    contract = capability["protocol_contract"]
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
    if contract == "openai-embeddings-v1":
        return "/v1/embeddings", {
            "model": public_name,
            "input": ["The local Worker is ready."],
        }
    if contract == "native-ar-trace-v1":
        return "/native/v1/autoregressive/traces", {
            "model": public_name,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "top_k": 3,
            "seed": 7,
        }
    if contract == "text-diffusion-v1":
        return "/native/v1/text-diffusion/refine", {
            "model": public_name,
            "prompt": "A local Worker is ready.",
            "denoising_steps": 4,
            "seed": 7,
        }
    if contract in {"translation-en-fr-v1", "translation-en-de-v1"}:
        target_language = "fr" if contract == "translation-en-fr-v1" else "de"
        return "/v1/translations", {
            "request_id": "modeldeck-route-smoke",
            "model": public_name,
            "input": "The local Worker is ready.",
            "source_language": "en",
            "target_language": target_language,
        }
    if contract == "speech-synthesis-v1":
        return "/v1/audio/speech", {
            "request_id": "modeldeck-route-smoke",
            "model": public_name,
            "input": "The local Worker is ready.",
            "voice": "ryan",
            "language": "en",
            "response_format": "wav",
        }
    if contract == "speech-recognition-v1":
        return "/v1/audio/transcriptions", {
            "request_id": "modeldeck-route-smoke",
            "model": public_name,
            "language": "en",
            "encoding": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "audio_base64": "AAAAAA==",
        }
    raise HTTPException(409, "This protocol requires an interactive smoke-test client")


def _capability_smoke_timeout(contract: str, request: Request) -> float:
    settings = request.app.state.settings
    if contract == "text-diffusion-v1":
        return settings.diffusion_timeout_seconds
    if contract == "speech-synthesis-v1":
        return settings.speech_synthesis_timeout_seconds
    if contract == "speech-recognition-v1":
        return settings.speech_recognition_timeout_seconds
    if contract.startswith("translation-"):
        return settings.translation_timeout_seconds
    return max(60.0, settings.scenechat_timeout_seconds)


def _worker_capability_request(
    definition: WorkerDefinition, capability_id: str
) -> tuple[str, dict[str, object] | None, dict[str, str] | None]:
    model = definition.to_profile().alias
    if capability_id == "general-chat":
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
    if capability_id == "text-completion":
        return (
            "/v1/completions",
            {
                "model": model,
                "prompt": "Reply with the word ready.",
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            },
            None,
        )
    if capability_id == "autoregressive-trace":
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
    if capability_id == "embeddings":
        return "/v1/embeddings", {"model": model, "input": ["The local Worker is ready."]}, None
    if capability_id == "scene-analysis":
        return (
            "/native/vision-language/smoke",
            None,
            {"Authorization": "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")},
        )
    if capability_id == "text-refinement":
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
    if capability_id == "speech-conversation":
        return "/smoke", None, None
    if capability_id in {"translation-en-fr", "translation-en-de"}:
        return "/native/text-translation/smoke", None, None
    if capability_id == "speech-synthesis":
        return "/native/speech-synthesis/smoke", None, None
    if capability_id == "speech-recognition":
        return "/native/speech-recognition/smoke", None, None
    raise HTTPException(409, "This capability has no bounded qualification adapter")


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
    if definition.generation_family == "embedding":
        return (
            "/v1/embeddings",
            {"model": model, "input": ["The local Worker is ready."]},
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
    if definition.generation_family == "text-translation":
        return "/native/text-translation/smoke", None, None
    if definition.generation_family == "speech-synthesis":
        return "/native/speech-synthesis/smoke", None, None
    if definition.generation_family == "speech-recognition":
        return "/native/speech-recognition/smoke", None, None
    raise HTTPException(409, "This Worker family does not support an automatic smoke test")


def _has_smoke_evidence(definition: WorkerDefinition, payload: dict[str, object]) -> bool:
    if definition.generation_family != "embedding":
        return bool(
            payload.get("events") or payload.get("frames") or payload.get("ok") or payload.get("choices")
        )
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False
    for index, item in enumerate(data):
        if not isinstance(item, dict) or item.get("object") != "embedding" or item.get("index") != index:
            return False
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) != 1024:
            return False
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
            return False
    return True
