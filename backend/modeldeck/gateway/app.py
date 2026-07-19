from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.demo_config import profile_has_recorded_success
from modeldeck.profile_registry import (
    ensure_seeded_profiles,
    load_local_profiles,
    profile_allowed,
    profile_verified,
)
from modeldeck.profiles import ModelProfile, default_model_profiles
from modeldeck.protocol import CapabilitySet
from modeldeck.provider_selection import SCENECHAT_ALIAS
from modeldeck.registry import ReservedAlias, reserved_aliases


def create_gateway_app(
    alias_routes: dict[str, list[ModelProfile]] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    app = FastAPI(title="ModelDeck stable local gateway", version="0.2.0")
    ensure_seeded_profiles(configured.data_dir, default_model_profiles())
    alias_contracts = reserved_aliases()
    base_routes = alias_routes or {}
    job_routes: dict[str, ModelProfile] = {}
    store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")

    def selected_provider_id(alias: str, contract: ReservedAlias) -> str | None:
        if alias_routes is not None:
            candidates = base_routes.get(alias, [])
            return candidates[0].id if candidates else contract.default_provider
        return store.gateway_provider_selection(alias) or contract.default_provider

    def active_routes(adapter_ids: set[str] | None = None) -> dict[str, list[ModelProfile]]:
        all_profiles = {profile.id: profile for profile in load_local_profiles(configured.data_dir)}
        routes = (
            {alias: list(candidates) for alias, candidates in base_routes.items()}
            if alias_routes is not None
            else {
                alias: [
                    all_profiles[profile_id]
                    for profile_id in contract.providers
                    if profile_id in all_profiles
                ]
                for alias, contract in alias_contracts.items()
            }
        )
        qualification_policies: dict[str, str] = {}
        if alias_routes is None:
            profile_origins = store.model_profile_origins()
            active_snapshot = store.active_routing_snapshot()
            if active_snapshot is not None:
                routes = {}
                for route in active_snapshot.get("routes", []):
                    if adapter_ids is not None and route.get("adapter_id") not in adapter_ids:
                        continue
                    alias = str(route.get("public_model", ""))
                    if not alias:
                        continue
                    routes[alias] = [
                        all_profiles[deployment_id]
                        for deployment_id in route.get("providers", [])
                        if deployment_id in all_profiles
                    ]
                    qualification_policies[alias] = str(route.get("qualification_policy", "registered"))
            else:
                for profile in all_profiles.values():
                    if profile_origins.get(profile.id) == "local" and profile.alias not in routes:
                        routes[profile.alias] = [profile]
                for alias, contract in alias_contracts.items():
                    if contract.selection != "explicit":
                        continue
                    selected = all_profiles.get(selected_provider_id(alias, contract) or "")
                    routes[alias] = [selected] if selected is not None and contract.accepts(selected) else []
        policy = store.list_model_cache_policy()
        compatibility_tests = store.list_tests()
        filtered = {}
        for alias, candidates in routes.items():
            allowed_candidates = [
                profile
                for profile in candidates
                if profile_allowed(profile, policy)
                and profile_verified(profile, compatibility_tests)
                and (
                    qualification_policies.get(alias) != "tested-working-recorded"
                    or profile_has_recorded_success(profile, compatibility_tests)
                )
            ]
            contract = alias_contracts.get(alias)
            if allowed_candidates or (contract is not None and contract.selection == "explicit"):
                filtered[alias] = allowed_candidates
        return filtered

    async def providers(routes: dict[str, list[ModelProfile]] | None = None) -> list[dict[str, Any]]:
        result = []
        profiles = {
            profile.id: profile
            for candidates in (routes if routes is not None else active_routes()).values()
            for profile in candidates
        }
        async with httpx.AsyncClient(timeout=0.3) as client:
            for profile in profiles.values():
                health, ready = await provider_health(client, profile)
                result.append(
                    {
                        "id": profile.id,
                        "alias": profile.alias,
                        "generation_family": profile.generation_family,
                        "endpoint": endpoint(profile),
                        "ready": ready,
                        "health": health,
                    }
                )
        return result

    @app.get("/v1/health")
    async def health():
        states = await providers()
        return {
            "status": "ok",
            "service": "modeldeck-gateway",
            "ready_providers": sum(provider["ready"] for provider in states),
        }

    @app.get("/v1/models")
    async def models():
        routes = active_routes()
        states = {state["id"]: state for state in await providers(routes)}
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "modeldeck-local",
                    "ready": any(states[profile.id]["ready"] for profile in candidates),
                    "selected_provider": (
                        selected_provider_id(alias, alias_contracts[alias])
                        if alias in alias_contracts and alias_contracts[alias].selection == "explicit"
                        else (candidates[0].id if candidates else None)
                    ),
                    "effective_provider": next(
                        (profile.id for profile in candidates if states[profile.id]["ready"]), None
                    ),
                }
                for alias, candidates in routes.items()
            ],
        }

    @app.get("/v1/capabilities")
    async def capabilities():
        routes = active_routes()
        profiles = {profile.id: profile for profile in load_local_profiles(configured.data_dir)}
        return {
            alias: alias_capabilities(alias, candidates, alias_contracts, profiles)
            for alias, candidates in routes.items()
        }

    @app.get("/v1/providers")
    async def provider_list():
        return {"providers": await providers(), "cloud_fallback": False}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy_request(
            request,
            active_routes({"openai-chat-v1", "scene-analysis-v1"}),
            "/v1/chat/completions",
            "fast-chat",
            timeout_seconds=configured.scenechat_timeout_seconds,
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy_request(
            request, active_routes({"openai-completions-v1"}), "/v1/completions", "fast-chat"
        )

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request):
        return await proxy_request(
            request,
            active_routes({"native-ar-trace-v1"}),
            "/native/autoregressive/trace",
            "token-explainer",
        )

    @app.post("/v1/refine")
    async def refine(request: Request):
        return await proxy_request(
            request,
            active_routes({"text-diffusion-v1"}),
            "/v1/refine",
            "text-diffusion",
            timeout_seconds=configured.diffusion_timeout_seconds,
        )

    @app.post("/v1/diffuse")
    async def diffuse(request: Request):
        routes = active_routes({"text-diffusion-v1"})
        response = await proxy_request(request, routes, "/v1/diffuse", "text-diffusion")
        if isinstance(response, JSONResponse) and response.status_code < 300:
            payload = json_loads(response.body)
            provider_id = response.headers.get("x-modeldeck-provider")
            profiles = {profile.id: profile for candidates in routes.values() for profile in candidates}
            if payload.get("job_id") and provider_id in profiles:
                job_routes[str(payload["job_id"])] = profiles[provider_id]
        return response

    @app.get("/v1/jobs/{job_id}")
    async def diffusion_job(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, active_routes())
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(provider, f"/v1/jobs/{job_id}")

    @app.get("/v1/jobs/{job_id}/events")
    async def diffusion_job_events(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, active_routes())
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_events(provider, f"/v1/jobs/{job_id}/events")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_diffusion_job(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, active_routes())
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(provider, f"/v1/jobs/{job_id}/cancel", method="POST")

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel(request_id: str):
        cancelled = []
        profiles = {profile.id: profile for candidates in active_routes().values() for profile in candidates}
        async with httpx.AsyncClient(timeout=1.0) as client:
            for profile in profiles.values():
                health_payload, ready = await provider_health(client, profile)
                if not ready or not health_payload:
                    continue
                try:
                    response = await client.post(
                        f"{endpoint(profile)}/cancel", json={"request_id": request_id}
                    )
                    if response.is_success and response.json().get("ok"):
                        cancelled.append(profile.id)
                except (httpx.HTTPError, ValueError):
                    continue
        return {"ok": bool(cancelled), "request_id": request_id, "providers": cancelled}

    @app.post("/v1/vision/analyse")
    async def vision(request: Request):
        return await proxy_request(
            request,
            active_routes({"scene-analysis-v1"}),
            "/v1/chat/completions",
            SCENECHAT_ALIAS,
            timeout_seconds=configured.scenechat_timeout_seconds,
        )

    @app.websocket("/v1/speech/conversations")
    async def speech_conversation(client_socket: WebSocket):
        await client_socket.accept()
        try:
            first = await asyncio.wait_for(client_socket.receive_text(), timeout=5)
            start = json_loads(first.encode())
            alias = str(start.get("model") or "repartee-speech")
            candidates = route_candidates(active_routes({"speech-conversation-v1"}), alias)
            if not candidates:
                await client_socket.send_json({"type": "error", "code": "local_provider_unavailable"})
                await client_socket.close(code=1013)
                return
            selected = None
            async with httpx.AsyncClient(timeout=0.5) as health_client:
                for candidate in candidates:
                    if candidate.generation_family.value != "speech-conversation":
                        continue
                    _, ready = await provider_health(health_client, candidate)
                    if ready:
                        selected = candidate
                        break
            if selected is None:
                await client_socket.send_json({"type": "error", "code": "local_provider_unavailable"})
                await client_socket.close(code=1013)
                return
            from websockets.asyncio.client import connect

            async with connect(
                f"ws://127.0.0.1:{selected.port}/v1/speech/conversations",
                max_size=96_000,
            ) as upstream:
                await upstream.send(first)

                async def client_to_worker() -> None:
                    while True:
                        message = await client_socket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("bytes") is not None:
                            await upstream.send(message["bytes"])
                        elif message.get("text") is not None:
                            await upstream.send(message["text"])

                async def worker_to_client() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await client_socket.send_bytes(message)
                        else:
                            await client_socket.send_text(message)

                tasks = {asyncio.create_task(client_to_worker()), asyncio.create_task(worker_to_client())}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        except (ValueError, TimeoutError, WebSocketDisconnect):
            if client_socket.client_state.name != "DISCONNECTED":
                await client_socket.close(code=1008)

    return app


def alias_capabilities(
    alias: str,
    candidates: list[ModelProfile],
    contracts: dict[str, ReservedAlias],
    defaults: dict[str, ModelProfile],
) -> dict[str, Any]:
    if candidates:
        return candidates[0].capabilities.model_dump()
    contract = contracts[alias]
    if contract.default_provider and contract.default_provider in defaults:
        return defaults[contract.default_provider].capabilities.model_dump()
    capabilities = CapabilitySet()
    return capabilities.model_copy(
        update={name: True for name in contract.required_capabilities}
    ).model_dump()


async def proxy_request(
    request: Request,
    routes: dict[str, list[ModelProfile]],
    path: str,
    default_alias: str,
    *,
    timeout_seconds: float = 60.0,
):
    body = await request.json()
    alias = str(body.get("model") or default_alias)
    candidates = route_candidates(routes, alias)
    if not candidates:
        return unavailable(alias, "unknown")
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=0.5))
    selected = None
    try:
        for profile in candidates:
            _, ready = await provider_health(client, profile)
            if ready:
                selected = profile
                break
        if selected is None:
            await client.aclose()
            return unavailable(alias, candidates[0].generation_family.value)
        body["model"] = upstream_model(selected, alias)
        upstream = client.build_request(
            "POST",
            f"{endpoint(selected)}{path}",
            json=body,
            headers=upstream_headers(selected),
        )
        response = await client.send(upstream, stream=bool(body.get("stream")))
    except (httpx.HTTPError, ValueError):
        await client.aclose()
        return unavailable(alias, candidates[0].generation_family.value)
    if body.get("stream"):
        return StreamingResponse(
            forward_stream(response, client),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers=provider_response_headers(selected),
        )
    try:
        payload = await response.aread()
        response_payload = json_loads(payload)
        if path == "/native/autoregressive/trace" and response.is_success:
            metadata_error = trace_token_metadata_error(response_payload)
            if metadata_error:
                return invalid_trace_metadata(selected.id, metadata_error)
        return JSONResponse(
            response_payload,
            status_code=response.status_code,
            headers=provider_response_headers(selected),
        )
    finally:
        await response.aclose()
        await client.aclose()


async def forward_stream(response: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


async def resolve_job_provider(
    job_id: str,
    job_routes: dict[str, ModelProfile],
    routes: dict[str, list[ModelProfile]],
) -> ModelProfile | None:
    if provider := job_routes.get(job_id):
        return provider
    candidates = {
        profile.id: profile
        for route_candidates in routes.values()
        for profile in route_candidates
        if profile.generation_family.value == "text-diffusion"
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=0.5)) as client:
        for candidate in candidates.values():
            try:
                response = await client.get(f"{endpoint(candidate)}/v1/jobs/{job_id}")
            except httpx.HTTPError:
                continue
            if response.status_code != 404:
                job_routes[job_id] = candidate
                return candidate
    return None


async def proxy_job_request(
    provider: ModelProfile,
    path: str,
    *,
    method: str = "GET",
) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=0.5)) as client:
            response = await client.request(method, f"{endpoint(provider)}{path}")
            payload = json_loads(await response.aread())
    except (httpx.HTTPError, ValueError):
        return unavailable("text-diffusion", "text-diffusion")
    return JSONResponse(
        payload,
        status_code=response.status_code,
        headers={"x-modeldeck-provider": provider.id},
    )


async def proxy_job_events(provider: ModelProfile, path: str) -> StreamingResponse | JSONResponse:
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=0.5))
    try:
        response = await client.send(
            client.build_request("GET", f"{endpoint(provider)}{path}"),
            stream=True,
        )
    except httpx.HTTPError:
        await client.aclose()
        return unavailable("text-diffusion", "text-diffusion")
    return StreamingResponse(
        forward_stream(response, client),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
        headers={"x-modeldeck-provider": provider.id},
    )


async def provider_health(
    client: httpx.AsyncClient, profile: ModelProfile
) -> tuple[dict[str, Any] | None, bool]:
    try:
        response = await client.get(f"{endpoint(profile)}/health")
        health = response.json()
        return health, response.is_success and health.get("ready") is True
    except (httpx.HTTPError, ValueError):
        return None, False


def endpoint(profile: ModelProfile) -> str:
    return f"http://127.0.0.1:{profile.port}"


def route_candidates(
    routes: dict[str, list[ModelProfile]], alias_or_model_id: str
) -> list[ModelProfile] | None:
    if candidates := routes.get(alias_or_model_id):
        return candidates
    for candidates in routes.values():
        if any(
            profile.model_id == alias_or_model_id and profile.generation_family.value == "vision-language"
            for profile in candidates
        ):
            return candidates
    return None


def upstream_model(profile: ModelProfile, alias: str) -> str:
    if profile.generation_family.value == "vision-language":
        return profile.model_id
    return alias


def upstream_headers(profile: ModelProfile) -> dict[str, str]:
    if profile.generation_family.value != "vision-language":
        return {}
    return {"Authorization": "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")}


def provider_response_headers(profile: ModelProfile) -> dict[str, str]:
    headers = {"x-modeldeck-provider": profile.id}
    if profile.preferred_runtime == "mock":
        headers["x-modeldeck-fallback"] = "mock"
    return headers


def json_loads(payload: bytes) -> Any:
    import json

    return json.loads(payload)


def unavailable(alias: str, family: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "local_provider_unavailable",
                "message": f"No ready local provider supplies alias '{alias}'.",
                "alias": alias,
                "required_generation_family": family,
                "cloud_fallback_attempted": False,
            }
        },
        status_code=503,
    )


def trace_token_metadata_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "trace response must be a JSON object"
    prompt_ids = payload.get("prompt_token_ids")
    prompt_tokens = payload.get("prompt_tokens")
    user_ids = payload.get("user_prompt_token_ids")
    user_tokens = payload.get("user_prompt_tokens")
    if not isinstance(prompt_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_ids
    ):
        return "prompt_token_ids must be an array of integers"
    if not isinstance(prompt_tokens, list) or not all(isinstance(token, str) for token in prompt_tokens):
        return "prompt_tokens must be an array of strings"
    if len(prompt_tokens) != len(prompt_ids):
        return "prompt_tokens must align one-to-one with prompt_token_ids"
    if not isinstance(user_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in user_ids
    ):
        return "user_prompt_token_ids must be an array of integers"
    if not isinstance(user_tokens, list) or not all(isinstance(token, str) for token in user_tokens):
        return "user_prompt_tokens must be an array of strings"
    if len(user_tokens) != len(user_ids):
        return "user_prompt_tokens must align one-to-one with user_prompt_token_ids"
    return None


def invalid_trace_metadata(provider_id: str, reason: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "invalid_worker_trace_metadata",
                "message": f"Local provider '{provider_id}' returned invalid trace token metadata: {reason}.",
                "provider": provider_id,
            }
        },
        status_code=502,
        headers={"x-modeldeck-provider": provider_id},
    )


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_gateway_app(settings=settings),
        host=settings.host,
        port=settings.gateway_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
