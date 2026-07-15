from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from modeldeck.config import Settings
from modeldeck.profiles import ModelProfile, default_model_profiles


def create_gateway_app(
    alias_routes: dict[str, list[ModelProfile]] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    app = FastAPI(title="ModelDeck stable local gateway", version="0.2.0")
    defaults = {profile.id: profile for profile in default_model_profiles()}
    routes = alias_routes or {
        "fast-chat": [defaults["qwen-small-rocm"], defaults["mock-ar"]],
        "token-explainer": [defaults["qwen-small-rocm"], defaults["mock-ar"]],
        "qwen-0-5b": [defaults["qwen-small-rocm"]],
        "qwen-1-5b": [defaults["qwen-1-5b-rocm"]],
        "qwen-3b": [defaults["qwen-3b-rocm"]],
        "text-diffusion": [
            defaults["diffusiongemma-q4-rocm"],
            defaults["mock-diffusion"],
        ],
        "text-diffusion-bf16": [defaults["diffusiongemma-rocm"]],
    }
    profiles = {profile.id: profile for candidates in routes.values() for profile in candidates}
    job_routes: dict[str, ModelProfile] = {}

    async def providers() -> list[dict[str, Any]]:
        result = []
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
        states = {state["id"]: state for state in await providers()}
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "modeldeck-local",
                    "ready": any(states[profile.id]["ready"] for profile in candidates),
                    "effective_provider": next(
                        (profile.id for profile in candidates if states[profile.id]["ready"]), None
                    ),
                }
                for alias, candidates in routes.items()
            ],
        }

    @app.get("/v1/capabilities")
    async def capabilities():
        return {alias: candidates[0].capabilities.model_dump() for alias, candidates in routes.items()}

    @app.get("/v1/providers")
    async def provider_list():
        return {"providers": await providers(), "cloud_fallback": False}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy_request(request, routes, "/v1/chat/completions", "fast-chat")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy_request(request, routes, "/v1/completions", "fast-chat")

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request):
        return await proxy_request(request, routes, "/native/autoregressive/trace", "token-explainer")

    @app.post("/v1/refine")
    async def refine(request: Request):
        return await proxy_request(
            request,
            routes,
            "/v1/refine",
            "text-diffusion",
            timeout_seconds=configured.diffusion_timeout_seconds,
        )

    @app.post("/v1/diffuse")
    async def diffuse(request: Request):
        response = await proxy_request(request, routes, "/v1/diffuse", "text-diffusion")
        if isinstance(response, JSONResponse) and response.status_code < 300:
            payload = json_loads(response.body)
            provider_id = response.headers.get("x-modeldeck-provider")
            if payload.get("job_id") and provider_id in profiles:
                job_routes[str(payload["job_id"])] = profiles[provider_id]
        return response

    @app.get("/v1/jobs/{job_id}")
    async def diffusion_job(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, routes)
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(provider, f"/v1/jobs/{job_id}")

    @app.get("/v1/jobs/{job_id}/events")
    async def diffusion_job_events(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, routes)
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_events(provider, f"/v1/jobs/{job_id}/events")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_diffusion_job(job_id: str):
        provider = await resolve_job_provider(job_id, job_routes, routes)
        if provider is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(provider, f"/v1/jobs/{job_id}/cancel", method="POST")

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel(request_id: str):
        cancelled = []
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
    async def vision():
        raise HTTPException(501, "No vision worker is implemented in this phase")

    return app


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
    candidates = routes.get(alias)
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
        body["model"] = alias
        upstream = client.build_request("POST", f"{endpoint(selected)}{path}", json=body)
        response = await client.send(upstream, stream=bool(body.get("stream")))
    except (httpx.HTTPError, ValueError):
        await client.aclose()
        return unavailable(alias, candidates[0].generation_family.value)
    if body.get("stream"):
        return StreamingResponse(
            forward_stream(response, client),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers={"x-modeldeck-provider": selected.id},
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
            headers={"x-modeldeck-provider": selected.id},
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
