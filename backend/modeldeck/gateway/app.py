from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.profiles import ModelProfile
from modeldeck.protocol import CapabilitySet, GenerationFamily


def create_gateway_app(
    alias_routes: dict[str, list[ModelProfile]] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    app = FastAPI(title="ModelDeck stable local gateway", version="0.2.0")
    app.state.last_request_diagnostics = None
    app.state.active_request_workers = {}
    app.state.active_request_lock = asyncio.Lock()
    base_routes = alias_routes or {}
    job_routes: dict[str, ModelProfile] = {}
    store = CompatibilityStore(configured.data_dir / "modeldeck.sqlite3")
    if alias_routes is None:
        store.initialise_v2()

    def active_routes(adapter_ids: set[str] | None = None) -> dict[str, list[ModelProfile]]:
        if alias_routes is not None:
            return {name: list(candidates) for name, candidates in base_routes.items()}
        definitions = {
            definition.id: definition
            for definition in (
                WorkerDefinition.model_validate(record["definition"]) for record in store.list_workers()
            )
        }
        snapshot = store.active_routing_snapshot()
        if snapshot is None or snapshot.get("format") != "modeldeck-event-routing":
            return {}
        routes: dict[str, list[ModelProfile]] = {}
        for route in snapshot.get("routes", []):
            if adapter_ids is not None and route.get("protocol_contract") not in adapter_ids:
                continue
            public_name = str(route.get("public_name", ""))
            if not public_name:
                continue
            routes[public_name] = [
                definitions[worker_id].to_profile()
                for worker_id in route.get("worker_ids", [])
                if worker_id in definitions
            ]
        return routes

    async def worker_states(routes: dict[str, list[ModelProfile]] | None = None) -> list[dict[str, Any]]:
        result = []
        profiles = {
            profile.id: profile
            for candidates in (routes if routes is not None else active_routes()).values()
            for profile in candidates
        }
        async with httpx.AsyncClient(timeout=0.3) as client:
            for profile in profiles.values():
                health, ready = await worker_health(client, profile)
                result.append(
                    {
                        "id": profile.id,
                        "name": next(
                            (
                                record["definition"]["name"]
                                for record in store.list_workers()
                                if record["definition"]["id"] == profile.id
                            ),
                            profile.id,
                        ),
                        "generation_family": profile.generation_family,
                        "endpoint": endpoint(profile),
                        "ready": ready,
                        "health": health,
                    }
                )
        return result

    @app.get("/v1/health")
    async def health():
        states = await worker_states()
        return {
            "status": "ok",
            "service": "modeldeck-gateway",
            "ready_workers": sum(worker["ready"] for worker in states),
        }

    @app.get("/v1/models")
    async def models():
        routes = active_routes()
        states = {state["id"]: state for state in await worker_states(routes)}
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "modeldeck-local",
                    "ready": any(states[profile.id]["ready"] for profile in candidates),
                }
                for alias, candidates in routes.items()
            ],
        }

    @app.get("/v1/capabilities")
    async def capabilities():
        routes = active_routes()
        return {
            alias: (candidates[0].capabilities.model_dump() if candidates else CapabilitySet().model_dump())
            for alias, candidates in routes.items()
        }

    @app.get("/v1/metrics")
    async def metrics(request: Request):
        return {"last_request": request.app.state.last_request_diagnostics}

    @app.get("/v1/routes")
    async def route_list():
        routes = active_routes()
        states = {state["id"]: state for state in await worker_states(routes)}
        return {
            "routes": [
                {
                    "public_name": name,
                    "ready": any(states[worker.id]["ready"] for worker in workers),
                }
                for name, workers in routes.items()
            ],
            "cloud_fallback": False,
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy_request(
            request,
            active_routes({"openai-chat-v1", "scene-analysis-v1"}),
            "/v1/chat/completions",
            None,
            timeout_seconds=configured.scenechat_timeout_seconds,
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy_request(request, active_routes({"openai-completions-v1"}), "/v1/completions", None)

    @app.post("/v1/translations")
    async def translations(request: Request):
        return await proxy_request(
            request,
            active_routes({"translation-en-fr-v1", "translation-en-de-v1"}),
            "/v1/translations",
            None,
            timeout_seconds=configured.translation_timeout_seconds,
        )

    @app.post("/v1/audio/speech")
    async def speech_synthesis(request: Request):
        return await proxy_binary_request(
            request,
            active_routes({"speech-synthesis-v1"}),
            "/v1/audio/speech",
            timeout_seconds=configured.speech_synthesis_timeout_seconds,
        )

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request):
        return await proxy_request(
            request,
            active_routes({"native-ar-trace-v1"}),
            "/native/autoregressive/trace",
            None,
        )

    @app.post("/v1/refine")
    async def refine(request: Request):
        return await proxy_request(
            request,
            active_routes({"text-diffusion-v1"}),
            "/v1/refine",
            None,
            timeout_seconds=configured.diffusion_timeout_seconds,
        )

    @app.post("/v1/diffuse")
    async def diffuse(request: Request):
        routes = active_routes({"text-diffusion-v1"})
        response = await proxy_request(request, routes, "/v1/diffuse", None)
        if isinstance(response, JSONResponse) and response.status_code < 300:
            payload = json_loads(response.body)
            if payload.get("job_id"):
                await resolve_job_worker(str(payload["job_id"]), job_routes, routes)
        return response

    @app.get("/v1/jobs/{job_id}")
    async def diffusion_job(job_id: str):
        worker = await resolve_job_worker(job_id, job_routes, active_routes())
        if worker is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(worker, f"/v1/jobs/{job_id}")

    @app.get("/v1/jobs/{job_id}/events")
    async def diffusion_job_events(job_id: str):
        worker = await resolve_job_worker(job_id, job_routes, active_routes())
        if worker is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_events(worker, f"/v1/jobs/{job_id}/events")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_diffusion_job(job_id: str):
        worker = await resolve_job_worker(job_id, job_routes, active_routes())
        if worker is None:
            raise HTTPException(404, "Unknown diffusion job")
        return await proxy_job_request(worker, f"/v1/jobs/{job_id}/cancel", method="POST")

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel(request_id: str, request: Request):
        async with request.app.state.active_request_lock:
            profile = request.app.state.active_request_workers.get(request_id)
        if profile is None:
            return {
                "ok": False,
                "request_id": request_id,
                "state": "not-found",
                "worker_id": None,
            }
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.post(f"{endpoint(profile)}/cancel", json={"request_id": request_id})
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return {
                "ok": False,
                "request_id": request_id,
                "state": "worker-unavailable",
                "worker_id": profile.id,
            }
        return {
            "ok": response.is_success and payload.get("ok") is True,
            "request_id": request_id,
            "state": payload.get("state", "cancelling" if payload.get("ok") else "not-found"),
            "worker_id": profile.id,
        }

    @app.post("/v1/vision/analyse")
    async def vision(request: Request):
        return await proxy_request(
            request,
            active_routes({"scene-analysis-v1"}),
            "/v1/chat/completions",
            None,
            timeout_seconds=configured.scenechat_timeout_seconds,
        )

    @app.websocket("/v1/speech/conversations")
    async def speech_conversation(client_socket: WebSocket):
        await client_socket.accept()
        try:
            first = await asyncio.wait_for(client_socket.receive_text(), timeout=5)
            start = json_loads(first.encode())
            alias = str(start.get("model") or "")
            candidates = route_candidates(active_routes({"speech-conversation-v1"}), alias)
            if not candidates:
                await client_socket.send_json({"type": "error", "code": "local_route_unavailable"})
                await client_socket.close(code=1013)
                return
            selected = None
            async with httpx.AsyncClient(timeout=0.5) as health_client:
                for candidate in candidates:
                    if candidate.generation_family.value != "speech-conversation":
                        continue
                    _, ready = await worker_health(health_client, candidate)
                    if ready:
                        selected = candidate
                        break
            if selected is None:
                await client_socket.send_json({"type": "error", "code": "local_route_unavailable"})
                await client_socket.close(code=1013)
                return
            public_name = alias
            start["model"] = selected.alias
            from websockets.asyncio.client import connect

            async with connect(
                f"ws://127.0.0.1:{selected.port}/v1/speech/conversations",
                max_size=96_000,
            ) as upstream:
                await upstream.send(json.dumps(start))

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
                            event = json.loads(message)
                            if isinstance(event, dict) and event.get("model") == selected.alias:
                                event["model"] = public_name
                            await client_socket.send_text(json.dumps(event))

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


async def proxy_request(
    request: Request,
    routes: dict[str, list[ModelProfile]],
    path: str,
    default_alias: str | None,
    *,
    timeout_seconds: float = 60.0,
):
    started = time.perf_counter()
    body = await request.json()
    alias = str(body.get("model") or default_alias or "")
    candidates = route_candidates(routes, alias)
    if not candidates:
        _record_gateway_diagnostic(request, alias, started, "error", "local_route_unavailable")
        return unavailable(alias, "unknown")
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=0.5))
    selected = None
    request_id = str(body.get("request_id") or "")
    request_claimed = False
    health_results: list[dict[str, Any] | None] = []
    try:
        for profile in candidates:
            health, ready = await worker_health(client, profile)
            health_results.append(health)
            if ready:
                selected = profile
                break
        if selected is None:
            await client.aclose()
            if health_results and all(
                health is not None and health.get("busy") is True for health in health_results
            ):
                response = gateway_error(
                    429,
                    "worker_busy",
                    f"Every ready-capable Worker for Route '{alias}' is currently busy.",
                    alias,
                )
                _record_gateway_diagnostic(request, alias, started, "error", "worker_busy")
                return response
            _record_gateway_diagnostic(request, alias, started, "error", "local_route_unavailable")
            return unavailable(alias, candidates[0].generation_family.value)
        if request_id:
            request_claimed = await claim_active_request(request, request_id, selected)
            if not request_claimed:
                await client.aclose()
                return gateway_error(
                    409,
                    "duplicate_request_id",
                    f"Request ID '{request_id}' is already active.",
                    alias,
                )
        body["model"] = selected.alias if path == "/v1/translations" else upstream_model(selected, alias)
        upstream = client.build_request(
            "POST",
            f"{endpoint(selected)}{path}",
            json=body,
            headers=upstream_headers(selected, request_id),
        )
        response = await client.send(upstream, stream=bool(body.get("stream")))
    except httpx.TimeoutException:
        await client.aclose()
        if request_claimed:
            await release_active_request(request, request_id, selected)
        _record_gateway_diagnostic(request, alias, started, "error", "gateway_timeout")
        return gateway_error(
            504,
            "gateway_timeout",
            f"The local Worker for Route '{alias}' did not respond within the gateway deadline.",
            alias,
        )
    except (httpx.HTTPError, ValueError):
        await client.aclose()
        if request_claimed:
            await release_active_request(request, request_id, selected)
        _record_gateway_diagnostic(request, alias, started, "error", "local_route_unavailable")
        return unavailable(alias, candidates[0].generation_family.value)
    if body.get("stream"):
        return StreamingResponse(
            forward_stream(
                response,
                client,
                (lambda: release_active_request(request, request_id, selected) if request_claimed else None),
            ),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "text/event-stream"),
            headers=worker_response_headers(selected),
        )
    try:
        payload = await response.aread()
        response_payload = json_loads(payload)
        if path == "/v1/translations" and response.is_success and isinstance(response_payload, dict):
            response_payload["model"] = alias
        if path == "/native/autoregressive/trace" and response.is_success:
            metadata_error = trace_token_metadata_error(response_payload)
            if metadata_error:
                return invalid_trace_metadata(selected.id, metadata_error)
        error_code = None
        if not response.is_success and isinstance(response_payload, dict):
            error = response_payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                error_code = error["code"]
        _record_gateway_diagnostic(
            request,
            alias,
            started,
            "success" if response.is_success else "error",
            error_code,
        )
        return JSONResponse(
            response_payload,
            status_code=response.status_code,
            headers=worker_response_headers(selected),
        )
    finally:
        await response.aclose()
        await client.aclose()
        if request_claimed:
            await release_active_request(request, request_id, selected)


async def proxy_binary_request(
    request: Request,
    routes: dict[str, list[ModelProfile]],
    path: str,
    *,
    timeout_seconds: float,
) -> Response:
    started = time.perf_counter()
    try:
        body = await request.json()
    except ValueError:
        return gateway_error(422, "invalid_request", "The request body must be JSON.", "")
    alias = str(body.get("model") or "")
    request_id = str(body.get("request_id") or "")
    if not request_id:
        return gateway_error(422, "invalid_request_id", "Supply a caller-generated request_id.", alias)
    candidates = route_candidates(routes, alias)
    if not candidates:
        return unavailable(alias, GenerationFamily.SPEECH_SYNTHESIS.value)
    selected: ModelProfile | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=0.5)) as client:
        for profile in candidates:
            health, ready = await worker_health(client, profile)
            if ready:
                selected = profile
                break
        if selected is None:
            _record_gateway_diagnostic(request, alias, started, "error", "local_route_unavailable")
            return unavailable(alias, GenerationFamily.SPEECH_SYNTHESIS.value)
        if not await claim_active_request(request, request_id, selected):
            return gateway_error(
                409,
                "duplicate_request_id",
                f"Request ID '{request_id}' is already active.",
                alias,
            )
        body["model"] = selected.alias
        try:
            upstream = await client.post(
                f"{endpoint(selected)}{path}",
                json=body,
                headers=upstream_headers(selected, request_id),
            )
        except httpx.TimeoutException:
            _record_gateway_diagnostic(request, alias, started, "error", "gateway_timeout")
            return gateway_error(
                504,
                "gateway_timeout",
                f"The local Worker for Route '{alias}' did not respond within the gateway deadline.",
                alias,
            )
        except httpx.HTTPError:
            _record_gateway_diagnostic(request, alias, started, "error", "local_route_unavailable")
            return unavailable(alias, GenerationFamily.SPEECH_SYNTHESIS.value)
        finally:
            await release_active_request(request, request_id, selected)
    response_headers = {
        key: upstream.headers[key]
        for key in (
            "x-request-id",
            "x-modeldeck-sample-rate-hz",
            "x-modeldeck-audio-duration-seconds",
        )
        if key in upstream.headers
    }
    response_headers.update(worker_response_headers(selected))
    if not upstream.is_success:
        try:
            payload = upstream.json()
        except ValueError:
            payload = {
                "error": {
                    "code": "invalid_worker_response",
                    "message": "The local speech synthesis Worker returned an invalid error response.",
                }
            }
        error_code = (
            payload.get("error", {}).get("code")
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict)
            else None
        )
        _record_gateway_diagnostic(request, alias, started, "error", error_code)
        return JSONResponse(payload, status_code=upstream.status_code, headers=response_headers)
    if upstream.headers.get("content-type", "").split(";", 1)[0].strip() != "audio/wav":
        _record_gateway_diagnostic(request, alias, started, "error", "invalid_worker_audio")
        return gateway_error(
            502,
            "invalid_worker_audio",
            "The local speech synthesis Worker did not return WAV audio.",
            alias,
        )
    _record_gateway_diagnostic(request, alias, started, "success", None)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="audio/wav",
        headers=response_headers,
    )


async def claim_active_request(request: Request, request_id: str, profile: ModelProfile) -> bool:
    async with request.app.state.active_request_lock:
        if request_id in request.app.state.active_request_workers:
            return False
        request.app.state.active_request_workers[request_id] = profile
        return True


async def release_active_request(request: Request, request_id: str, profile: ModelProfile | None) -> None:
    async with request.app.state.active_request_lock:
        if request.app.state.active_request_workers.get(request_id) is profile:
            request.app.state.active_request_workers.pop(request_id, None)


async def forward_stream(
    response: httpx.Response,
    client: httpx.AsyncClient,
    on_complete: Callable[[], Awaitable[None] | None] | None = None,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()
        if on_complete is not None:
            completion = on_complete()
            if completion is not None:
                await completion


async def resolve_job_worker(
    job_id: str,
    job_routes: dict[str, ModelProfile],
    routes: dict[str, list[ModelProfile]],
) -> ModelProfile | None:
    if worker := job_routes.get(job_id):
        return worker
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
    worker: ModelProfile,
    path: str,
    *,
    method: str = "GET",
) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=0.5)) as client:
            response = await client.request(method, f"{endpoint(worker)}{path}")
            payload = json_loads(await response.aread())
    except (httpx.HTTPError, ValueError):
        return unavailable("text-diffusion", "text-diffusion")
    return JSONResponse(
        payload,
        status_code=response.status_code,
        headers={},
    )


async def proxy_job_events(worker: ModelProfile, path: str) -> StreamingResponse | JSONResponse:
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=0.5))
    try:
        response = await client.send(
            client.build_request("GET", f"{endpoint(worker)}{path}"),
            stream=True,
        )
    except httpx.HTTPError:
        await client.aclose()
        return unavailable("text-diffusion", "text-diffusion")
    return StreamingResponse(
        forward_stream(response, client),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
        headers={},
    )


async def worker_health(
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


def upstream_headers(profile: ModelProfile, request_id: str = "") -> dict[str, str]:
    headers = {}
    if profile.generation_family.value == "vision-language":
        headers["Authorization"] = "Bearer " + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")
    if request_id:
        headers["X-Request-ID"] = request_id
    return headers


def worker_response_headers(profile: ModelProfile) -> dict[str, str]:
    headers = {}
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
                "code": "local_route_unavailable",
                "message": (
                    f"No ready Worker supplies Route '{alias}'."
                    if alias
                    else "Supply the public Route name in the model field."
                ),
                "route": alias or None,
                "required_generation_family": family,
                "cloud_fallback_attempted": False,
            }
        },
        status_code=503,
    )


def gateway_error(status_code: int, code: str, message: str, alias: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": code,
                "message": message,
                "route": alias or None,
                "cloud_fallback_attempted": False,
            }
        },
        status_code=status_code,
    )


def _record_gateway_diagnostic(
    request: Request,
    alias: str,
    started: float,
    outcome: str,
    error_code: str | None,
) -> None:
    request.app.state.last_request_diagnostics = {
        "route": alias or None,
        "total_gateway_seconds": round(time.perf_counter() - started, 6),
        "outcome": outcome,
        "error_code": error_code,
    }


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


def invalid_trace_metadata(worker_id: str, reason: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "invalid_worker_trace_metadata",
                "message": f"Local Worker '{worker_id}' returned invalid trace token metadata: {reason}.",
                "worker_id": worker_id,
            }
        },
        status_code=502,
        headers={},
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
