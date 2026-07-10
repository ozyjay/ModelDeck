from __future__ import annotations

from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from modeldeck.config import Settings
from modeldeck.profiles import default_model_profiles


def create_gateway_app() -> FastAPI:
    app = FastAPI(title="ModelDeck stable local gateway", version="0.1.0")
    profiles = {profile.alias: profile for profile in default_model_profiles()}

    async def providers() -> list[dict[str, Any]]:
        result = []
        async with httpx.AsyncClient(timeout=0.3) as client:
            for profile in profiles.values():
                try:
                    response = await client.get(f"http://127.0.0.1:{profile.port}/health")
                    health = response.json()
                    ready = response.is_success and health.get("ready") is True
                except (httpx.HTTPError, ValueError):
                    health, ready = None, False
                result.append(
                    {
                        "id": profile.id,
                        "alias": profile.alias,
                        "generation_family": profile.generation_family,
                        "endpoint": f"http://127.0.0.1:{profile.port}",
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
            "ready_providers": sum(p["ready"] for p in states),
        }

    @app.get("/v1/models")
    async def models():
        states = await providers()
        return {
            "object": "list",
            "data": [
                {
                    "id": profile.alias,
                    "object": "model",
                    "owned_by": "modeldeck-local",
                    "ready": state["ready"],
                }
                for profile, state in zip(profiles.values(), states, strict=True)
            ],
        }

    @app.get("/v1/capabilities")
    async def capabilities():
        return {alias: profile.capabilities.model_dump() for alias, profile in profiles.items()}

    @app.get("/v1/providers")
    async def provider_list():
        return {"providers": await providers(), "cloud_fallback": False}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy_json(request, profiles["fast-chat"], "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy_json(request, profiles["fast-chat"], "/v1/completions")

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request):
        return await proxy_json(request, profiles["fast-chat"], "/native/autoregressive/trace")

    @app.post("/v1/refine")
    async def refine(request: Request):
        return await proxy_json(request, profiles["text-diffusion"], "/v1/refine")

    @app.post("/v1/diffuse")
    async def diffuse(request: Request):
        return await proxy_json(request, profiles["text-diffusion"], "/v1/diffuse")

    @app.post("/v1/vision/analyse")
    async def vision():
        raise HTTPException(501, "No vision worker is implemented in this slice")

    return app


async def proxy_json(request: Request, profile, path: str):
    body = await request.json()
    body.setdefault("model", profile.alias)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            health = await client.get(f"http://127.0.0.1:{profile.port}/health")
            if not health.is_success or health.json().get("ready") is not True:
                return unavailable(profile.alias, profile.generation_family.value)
            response = await client.post(f"http://127.0.0.1:{profile.port}{path}", json=body)
    except (httpx.HTTPError, ValueError):
        return unavailable(profile.alias, profile.generation_family.value)
    return JSONResponse(response.json(), status_code=response.status_code)


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


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(create_gateway_app(), host=settings.host, port=settings.gateway_port, log_level="info")


if __name__ == "__main__":
    main()
