from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import (
    invalid_trace_metadata,
    proxy_binary_request,
    proxy_request,
    route_candidates,
    trace_token_metadata_error,
    upstream_headers,
    upstream_model,
)
from starlette.requests import Request


def worker() -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name="Trace Worker",
        model_id="example/model",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="mock",
        lifecycle="on-demand",
        port=8630,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


@pytest.mark.asyncio
async def test_gateway_has_no_routes_or_implicit_defaults_before_publication(tmp_path) -> None:
    app = create_gateway_app(settings=Settings(data_dir=tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = await client.get("/v1/models")
        unavailable = await client.post("/v1/completions", json={"prompt": "hello"})

    assert models.json() == {"object": "list", "data": []}
    assert unavailable.status_code == 503
    assert unavailable.json()["error"] == {
        "code": "local_route_unavailable",
        "message": "Supply the public Route name in the model field.",
        "route": None,
        "required_generation_family": "unknown",
        "cloud_fallback_attempted": False,
    }


@pytest.mark.asyncio
async def test_gateway_advertises_only_routes_from_active_event_snapshot(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    definition = worker()
    store.save_worker_definition(definition.model_dump(mode="json"))
    event_id = str(uuid4())
    event = {
        "id": event_id,
        "name": "Open Day",
        "description": "",
        "qualification": "compatible",
        "demos": [],
        "routes": [],
    }
    store.save_event_draft(event)
    store.publish_event(
        event,
        {
            "format": "modeldeck-event-routing",
            "version": 2,
            "event_id": event_id,
            "event_name": "Open Day",
            "revision": 0,
            "routes": [
                {
                    "route_id": str(uuid4()),
                    "display_name": "Visitor trace",
                    "public_name": "visitor-trace",
                    "protocol_contract": "native-ar-trace-v1",
                    "worker_ids": [definition.id],
                }
            ],
        },
    )
    app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = (await client.get("/v1/models")).json()["data"]
        routes = (await client.get("/v1/routes")).json()["routes"]

    assert models == [
        {"id": "visitor-trace", "object": "model", "owned_by": "modeldeck-local", "ready": False}
    ]
    assert routes == [{"public_name": "visitor-trace", "ready": False}]
    assert "provider" not in str(models).lower()


def test_trace_metadata_validation_requires_aligned_readable_tokens() -> None:
    valid = {
        "prompt_token_ids": [1, 2],
        "prompt_tokens": ["one", "two"],
        "user_prompt_token_ids": [2],
        "user_prompt_tokens": ["two"],
    }
    assert trace_token_metadata_error(valid) is None
    assert "align" in trace_token_metadata_error({**valid, "prompt_tokens": ["one"]})


def test_invalid_trace_metadata_error_uses_worker_language_without_route_leakage() -> None:
    response = invalid_trace_metadata("worker-id", "tokens do not align")
    assert response.status_code == 502
    assert b"Local Worker" in response.body
    assert b'"worker_id":"worker-id"' in response.body
    assert "x-modeldeck-provider" not in response.headers


def test_vision_translation_keeps_internal_model_and_scoped_credential(monkeypatch) -> None:
    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "vision-language",
                "capabilities": {"image_input": True, "structured_output": True},
            }
        )
        .to_profile()
    )
    monkeypatch.setenv("MODELDECK_SCENECHAT_API_KEY", "test-key")
    assert upstream_model(profile, "visitor-scene") == "example/model"
    assert upstream_headers(profile) == {"Authorization": "Bearer test-key"}


def test_route_candidates_accept_only_public_route_or_vision_model_identity() -> None:
    profile = worker().to_profile()
    routes = {"visitor-trace": [profile]}
    assert route_candidates(routes, "visitor-trace") == [profile]
    assert route_candidates(routes, profile.model_id) is None


def gateway_request(payload: dict) -> Request:
    encoded = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        return {"type": "http.disconnect"}

    app = SimpleNamespace(
        state=SimpleNamespace(
            last_request_diagnostics=None,
            active_request_workers={},
            active_request_lock=asyncio.Lock(),
        )
    )
    return Request({"type": "http", "method": "POST", "headers": [], "app": app}, receive)


class FakeGatewayClient:
    def __init__(
        self,
        health: dict,
        *,
        timeout: bool = False,
        response_status: int = 200,
        response_payload: dict | None = None,
    ) -> None:
        self.health = health
        self.timeout = timeout
        self.response_status = response_status
        self.response_payload = response_payload or {"ok": True}

    async def get(self, _url: str):
        return SimpleNamespace(
            json=lambda: self.health,
            is_success=True,
        )

    def build_request(self, method: str, url: str, **kwargs):
        return httpx.Request(method, url, **kwargs)

    async def send(self, request: httpx.Request, *, stream: bool = False):
        if self.timeout:
            raise httpx.ReadTimeout("benchmark deadline", request=request)
        return httpx.Response(
            self.response_status,
            json=self.response_payload,
            request=request,
        )

    async def aclose(self) -> None:
        pass


class FakeBinaryGatewayClient:
    def __init__(self, profile, payload: bytes) -> None:
        self.profile = profile
        self.payload = payload
        self.forwarded_json = None
        self.forwarded_headers = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _url: str):
        return SimpleNamespace(json=lambda: {"ready": True, "busy": False}, is_success=True)

    async def post(self, url: str, *, json: dict, headers: dict):
        self.forwarded_json = json
        self.forwarded_headers = headers
        return httpx.Response(
            200,
            content=self.payload,
            headers={
                "content-type": "audio/wav",
                "x-request-id": json["request_id"],
                "x-modeldeck-sample-rate-hz": "24000",
            },
            request=httpx.Request("POST", url),
        )


class FakeCancellationClient:
    def __init__(self) -> None:
        self.url = ""
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url: str, *, json: dict):
        self.url = url
        self.payload = json
        return SimpleNamespace(
            is_success=True,
            json=lambda: {"ok": True, "request_id": json["request_id"], "state": "cancelling"},
        )


@pytest.mark.asyncio
async def test_gateway_distinguishes_busy_worker_from_unavailable(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "vision-language",
                "capabilities": {"image_input": True, "structured_output": True},
            }
        )
        .to_profile()
    )
    fake = FakeGatewayClient({"ready": False, "busy": True})
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request({"model": "scenechat-vision"})

    response = await proxy_request(
        request,
        {"scenechat-vision": [profile]},
        "/v1/chat/completions",
        None,
    )

    assert response.status_code == 429
    assert json.loads(response.body)["error"]["code"] == "worker_busy"
    assert request.app.state.last_request_diagnostics["error_code"] == "worker_busy"


@pytest.mark.asyncio
async def test_gateway_reports_its_own_timeout_distinctly(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "vision-language",
                "capabilities": {"image_input": True, "structured_output": True},
            }
        )
        .to_profile()
    )
    fake = FakeGatewayClient({"ready": True, "busy": False}, timeout=True)
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request({"model": "scenechat-vision"})

    response = await proxy_request(
        request,
        {"scenechat-vision": [profile]},
        "/v1/chat/completions",
        None,
        timeout_seconds=120,
    )

    assert response.status_code == 504
    assert json.loads(response.body)["error"]["code"] == "gateway_timeout"
    diagnostic = request.app.state.last_request_diagnostics
    assert diagnostic["error_code"] == "gateway_timeout"
    assert diagnostic["total_gateway_seconds"] >= 0


@pytest.mark.asyncio
async def test_translation_gateway_preserves_public_model_and_forwards_internal_alias(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "text-translation",
                "capabilities": {"translation": True, "cancellation": True},
                "settings": {"source_language": "en", "target_language": "fr"},
            }
        )
        .to_profile()
    )
    fake = FakeGatewayClient(
        {"ready": True, "busy": False},
        response_payload={
            "id": "translation-1",
            "object": "translation",
            "model": profile.alias,
            "source_language": "en",
            "target_language": "fr",
            "output_text": "Bonjour",
        },
    )
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request(
        {
            "request_id": "translation-1",
            "model": "visitor-translation",
            "input": "Hello",
            "source_language": "en",
            "target_language": "fr",
        }
    )

    response = await proxy_request(
        request,
        {"visitor-translation": [profile]},
        "/v1/translations",
        None,
    )

    payload = json.loads(response.body)
    assert payload["model"] == "visitor-translation"
    assert payload["output_text"] == "Bonjour"
    assert request.app.state.active_request_workers == {}


@pytest.mark.asyncio
async def test_speech_gateway_returns_wav_and_labels_a_mock_fallback(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "speech-synthesis",
                "capabilities": {
                    "speech_synthesis": True,
                    "audio_output": True,
                    "cancellation": True,
                    "streaming": False,
                },
                "settings": {"sample_rate_hz": 24_000},
            }
        )
        .to_profile()
    )
    fake = FakeBinaryGatewayClient(profile, b"RIFFmock-wav")
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request(
        {
            "request_id": "speech-1",
            "model": "visitor-voice",
            "input": "Hello",
            "voice": "ryan",
            "language": "en",
            "response_format": "wav",
        }
    )

    response = await proxy_binary_request(
        request,
        {"visitor-voice": [profile]},
        "/v1/audio/speech",
        timeout_seconds=120,
    )

    assert response.status_code == 200
    assert response.body == b"RIFFmock-wav"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-modeldeck-fallback"] == "mock"
    assert fake.forwarded_json["model"] == profile.alias
    assert fake.forwarded_headers["X-Request-ID"] == "speech-1"
    assert request.app.state.active_request_workers == {}


@pytest.mark.asyncio
async def test_cancellation_targets_only_the_worker_that_owns_the_active_request(
    monkeypatch, tmp_path
) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = worker().to_profile()
    gateway = create_gateway_app(
        {"visitor-chat": [profile]},
        settings=Settings(data_dir=tmp_path),
    )
    gateway.state.active_request_workers["active-1"] = profile
    fake = FakeCancellationClient()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
    ) as client:
        monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
        response = await client.post("/v1/requests/active-1/cancel")

    assert response.json() == {
        "ok": True,
        "request_id": "active-1",
        "state": "cancelling",
        "worker_id": profile.id,
    }
    assert fake.url == f"http://127.0.0.1:{profile.port}/cancel"
    assert fake.payload == {"request_id": "active-1"}


@pytest.mark.asyncio
async def test_gateway_propagates_mock_failure_and_labels_fallback(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = worker().to_profile()
    fake = FakeGatewayClient(
        {"ready": True, "busy": False},
        response_status=503,
        response_payload={
            "error": {
                "code": "mock_request_failure",
                "message": "Deterministic failure",
            }
        },
    )
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request({"model": "visitor-chat", "messages": []})

    response = await proxy_request(
        request,
        {"visitor-chat": [profile]},
        "/v1/chat/completions",
        None,
    )

    assert response.status_code == 503
    assert json.loads(response.body)["error"]["code"] == "mock_request_failure"
    assert response.headers["x-modeldeck-fallback"] == "mock"
    assert request.app.state.last_request_diagnostics["error_code"] == "mock_request_failure"
