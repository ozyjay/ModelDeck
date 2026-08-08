from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import RoutingProfile, WorkerDefinition, routing_snapshot
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import (
    claim_thermal_capacity,
    invalid_trace_metadata,
    model_discovery_record,
    proxy_binary_request,
    proxy_request,
    release_thermal_capacity,
    resolve_job_worker,
    route_candidates,
    trace_token_metadata_error,
    upstream_headers,
    upstream_model,
)
from modeldeck.thermal import (
    THERMAL_STATUS_FILENAME,
    ThermalPolicyConfig,
    WorkloadClass,
    WorkloadRequest,
    write_thermal_status,
)
from starlette.requests import Request
from starlette.responses import StreamingResponse


def worker() -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name="Trace Worker",
        model_id="example/model",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="mock",
        lifecycle="on-demand",
        port=65535,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


def published_chat_profile(worker_id: str, *, profile_id: str | None = None) -> RoutingProfile:
    return RoutingProfile(
        id=profile_id or str(uuid4()),
        name="Local applications",
        capabilities=[
            {
                "id": str(uuid4()),
                "display_name": "Visitor chat",
                "public_name": "visitor-chat",
                "protocol_contract": "openai-chat-v1",
                "worker_ids": [worker_id],
            }
        ],
    )


async def listed_model_revision(settings: Settings) -> str:
    app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = (await client.get("/v1/models")).json()["data"]
    assert len(models) == 1
    return models[0]["revision"]


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
async def test_gateway_advertises_openai_models_and_native_capabilities_separately(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    definition = worker()
    store.save_worker_definition(definition.model_dump(mode="json"))
    profile = RoutingProfile(
        id=str(uuid4()),
        name="Local applications",
        capabilities=[
            {
                "id": str(uuid4()),
                "display_name": "Visitor trace",
                "public_name": "visitor-trace",
                "protocol_contract": "native-ar-trace-v1",
                "worker_ids": [definition.id],
            },
            {
                "id": str(uuid4()),
                "display_name": "Visitor chat",
                "public_name": "visitor-chat",
                "protocol_contract": "openai-chat-v1",
                "worker_ids": [definition.id],
            },
        ],
    )
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))
    app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = (await client.get("/v1/models")).json()["data"]
        routes = (await client.get("/v1/routes")).json()["routes"]
        native = (await client.get("/native/v1/capabilities")).json()["capabilities"]

    assert models == [
        {
            "id": "visitor-chat",
            "object": "model",
            "owned_by": "modeldeck-local",
            "revision": "revision-1",
            "ready": False,
            "runtime": "mock",
            "accelerator": "mock",
        }
    ]
    assert routes == [
        {"public_name": "visitor-trace", "ready": False},
        {"public_name": "visitor-chat", "ready": False},
    ]
    assert native == [
        {
            "id": profile.capabilities[0].id,
            "display_name": "Visitor trace",
            "public_name": "visitor-trace",
            "protocol_contract": "native-ar-trace-v1",
            "surfaces": ["POST /native/v1/autoregressive/traces"],
            "ready": False,
            "metadata": {"generation_family": "autoregressive", "worker_count": 1},
        }
    ]
    assert "provider" not in str(models).lower()


@pytest.mark.asyncio
async def test_gateway_model_revision_is_stable_across_restart_and_changes_with_snapshot(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    original = worker()
    store.save_worker_definition(original.model_dump(mode="json"))
    profile = published_chat_profile(original.id)
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))

    assert await listed_model_revision(settings) == "revision-1"
    # A new gateway instance reads the immutable Worker revision from the persisted profile.
    assert await listed_model_revision(settings) == "revision-1"

    replacement = worker().model_copy(update={"name": "Trace Worker revision 2", "revision": "revision-2"})
    store.save_worker_definition(replacement.model_dump(mode="json"))
    updated_capability = profile.capabilities[0].model_copy(update={"worker_ids": [replacement.id]})
    updated_profile = profile.model_copy(update={"capabilities": [updated_capability]})
    store.save_routing_profile_draft(updated_profile.model_dump(mode="json"))
    store.publish_routing_profile(
        updated_profile.model_dump(mode="json"), routing_snapshot(updated_profile, 2)
    )

    assert await listed_model_revision(settings) == "revision-2"


@pytest.mark.asyncio
async def test_gateway_model_revision_uses_and_tracks_loaded_derivative_artifact(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    original = worker().model_copy(
        update={
            "model_id": "example/base-model",
            "revision": "base-revision",
            "artifact_model_id": "example/quantised-model",
            "artifact_revision": "artifact-revision-1",
        }
    )
    store.save_worker_definition(original.model_dump(mode="json"))
    profile = published_chat_profile(original.id)
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))

    assert await listed_model_revision(settings) == "artifact-revision-1"

    replacement = worker().model_copy(
        update={
            "name": "Trace Worker artefact revision 2",
            "model_id": "example/base-model",
            "revision": "base-revision",
            "artifact_model_id": "example/quantised-model",
            "artifact_revision": "artifact-revision-2",
        }
    )
    store.save_worker_definition(replacement.model_dump(mode="json"))
    updated_capability = profile.capabilities[0].model_copy(update={"worker_ids": [replacement.id]})
    updated_profile = profile.model_copy(update={"capabilities": [updated_capability]})
    store.save_routing_profile_draft(updated_profile.model_dump(mode="json"))
    store.publish_routing_profile(
        updated_profile.model_dump(mode="json"), routing_snapshot(updated_profile, 2)
    )

    assert await listed_model_revision(settings) == "artifact-revision-2"


def test_trace_metadata_validation_requires_aligned_readable_tokens() -> None:
    valid = {
        "prompt_token_ids": [1, 2],
        "prompt_tokens": ["one", "two"],
        "user_prompt_token_ids": [2],
        "user_prompt_tokens": ["two"],
    }
    assert trace_token_metadata_error(valid) is None
    assert "align" in trace_token_metadata_error({**valid, "prompt_tokens": ["one"]})


def test_model_discovery_reports_ready_rocm_worker_accelerator_metadata() -> None:
    profile = worker().model_copy(update={"runtime": "transformers-rocm"}).to_profile()

    record = model_discovery_record(
        "sprintbot-qwen",
        [profile],
        {
            profile.id: {
                "ready": True,
                "health": {"runtime": "transformers-rocm", "rocm_version": "7.2.1"},
            }
        },
    )

    assert record["runtime"] == "transformers-rocm"
    assert record["accelerator"] == "rocm"
    assert record["ready"] is True


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


@pytest.mark.asyncio
async def test_diffusion_job_assignment_preserves_worker_affinity_after_gateway_restart(tmp_path) -> None:
    definition = worker().model_copy(
        update={
            "generation_family": "text-diffusion",
            "capabilities": {"iterative_refinement": True, "intermediate_frames": True},
        }
    )
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v3()
    store.save_worker_definition(definition.model_dump(mode="json"))
    store.save_gateway_job_assignment("job-1", definition.id, "text-diffusion", "text-diffusion-v1")

    resolved = await resolve_job_worker("job-1", {}, {}, store)

    assert resolved is not None
    assert resolved.id == definition.id


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


@pytest.mark.asyncio
async def test_gateway_queues_shared_heavy_capacity_with_a_bounded_retry(tmp_path) -> None:
    thermal = ThermalPolicyConfig(
        recovery_step_seconds=0.01,
        minimum_state_dwell_seconds=0,
        host_policy_status_enabled=False,
    )
    app = create_gateway_app({}, Settings(data_dir=tmp_path, thermal_throttling=thermal))
    write_thermal_status(
        tmp_path / THERMAL_STATUS_FILENAME,
        {
            "state": "normal",
            "published_monotonic": time.monotonic(),
            "telemetry_age_seconds": 0,
            "heavy_concurrency_limit": 1,
        },
    )
    request = Request({"type": "http", "method": "POST", "headers": [], "app": app})
    workload = WorkloadRequest("first", WorkloadClass.INTERACTIVE)

    _, first_claimed = await claim_thermal_capacity(request, workload)
    queued, second_claimed = await claim_thermal_capacity(
        request, WorkloadRequest("second", WorkloadClass.INTERACTIVE)
    )
    await release_thermal_capacity(request)

    assert first_claimed is True
    assert second_claimed is False
    assert queued.reason_code == "thermal_queue_timeout"
    assert app.state.thermal_queued_requests == 1


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


class FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"data: first\n\n"
        raise httpx.ReadError("stream ended unexpectedly")

    async def aclose(self) -> None:
        pass


class StreamingFailureClient(FakeGatewayClient):
    def __init__(self) -> None:
        super().__init__({"ready": True, "busy": False})
        self.health_urls: list[str] = []
        self.send_urls: list[str] = []

    async def get(self, url: str):
        self.health_urls.append(url)
        return await super().get(url)

    async def send(self, request: httpx.Request, *, stream: bool = False):
        self.send_urls.append(str(request.url))
        return httpx.Response(200, stream=FailingStream(), request=request)


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
async def test_gateway_never_fails_over_after_a_stream_has_started(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    primary = worker().to_profile()
    backup = worker().model_copy(update={"id": str(uuid4()), "port": 8631}).to_profile()
    fake = StreamingFailureClient()
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request({"model": "visitor-chat", "prompt": "hello", "stream": True})

    response = await proxy_request(
        request,
        {"visitor-chat": [primary, backup]},
        "/v1/completions",
        None,
    )

    assert isinstance(response, StreamingResponse)
    assert await anext(response.body_iterator) == b"data: first\n\n"
    with pytest.raises(httpx.ReadError, match="stream ended unexpectedly"):
        await anext(response.body_iterator)
    assert fake.health_urls == [f"http://127.0.0.1:{primary.port}/health"]
    assert fake.send_urls == [f"http://127.0.0.1:{primary.port}/v1/completions"]


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
async def test_recognition_gateway_preserves_public_model_and_forwards_bounded_audio(monkeypatch) -> None:
    import modeldeck.gateway.app as gateway_module

    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "speech-recognition",
                "capabilities": {
                    "speech_recognition": True,
                    "audio_input": True,
                    "cancellation": True,
                    "streaming": False,
                },
                "settings": {"sample_rate_hz": 16_000, "channels": 1},
            }
        )
        .to_profile()
    )
    fake = FakeGatewayClient(
        {"ready": True, "busy": False},
        response_payload={
            "id": "recognition-1",
            "object": "audio.transcription",
            "model": profile.alias,
            "language": "en",
            "text": "Ready.",
            "metrics": {"audio_seconds": 0.1, "inference_seconds": 0.01},
        },
    )
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", lambda *args, **kwargs: fake)
    request = gateway_request(
        {
            "request_id": "recognition-1",
            "model": "speechshift-stt",
            "language": "en",
            "encoding": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "audio_base64": base64.b64encode(bytes(3200)).decode("ascii"),
        }
    )

    response = await proxy_request(
        request,
        {"speechshift-stt": [profile]},
        "/v1/audio/transcriptions",
        None,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["model"] == "speechshift-stt"
    assert request.app.state.active_request_workers == {}


@pytest.mark.asyncio
async def test_recognition_gateway_rejects_oversized_audio_before_routing(tmp_path) -> None:
    app = create_gateway_app({}, settings=Settings(data_dir=tmp_path))
    oversized = base64.b64encode(bytes(256002)).decode("ascii")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            json={"model": "speechshift-stt", "audio_base64": oversized},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_audio"


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
    assert "x-modeldeck-fallback" not in response.headers
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
async def test_gateway_propagates_fixture_worker_failure_without_fallback_header(monkeypatch) -> None:
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
    assert "x-modeldeck-fallback" not in response.headers
    assert request.app.state.last_request_diagnostics["error_code"] == "mock_request_failure"
