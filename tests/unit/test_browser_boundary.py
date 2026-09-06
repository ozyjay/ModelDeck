from __future__ import annotations

import httpx
import pytest
from modeldeck.config import Settings
from modeldeck.gateway import create_gateway_app
from modeldeck.main import create_app
from modeldeck.security import LocalBrowserBoundary, is_trusted_browser_origin


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "null",
        "https://evil.invalid",
        "http://[",
        "http://localhost:bad",
        "http://localhost:0",
        "http://localhost/#fragment",
        "http://user@localhost",
        "http://localhost\\evil",
        "http://local\nhost",
    ],
)
def test_invalid_origins_are_rejected(origin):
    assert not is_trusted_browser_origin(origin)


@pytest.mark.parametrize("factory", [create_app, create_gateway_app])
@pytest.mark.parametrize(
    "host",
    [
        "evil.invalid",
        "localhost.evil.invalid",
        "127.0.0.1@evil.invalid",
        "[",
        "localhost:bad",
        "localhost/path",
    ],
)
async def test_rebinding_hosts_are_rejected_before_handlers(factory, host):
    app = factory(settings=Settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/", headers={"Host": host})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "headers,accepted",
    [
        ([(b"host", b"localhost")], True),
        ([(b"host", b"[::1]:8600"), (b"origin", b"http://localhost:3600")], True),
        ([(b"host", b"localhost"), (b"origin", b"")], False),
        ([(b"host", b"localhost"), (b"origin", b"null")], False),
        ([(b"host", b"localhost"), (b"origin", b"http://evil.invalid")], False),
        ([(b"host", b"evil.invalid")], False),
        ([(b"host", b"localhost"), (b"host", b"evil.invalid")], False),
        (
            [(b"host", b"localhost"), (b"origin", b"http://localhost"), (b"origin", b"http://evil.invalid")],
            False,
        ),
    ],
)
async def test_websocket_boundary_checks_before_accept(headers, accepted):
    messages = []

    async def endpoint(scope, receive, send):
        await send({"type": "websocket.accept"})

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        messages.append(message)

    await LocalBrowserBoundary(endpoint)({"type": "websocket", "headers": headers}, receive, send)
    assert messages == (
        [{"type": "websocket.accept"}] if accepted else [{"type": "websocket.close", "code": 1008}]
    )


@pytest.mark.parametrize("enabled,status", [(False, 400), (True, 404)])
async def test_docker_bridge_host_requires_explicit_gateway_configuration(enabled, status):
    app = create_gateway_app({}, Settings(docker_bridge_enabled=enabled))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://172.17.0.1:8600"
    ) as client:
        response = await client.get("/missing")
    assert response.status_code == status
