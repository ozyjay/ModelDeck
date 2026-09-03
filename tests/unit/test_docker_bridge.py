from __future__ import annotations

import asyncio

import pytest
from modeldeck.gateway import docker_bridge


@pytest.mark.asyncio
async def test_bridge_connection_forwards_bytes_without_gateway_state() -> None:
    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        assert await reader.readexactly(4) == b"ping"
        writer.write(b"pong")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    upstream = await asyncio.start_server(respond, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]
    bridge = await asyncio.start_server(
        lambda reader, writer: docker_bridge._forward_connection(
            reader,
            writer,
            target_host="127.0.0.1",
            target_port=upstream_port,
        ),
        "127.0.0.1",
        0,
    )
    bridge_port = bridge.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", bridge_port)
        writer.write(b"ping")
        await writer.drain()
        assert await reader.readexactly(4) == b"pong"
        writer.close()
        await writer.wait_closed()
    finally:
        bridge.close()
        upstream.close()
        await bridge.wait_closed()
        await upstream.wait_closed()


@pytest.mark.asyncio
async def test_forwarder_rejects_non_bridge_bind_and_non_loopback_target() -> None:
    with pytest.raises(ValueError, match="must bind"):
        await docker_bridge.start_forwarder(
            bind_host="0.0.0.0",
            bind_port=8600,
            target_host="127.0.0.1",
            target_port=8600,
        )

    with pytest.raises(ValueError, match="must be loopback"):
        await docker_bridge.start_forwarder(
            bind_host=docker_bridge.DOCKER_BRIDGE_HOST,
            bind_port=8600,
            target_host="192.168.1.10",
            target_port=8600,
        )
