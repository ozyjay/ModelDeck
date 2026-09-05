"""Narrow TCP forwarder from Docker's bridge to the authoritative gateway."""

from __future__ import annotations

import asyncio
import logging
from ipaddress import ip_address

from modeldeck.config import Settings

LOGGER = logging.getLogger("modeldeck.gateway.docker_bridge")
DOCKER_BRIDGE_HOST = "172.17.0.1"


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass


async def _forward_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
    except OSError:
        LOGGER.warning("Authoritative loopback gateway is unavailable")
        client_writer.close()
        await client_writer.wait_closed()
        return

    try:
        await asyncio.gather(
            _copy_stream(client_reader, upstream_writer),
            _copy_stream(upstream_reader, client_writer),
        )
    finally:
        upstream_writer.close()
        client_writer.close()
        await asyncio.gather(
            upstream_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )


async def start_forwarder(
    *,
    bind_host: str,
    bind_port: int,
    target_host: str,
    target_port: int,
) -> asyncio.AbstractServer:
    if bind_host != DOCKER_BRIDGE_HOST:
        raise ValueError(f"Docker bridge forwarder must bind to {DOCKER_BRIDGE_HOST}")
    try:
        target_address = ip_address(target_host)
    except ValueError as error:
        raise ValueError("Docker bridge forwarder target must be a loopback IP address") from error
    if not target_address.is_loopback:
        raise ValueError("Docker bridge forwarder target must be loopback")

    return await asyncio.start_server(
        lambda reader, writer: _forward_connection(
            reader,
            writer,
            target_host=target_host,
            target_port=target_port,
        ),
        bind_host,
        bind_port,
    )


async def _serve() -> None:
    settings = Settings.from_env()
    if not settings.docker_bridge_enabled:
        raise RuntimeError("MODELDECK_ENABLE_DOCKER_BRIDGE must be enabled for the bridge forwarder")

    server = await start_forwarder(
        bind_host=DOCKER_BRIDGE_HOST,
        bind_port=settings.gateway_port,
        target_host=settings.gateway_host,
        target_port=settings.gateway_port,
    )
    LOGGER.info(
        "Docker bridge forwarder listening on %s:%s and targeting %s:%s",
        DOCKER_BRIDGE_HOST,
        settings.gateway_port,
        settings.gateway_host,
        settings.gateway_port,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
