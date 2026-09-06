"""Local-browser boundary checks shared by ModelDeck ASGI applications."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_trusted_browser_origin(origin: str | None) -> bool:
    """Accept only explicit loopback browser origins.

    Native local clients deliberately omit ``Origin`` and are handled by the
    caller. An Origin header is browser provenance, so accepting arbitrary
    origins would permit a remote website to mutate the local control plane.
    """

    if not origin:
        return False
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError:
        return False
    if any(char.isspace() for char in origin) or "\\" in origin or parsed.fragment:
        return False
    if port is not None and not 1 <= port <= 65535:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query:
        return False
    if parsed.hostname.casefold() == "localhost":
        return True
    try:
        return ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def rejects_browser_mutation(connection: HTTPConnection) -> bool:
    """Return whether a browser-originated non-safe request must be rejected."""

    origin = connection.headers.get("origin")
    origins = connection.headers.getlist("origin")
    return len(origins) > 1 or (origin is not None and not is_trusted_browser_origin(origin))


class LocalBrowserBoundary:
    """Check Host on every request, including native clients and WebSockets.

    Never resolve names or trust forwarded headers: a remote name resolving to
    loopback is precisely the DNS rebinding case this boundary rejects.
    """

    def __init__(self, app: ASGIApp, *, docker_bridge: bool = False) -> None:
        self.app = app
        self.docker_bridge = docker_bridge

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        connection = HTTPConnection(scope)
        hosts = connection.headers.getlist("host")
        trusted = (
            len(hosts) == 1
            and not any(char in hosts[0] for char in "/?#@\\")
            and is_trusted_browser_origin("http://" + hosts[0])
        )
        if self.docker_bridge and len(hosts) == 1:
            try:
                parsed = urlsplit("http://" + hosts[0])
                trusted |= (
                    parsed.hostname == "172.17.0.1"
                    and (parsed.port is None or 1 <= parsed.port <= 65535)
                    and not (
                        parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                    )
                )
            except ValueError:
                pass
        mutation = scope["type"] == "websocket" or scope["method"] not in SAFE_HTTP_METHODS
        rejected_origin = mutation and rejects_browser_mutation(connection)
        if not trusted or rejected_origin:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                response = JSONResponse(
                    {"detail": "Untrusted browser origin" if trusted else "Untrusted Host header"},
                    status_code=403 if trusted else 400,
                )
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
