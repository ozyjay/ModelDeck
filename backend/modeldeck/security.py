"""Local-browser boundary checks shared by ModelDeck ASGI applications."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection

SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_trusted_browser_origin(origin: str | None) -> bool:
    """Accept only explicit loopback browser origins.

    Native local clients deliberately omit ``Origin`` and are handled by the
    caller. An Origin header is browser provenance, so accepting arbitrary
    origins would permit a remote website to mutate the local control plane.
    """

    if not origin:
        return False
    parsed = urlsplit(origin)
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
    return bool(origin) and not is_trusted_browser_origin(origin)
