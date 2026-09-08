"""Read-only Open Day discovery checks; never initialise stores or invoke inference."""

from __future__ import annotations

import argparse
import asyncio
import json
from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx

from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS

DEMO_ROUTES = {
    "TokenTrail": {
        "qwen-0-5b": "native-ar-trace-v1",
        "qwen-1-5b": "native-ar-trace-v1",
        "qwen-3b": "native-ar-trace-v1",
    },
    "TextDiffusionDemo": {"text-diffusion-lab-q4": "text-diffusion-v1"},
    "SceneChat": {"scenechat-vision": "scene-analysis-v1"},
    "SpeechShift": {
        "speechshift-stt": "speech-recognition-v1",
        "speechshift-en-fr": "translation-en-fr-v1",
        "speechshift-en-de": "translation-en-de-v1",
        "speechshift-voice": "speech-synthesis-v1",
    },
}


def local_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        local = parsed.hostname == "localhost" or ip_address(parsed.hostname or "").is_loopback
        port = parsed.port
    except ValueError:
        local, port = False, None
    if (
        parsed.scheme != "http"
        or not local
        or not port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError("Use an HTTP loopback base URL with an explicit port and no path.")
    return value.rstrip("/")


async def fetch(client: httpx.AsyncClient, url: str, field: str, kind: type) -> dict:
    try:
        response = await client.get(url)
        if not response.is_success:
            result = {"ok": False, "url": url, "status": response.status_code, "error": "http_error"}
            try:
                payload = response.json()
                detail = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(detail, dict):
                    result["detail"] = {
                        key: detail[key] for key in ("code", "component", "operation") if key in detail
                    }
            except ValueError:
                pass  # The HTTP failure remains explicit even without a JSON error envelope.
            return result
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get(field), kind):
            raise ValueError("Unexpected discovery envelope")
        if kind is list and any(not isinstance(item, dict) for item in body[field]):
            raise ValueError("Unexpected discovery record")
        return {"ok": True, "url": url, "data": body}
    except httpx.HTTPError as error:
        return {"ok": False, "url": url, "error": type(error).__name__}
    except ValueError:
        return {"ok": False, "url": url, "error": "invalid_response"}


async def preflight(
    client: httpx.AsyncClient,
    management_url: str,
    gateway_url: str,
    requirements: dict[str, dict[str, str]] | None = None,
) -> dict:
    management_url, gateway_url = local_url(management_url), local_url(gateway_url)
    endpoints = {
        "management": (management_url + "/api/health", "status", str),
        "gateway": (gateway_url + "/v1/health", "status", str),
        "live": (management_url + "/api/live", "active_profiles", list),
        "routes": (gateway_url + "/v1/routes", "routes", list),
        "thermal": (gateway_url + "/v1/thermal", "state", str),
    }
    responses = dict(
        zip(endpoints, await asyncio.gather(*(fetch(client, *v) for v in endpoints.values())), strict=True)
    )
    live = responses["live"].get("data", {})
    discovery = responses["routes"].get("data", {})
    routes = {item.get("public_name"): item for item in discovery.get("routes", [])}
    checks = []
    for demo, names in (requirements if requirements is not None else DEMO_ROUTES).items():
        for name, contract in names.items():
            route = routes.get(name)
            diagnostic = []
            present = route is not None if responses["routes"]["ok"] else None
            protocol = route.get("protocol_contract") if route else None
            compatible = protocol == contract if protocol else None
            ready = route.get("ready") if route and isinstance(route.get("ready"), bool) else None
            if present is None:
                diagnostic.append(
                    "Gateway discovery failed; route presence is unknown. Check the service URL."
                )
            elif not present:
                diagnostic.append(
                    f"Publish '{name}' with {contract} in the intended store's Routing Profile. "
                    "Review existing Workers and qualification first; "
                    "preserve other active profiles and backups."
                )
            elif compatible is None:
                diagnostic.append(
                    "Protocol is unreported; use a gateway build with typed /v1/routes discovery."
                )
            elif not compatible:
                diagnostic.append(f"Published protocol is {protocol}; this consumer requires {contract}.")
            if present and ready is not True:
                diagnostic.append(
                    "Inspect configured Worker health in ModelDeck; no Worker was started by this check."
                )
            resolution = discovery.get("resolution", {})
            checks.append(
                {
                    "demo": demo,
                    "public_name": name,
                    "present": present,
                    "required_protocol": contract,
                    "reported_protocol": protocol,
                    "protocol_compatible": compatible,
                    "required_surfaces": list(PROTOCOL_CONTRACTS[contract].surfaces),
                    "worker_ready": ready,
                    "resolution": resolution.get(name) if isinstance(resolution, dict) else None,
                    "diagnostics": diagnostic,
                }
            )
    thermal = responses["thermal"].get("data", {})
    temperature = thermal.get("temperature_c")
    thermal_check = "unknown"
    if thermal.get("state") == "critical":
        thermal_check = "terminate"
    elif thermal.get("state") in {"hot", "very_hot"}:
        thermal_check = "pause_or_reduce"
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        if temperature >= 90 or thermal_check == "terminate":
            thermal_check = "terminate"
        elif temperature > 85 or thermal_check == "pause_or_reduce":
            thermal_check = "pause_or_reduce"
        elif thermal.get("enabled") is True and thermal.get("state") in {"normal", "warm"}:
            thermal_check = "snapshot_permits_rehearsal_review"
    services_ok = all(
        responses[key]["ok"] and responses[key]["data"]["status"] == "ok" for key in ("management", "gateway")
    )
    return {
        "read_only": True,
        "hardware_qualified": False,
        "services": {key: responses[key] for key in ("management", "gateway")},
        "active_profiles": live.get("active_profiles"),
        "discovery": {
            key: {k: v for k, v in responses[key].items() if k != "data"} for key in ("live", "routes")
        },
        "routes": checks,
        "thermal_admission": {
            "assessment": thermal_check,
            "observation": responses["thermal"],
            "note": (
                "Snapshot only; no capacity reserved. "
                "Inspect sensors, memory and stricter host policy before inference."
            ),
        },
        "checks_passed": (
            services_ok
            and responses["live"]["ok"]
            and all(c["present"] and c["protocol_compatible"] and c["worker_ready"] for c in checks)
            and thermal_check == "snapshot_permits_rehearsal_review"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--management-url", type=local_url, default="http://127.0.0.1:3600")
    parser.add_argument("--gateway-url", type=local_url, default="http://127.0.0.1:8600")
    parser.add_argument("--demo", choices=DEMO_ROUTES, action="append")
    parser.add_argument("--diffusion-model", default="text-diffusion-lab-q4")
    args = parser.parse_args()
    requirements = {name: dict(DEMO_ROUTES[name]) for name in (args.demo or DEMO_ROUTES)}
    if "TextDiffusionDemo" in requirements:
        requirements["TextDiffusionDemo"] = {args.diffusion_model: "text-diffusion-v1"}

    async def run() -> dict:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False, follow_redirects=False) as client:
            return await preflight(client, args.management_url, args.gateway_url, requirements)

    report = asyncio.run(run())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
