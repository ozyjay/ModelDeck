from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

MAXIMUM_REPORT_BYTES = 16 * 1024 * 1024
MAXIMUM_HISTORY_POINTS = 500


def read_benchmark_history(directory: Path) -> dict[str, Any]:
    """Read bounded, privacy-safe throughput summaries from local benchmark reports."""

    points: list[dict[str, Any]] = []
    reports_scanned = 0
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                if path.is_symlink() or path.stat().st_size > MAXIMUM_REPORT_BYTES:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            extracted = _extract_points(payload)
            if extracted:
                reports_scanned += 1
                points.extend(extracted)
    points.sort(key=lambda point: (point["observed_at"], point["series_key"]))
    return {
        "points": points[-MAXIMUM_HISTORY_POINTS:],
        "reports_scanned": reports_scanned,
        "measurement": "median benchmark throughput",
    }


def _extract_points(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    report_format = report.get("format")
    if report_format == "modeldeck-benchmark":
        return _standard_points(report)
    if report_format == "modeldeck-scenechat-visual-token-benchmark":
        return _scenechat_points(report)
    return []


def _standard_points(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    configuration = _mapping(report.get("configuration"))
    preset = str(configuration.get("preset") or "unknown")
    observed_at = _timestamp(report.get("completed_at") or report.get("started_at"))
    if observed_at is None:
        return []
    points = []
    for raw_result in report.get("results", []):
        result = _mapping(raw_result)
        if result.get("status") != "success":
            continue
        summary = _mapping(result.get("summary"))
        throughput = _mapping(summary.get("throughput_tokens_per_second"))
        value = _positive_number(throughput.get("median"))
        model_id = _text(result.get("model_id"))
        revision = _text(result.get("model_revision"))
        runtime = _text(result.get("runtime"))
        family = _text(result.get("generation_family"))
        if value is None or not all((model_id, revision, runtime, family)):
            continue
        token_setting = configuration.get(
            "llama_tokens"
            if "llama" in runtime
            else {
                "autoregressive": "autoregressive_tokens",
                "text-diffusion": "diffusion_tokens",
                "vision-language": "vision_tokens",
            }.get(family, "")
        )
        workload = f"Standard · {preset} · {token_setting or 'bounded'} output tokens"
        points.append(
            _point(
                observed_at=observed_at,
                model_id=model_id,
                revision=revision,
                runtime=runtime,
                dtype=_text(result.get("dtype")) or "unknown",
                generation_family=family,
                worker_id=_text(result.get("worker_id") or result.get("profile_id")),
                worker_name=_text(result.get("worker_name")),
                throughput=value,
                workload=workload,
                workload_identity={"kind": "standard", "preset": preset, "tokens": token_setting},
                fingerprint=_text(result.get("fingerprint")),
                sample_count=_integer(summary.get("successful_requests")),
            )
        )
    return points


def _scenechat_points(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed_at = _timestamp(report.get("created_at"))
    model_id = _text(report.get("model_id"))
    revision = _text(report.get("revision"))
    if observed_at is None or not model_id or not revision or report.get("thermal_abort") is True:
        return []
    points = []
    for raw_arm in report.get("arms", []):
        arm = _mapping(raw_arm)
        if arm.get("benchmark_status") not in {None, "accepted", "completed", "success"}:
            continue
        configuration = _mapping(arm.get("configuration"))
        throughput = _mapping(arm.get("tokens_per_second"))
        value = _positive_number(throughput.get("p50"))
        runtime = _text(configuration.get("runtime"))
        budget = _integer(arm.get("visual_token_budget") or configuration.get("visual_token_budget"))
        if value is None or not runtime or budget is None:
            continue
        workload = f"SceneChat · {budget} visual tokens"
        points.append(
            _point(
                observed_at=observed_at,
                model_id=model_id,
                revision=revision,
                runtime=runtime,
                dtype=_text(configuration.get("dtype")) or "unknown",
                generation_family="vision-language",
                worker_id=_text(configuration.get("worker_id")),
                worker_name=None,
                throughput=value,
                workload=workload,
                workload_identity={
                    "kind": "scenechat",
                    "visual_token_budget": budget,
                    "maximum_new_tokens": configuration.get("maximum_new_tokens"),
                    "prompt_sha256": report.get("prompt_sha256"),
                    "schema_sha256": report.get("schema_sha256"),
                },
                fingerprint=None,
                sample_count=_integer(arm.get("valid_responses")),
            )
        )
    return points


def _point(
    *,
    observed_at: str,
    model_id: str,
    revision: str,
    runtime: str,
    dtype: str,
    generation_family: str,
    worker_id: str | None,
    worker_name: str | None,
    throughput: float,
    workload: str,
    workload_identity: Mapping[str, Any],
    fingerprint: str | None,
    sample_count: int | None,
) -> dict[str, Any]:
    identity = {
        "model_id": model_id,
        "revision": revision,
        "runtime": runtime,
        "dtype": dtype,
        "generation_family": generation_family,
        "workload": dict(workload_identity),
    }
    series_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return {
        "series_key": series_key,
        "observed_at": observed_at,
        "model_id": model_id,
        "model_revision": revision,
        "runtime": runtime,
        "dtype": dtype,
        "generation_family": generation_family,
        "worker_id": worker_id,
        "worker_name": worker_name,
        "tokens_per_second": throughput,
        "workload": workload,
        "configuration_fingerprint": fingerprint,
        "sample_count": sample_count,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
