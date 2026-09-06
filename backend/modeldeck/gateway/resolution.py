"""Configured route identity is independent of currently usable Workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modeldeck.domain import WorkerDefinition
from modeldeck.profiles import ModelProfile


class RouteCandidates(list[ModelProfile]):
    def __init__(self, profiles: list[ModelProfile], configured_ids: tuple[str, ...]):
        super().__init__(profiles)
        self.configured_ids = configured_ids


@dataclass(frozen=True)
class ResolvedRoute:
    public_name: str
    configured_worker_ids: tuple[str, ...]
    candidates: RouteCandidates
    configured_workers: tuple[dict[str, Any], ...]

    @classmethod
    def resolve(cls, name: str, worker_ids: list[str], records: list[dict]) -> ResolvedRoute:
        by_id = {record.get("id", record["definition"].get("id")): record for record in records}
        configured = []
        candidates = []
        for worker_id in worker_ids:
            record = by_id.get(worker_id)
            document = record["definition"] if record else {}
            status = "missing" if record is None else "configured"
            profile = None
            if record:
                try:
                    definition = WorkerDefinition.model_validate(document)
                    if definition.id != worker_id:
                        raise ValueError("Persisted Worker ID differs from its row identity")
                    profile = definition.to_profile()
                    if record.get("archived_at") or definition.archived:
                        status = "archived"
                    else:
                        candidates.append(profile)
                except ValueError:
                    status = "invalid_or_retired"
            configured.append(
                {
                    "worker_id": worker_id,
                    "status": status,
                    "requested": execution_identity(document),
                }
            )
        ids = tuple(worker_ids)
        return cls(name, ids, RouteCandidates(candidates, ids), tuple(configured))

    def metadata(self) -> dict[str, Any]:
        return {
            "configured_worker_ids": list(self.configured_worker_ids),
            "configured_workers": list(self.configured_workers),
            "candidate_worker_ids": [profile.id for profile in self.candidates],
        }


def execution_identity(document: dict) -> dict[str, Any]:
    settings = document.get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    capabilities = document.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    return {
        "model_id": document.get("model_id"),
        "model_revision": document.get("revision"),
        "artifact_model_id": document.get("artifact_model_id"),
        "artifact_revision": document.get("artifact_revision"),
        "artifact_id": settings.get("artifact_id"),
        "artifact_format": settings.get("artifact_format"),
        "artifact_sha256": settings.get("artifact_sha256"),
        "quantisation": settings.get("quantisation"),
        "runtime": document.get("runtime") or document.get("preferred_runtime"),
        "runtime_template_id": document.get("runtime_template_id"),
        "runtime_template_version": document.get("runtime_template_version"),
        "backend": settings.get("backend"),
        "device": settings.get("device"),
        "precision": document.get("dtype"),
        "context_length": settings.get("context_length"),
        "kv_cache": {
            key: value
            for key, value in {**capabilities, **settings}.items()
            if "cache" in key and "path" not in key and "root" not in key
        },
    }
