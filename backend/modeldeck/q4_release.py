from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_BASE_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
EXPECTED_BASE_REVISION = "52de6b914ee1749a7d4933202505ddf5b414ec43"
EXPECTED_QUANTISATION = {
    "method": "gptq",
    "bits": 4,
    "group_size": 32,
    "symmetric": True,
    "desc_act": False,
    "qzero_format": 2,
}


class Q4ReleaseError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q4ReleaseError(f"Invalid or missing release metadata: {path.name}") from error
    if not isinstance(payload, dict):
        raise Q4ReleaseError(f"Release metadata must be an object: {path.name}")
    return payload


def _safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise Q4ReleaseError("Release file path is missing")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise Q4ReleaseError(f"Unsafe release file path: {value!r}")
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Q4ReleaseError(f"Unsafe release file path: {value!r}") from error
    return candidate


def inspect_modeldeck_q4_release(snapshot: Path) -> dict[str, str] | None:
    release_path = snapshot / "release-manifest.json"
    q4_path = snapshot / "q4-manifest.json"
    if not release_path.exists() and not q4_path.exists():
        return None
    release = _read_json(release_path)
    q4 = _read_json(q4_path)
    if release.get("format") != "modeldeck-diffusiongemma-q4-release":
        raise Q4ReleaseError("Unsupported ModelDeck Q4 release format")
    if release.get("format_version") != 2:
        raise Q4ReleaseError("Only self-contained ModelDeck Q4 release format 2 is supported")
    if release.get("artifact_type") != "self-contained-q4-bf16-hybrid":
        raise Q4ReleaseError("ModelDeck Q4 release is not self-contained")
    base = release.get("base_model")
    if not isinstance(base, dict):
        raise Q4ReleaseError("ModelDeck Q4 release has no base-model identity")
    if base.get("id") != EXPECTED_BASE_MODEL_ID or base.get("revision") != EXPECTED_BASE_REVISION:
        raise Q4ReleaseError("ModelDeck Q4 release does not reference the supported base model")
    if q4.get("format") != "modeldeck-diffusiongemma-expert-gptq":
        raise Q4ReleaseError("Unsupported Q4 checkpoint manifest format")
    if q4.get("format_version") != 2 or q4.get("state") != "complete":
        raise Q4ReleaseError("Q4 checkpoint is not a complete self-contained format 2 release")
    if q4.get("artifact_type") != "self-contained":
        raise Q4ReleaseError("Q4 checkpoint is not self-contained")
    if q4.get("base_model_id") != base["id"] or q4.get("base_model_revision") != base["revision"]:
        raise Q4ReleaseError("Release and checkpoint base-model identities do not match")
    quantisation = q4.get("quantization")
    if not isinstance(quantisation, dict) or any(
        quantisation.get(key) != value for key, value in EXPECTED_QUANTISATION.items()
    ):
        raise Q4ReleaseError("Q4 checkpoint quantisation settings are unsupported")
    if release.get("quantization") != quantisation:
        raise Q4ReleaseError("Release and checkpoint quantisation settings do not match")
    experts = q4.get("experts")
    layers = experts.get("layers") if isinstance(experts, dict) else None
    if not isinstance(experts, dict) or experts.get("layer_count") != 30:
        raise Q4ReleaseError("Q4 checkpoint expert topology is unsupported")
    if not isinstance(layers, list) or len(layers) != 30:
        raise Q4ReleaseError("Q4 checkpoint does not describe every expert layer")
    if [entry.get("layer") for entry in layers if isinstance(entry, dict)] != list(range(30)):
        raise Q4ReleaseError("Q4 checkpoint layer metadata is not contiguous")
    for entry in layers:
        layer_path = _safe_path(snapshot, entry.get("file"))
        if not layer_path.is_file():
            raise Q4ReleaseError(f"Q4 expert shard is missing: {layer_path.name}")
        if isinstance(entry.get("file_bytes"), int) and layer_path.stat().st_size != entry["file_bytes"]:
            raise Q4ReleaseError(f"Q4 expert shard size does not match: {layer_path.name}")
    non_experts = q4.get("non_expert_weights")
    if not isinstance(non_experts, dict) or non_experts.get("source") != "checkpoint":
        raise Q4ReleaseError("Q4 checkpoint has no self-contained non-expert weights")
    if non_experts.get("index_file") != "non-expert-model.safetensors.index.json":
        raise Q4ReleaseError("Q4 checkpoint has an unsupported non-expert index")
    shards = non_experts.get("shards")
    if not isinstance(shards, list) or len(shards) != non_experts.get("shard_count"):
        raise Q4ReleaseError("Q4 checkpoint non-expert shard metadata is incomplete")
    shard_names: set[str] = set()
    for entry in shards:
        if not isinstance(entry, dict):
            raise Q4ReleaseError("Q4 checkpoint contains invalid non-expert shard metadata")
        shard_path = _safe_path(snapshot, entry.get("file"))
        if not shard_path.is_file() or shard_path.stat().st_size != entry.get("file_bytes"):
            raise Q4ReleaseError(f"Q4 non-expert shard is missing or incomplete: {shard_path.name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise Q4ReleaseError(f"Q4 non-expert shard has no checksum: {shard_path.name}")
        shard_names.add(str(entry["file"]))
    index = _read_json(snapshot / "non-expert-model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map.values()) != shard_names:
        raise Q4ReleaseError("Q4 non-expert index does not match its shard metadata")
    for required in (
        "SHA256SUMS",
        "config.json",
        "non-expert-model.safetensors.index.json",
        "q4-quality-evaluation.json",
    ):
        if not (snapshot / required).is_file():
            raise Q4ReleaseError(f"ModelDeck Q4 release file is missing: {required}")
    return {
        "base_model_id": str(base["id"]),
        "base_model_revision": str(base["revision"]),
        "release_name": str(release.get("release_name") or "ModelDeck DiffusionGemma Q4"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_modeldeck_q4_release(snapshot: Path) -> dict[str, Any]:
    metadata = inspect_modeldeck_q4_release(snapshot)
    if metadata is None:
        raise Q4ReleaseError("Snapshot is not a ModelDeck Q4 release")
    release_path = snapshot / "release-manifest.json"
    release = _read_json(release_path)
    files = release.get("files")
    if not isinstance(files, list) or not files:
        raise Q4ReleaseError("ModelDeck Q4 release contains no file inventory")
    expected: dict[str, str] = {}
    payload_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise Q4ReleaseError("ModelDeck Q4 release contains invalid file metadata")
        name = entry.get("path")
        path = _safe_path(snapshot, name)
        if not path.is_file():
            raise Q4ReleaseError(f"ModelDeck Q4 release file is missing: {name}")
        size = path.stat().st_size
        if size != entry.get("size_bytes"):
            raise Q4ReleaseError(f"ModelDeck Q4 release file size does not match: {name}")
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            raise Q4ReleaseError(f"ModelDeck Q4 release checksum does not match: {name}")
        expected[str(name)] = digest
        payload_bytes += size
    q4 = _read_json(snapshot / "q4-manifest.json")
    required_inventory = {
        "config.json",
        "non-expert-model.safetensors.index.json",
        "q4-manifest.json",
        "q4-quality-evaluation.json",
        *(entry["file"] for entry in q4["experts"]["layers"]),
        *(entry["file"] for entry in q4["non_expert_weights"]["shards"]),
    }
    if not required_inventory.issubset(expected):
        raise Q4ReleaseError("ModelDeck Q4 release inventory omits required checkpoint files")
    expected[release_path.name] = _sha256(release_path)
    checksums: dict[str, str] = {}
    for line in (snapshot / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise Q4ReleaseError("ModelDeck Q4 SHA256SUMS file is invalid")
        digest, name = match.groups()
        _safe_path(snapshot, name)
        if name in checksums:
            raise Q4ReleaseError(f"Duplicate ModelDeck Q4 checksum entry: {name}")
        checksums[name] = digest
    if checksums != expected:
        raise Q4ReleaseError("ModelDeck Q4 SHA256SUMS does not match the release inventory")
    evaluation = _read_json(snapshot / "q4-quality-evaluation.json")
    release_evaluation = release.get("evaluation")
    checks = release_evaluation.get("checks") if isinstance(release_evaluation, dict) else None
    if (
        evaluation.get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise Q4ReleaseError("ModelDeck Q4 release quality evidence did not pass")
    return {**metadata, "files_verified": len(files), "payload_bytes": payload_bytes}
