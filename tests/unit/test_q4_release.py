from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.q4_release import Q4ReleaseError, verify_modeldeck_q4_release


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def make_q4_release(snapshot: Path) -> None:
    snapshot.mkdir(parents=True)
    quantisation = {
        "method": "gptq",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "desc_act": False,
        "qzero_format": 2,
    }
    _write_json(
        snapshot / "config.json",
        {
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "model_type": "diffusion_gemma",
        },
    )
    layers = []
    for layer in range(30):
        name = f"experts-layer-{layer:02d}.safetensors"
        (snapshot / name).write_bytes(f"expert-{layer}".encode())
        layers.append({"layer": layer, "file": name, "file_bytes": (snapshot / name).stat().st_size})
    non_expert = snapshot / "non-expert-model-00001-of-00001.safetensors"
    non_expert.write_bytes(b"non-expert")
    _write_json(
        snapshot / "non-expert-model.safetensors.index.json",
        {"weight_map": {"x": non_expert.name}},
    )
    _write_json(snapshot / "q4-quality-evaluation.json", {"passed": True})
    _write_json(
        snapshot / "q4-manifest.json",
        {
            "format": "modeldeck-diffusiongemma-expert-gptq",
            "format_version": 2,
            "state": "complete",
            "artifact_type": "self-contained",
            "base_model_id": "google/diffusiongemma-26B-A4B-it",
            "base_model_revision": "52de6b914ee1749a7d4933202505ddf5b414ec43",
            "quantization": quantisation,
            "experts": {"layer_count": 30, "layers": layers},
            "non_expert_weights": {
                "source": "checkpoint",
                "index_file": "non-expert-model.safetensors.index.json",
                "shard_count": 1,
                "shards": [
                    {
                        "file": non_expert.name,
                        "file_bytes": non_expert.stat().st_size,
                        "sha256": _digest(non_expert),
                    }
                ],
            },
        },
    )
    payload_paths = [
        path for path in snapshot.iterdir() if path.name not in {"release-manifest.json", "SHA256SUMS"}
    ]
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in sorted(payload_paths)
    ]
    _write_json(
        snapshot / "release-manifest.json",
        {
            "format": "modeldeck-diffusiongemma-q4-release",
            "format_version": 2,
            "release_name": "test-q4",
            "artifact_type": "self-contained-q4-bf16-hybrid",
            "base_model": {
                "id": "google/diffusiongemma-26B-A4B-it",
                "revision": "52de6b914ee1749a7d4933202505ddf5b414ec43",
            },
            "quantization": quantisation,
            "evaluation": {"checks": {"quality": True}},
            "files": files,
        },
    )
    expected = {entry["path"]: entry["sha256"] for entry in files}
    expected["release-manifest.json"] = _digest(snapshot / "release-manifest.json")
    (snapshot / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(expected.items())),
        encoding="utf-8",
    )


def test_discovers_and_verifies_modeldeck_q4_hf_release(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--ozyjay--q4" / "snapshots" / "commit"
    make_q4_release(snapshot)

    model = discover_huggingface_models([tmp_path])[0]
    verification = verify_modeldeck_q4_release(snapshot)

    assert model["model_id"] == "ozyjay/q4"
    assert model["configuration_support"] == "diffusiongemma-modeldeck-q4"
    assert model["base_model_id"] == "google/diffusiongemma-26B-A4B-it"
    assert verification["files_verified"] == 35


def test_q4_release_checksum_verification_rejects_tampering(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    make_q4_release(snapshot)
    (snapshot / "experts-layer-00.safetensors").write_bytes(b"tampered")

    with pytest.raises(Q4ReleaseError, match="size does not match|checksum does not match"):
        verify_modeldeck_q4_release(snapshot)


def test_q4_release_accepts_hugging_face_repository_blob_links(tmp_path: Path) -> None:
    repository = tmp_path / "models--ozyjay--q4"
    snapshot = repository / "snapshots" / "commit"
    make_q4_release(snapshot)
    shard = snapshot / "experts-layer-00.safetensors"
    blob = repository / "blobs" / "expert-00"
    blob.parent.mkdir()
    shard.replace(blob)
    shard.symlink_to(Path("../../blobs") / blob.name)

    verification = verify_modeldeck_q4_release(snapshot)

    assert verification["files_verified"] == 35


def test_q4_release_rejects_blob_link_outside_hugging_face_repository(tmp_path: Path) -> None:
    repository = tmp_path / "models--ozyjay--q4"
    snapshot = repository / "snapshots" / "commit"
    make_q4_release(snapshot)
    shard = snapshot / "experts-layer-00.safetensors"
    outside = tmp_path / "outside.safetensors"
    shard.replace(outside)
    shard.symlink_to(outside)

    with pytest.raises(Q4ReleaseError, match="Unsafe release file path"):
        verify_modeldeck_q4_release(snapshot)
