from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_packager() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "package_diffusiongemma_q4_release.py"
    spec = importlib.util.spec_from_file_location("modeldeck_q4_packager", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_fixture(
    tmp_path: Path,
    *,
    self_contained: bool = True,
) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    layers = []
    for layer in range(30):
        path = checkpoint / f"experts-layer-{layer:02d}.safetensors"
        path.write_bytes(f"packed-layer-{layer}".encode())
        layers.append(
            {
                "layer": layer,
                "file": path.name,
                "tensor_bytes": path.stat().st_size,
                "file_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "format": "modeldeck-diffusiongemma-expert-gptq",
        "format_version": 2 if self_contained else 1,
        "state": "complete",
        "base_model_id": "google/diffusiongemma-26B-A4B-it",
        "base_model_revision": "52de6b914ee1749a7d4933202505ddf5b414ec43",
        "quantization": {
            "method": "gptq",
            "bits": 4,
            "group_size": 32,
            "symmetric": True,
            "desc_act": False,
            "qzero_format": 2,
            "runtime": "gptqmodel-triton-v2",
        },
        "experts": {
            "layer_count": 30,
            "experts_per_layer": 128,
            "encoder_decoder_storage": "shared",
            "gate_up_shape": [1408, 2816],
            "down_shape": [2816, 704],
            "state_tensors": ["qweight", "qzeros", "scales", "g_idx"],
            "layers": layers,
        },
    }
    if self_contained:
        non_expert = checkpoint / "non-expert-model-00001-of-00001.safetensors"
        non_expert.write_bytes(b"non-expert-bf16-fixture")
        index = {
            "metadata": {"total_size": non_expert.stat().st_size, "tensor_count": 1},
            "weight_map": {"model.decoder.embed_tokens.weight": non_expert.name},
        }
        write_json(checkpoint / "non-expert-model.safetensors.index.json", index)
        runtime_files = []
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            path = checkpoint / name
            write_json(path, {"fixture": name})
            runtime_files.append(
                {
                    "file": name,
                    "file_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        manifest.update(
            {
                "artifact_type": "self-contained",
                "non_expert_weights": {
                    "source": "checkpoint",
                    "format": "safetensors",
                    "index_file": "non-expert-model.safetensors.index.json",
                    "tensor_count": 1,
                    "tensor_bytes": non_expert.stat().st_size,
                    "shard_count": 1,
                    "shards": [
                        {
                            "file": non_expert.name,
                            "tensor_count": 1,
                            "tensor_bytes": non_expert.stat().st_size,
                            "file_bytes": non_expert.stat().st_size,
                            "sha256": file_sha256(non_expert),
                        }
                    ],
                    "excluded_suffixes": [
                        ".experts.gate_up_proj",
                        ".experts.down_proj",
                    ],
                },
                "runtime_files": runtime_files,
            }
        )
    write_json(checkpoint / "q4-manifest.json", manifest)

    summary = {
        "runs": 8,
        "contract_passes": 8,
        "constraint_passes": 8,
        "constraint_pass_rate": 1.0,
        "median_wall_seconds": 10.0,
    }
    metrics = {
        "runtime": "text-diffusion-gptq-rocm",
        "quantization": "gptq-q4-g32-expert-only",
        "q4_gate_calls": 100,
        "q4_down_calls": 100,
        "q4_bytes": 1000,
        "memory_allocated_bytes": 18 * 1024**3,
        "peak_memory_allocated_bytes": 20 * 1024**3,
        "device": "cuda:0",
        "device_name": "AMD Radeon 8060S Graphics",
        "torch_version": "2.9.1+rocm7.2.1",
        "hip_version": "7.2",
        "transformers_version": "5.13.0",
    }
    evaluation = {
        "format": "modeldeck-diffusiongemma-q4-evaluation",
        "format_version": 1,
        "passed": True,
        "completed_at": "2026-07-13T00:00:00+00:00",
        "checks": {"all_contracts": True, "memory": True},
        "configuration": {
            "model_id": manifest["base_model_id"],
            "revision": manifest["base_model_revision"],
            "prompt_count": 8,
            "max_length": 256,
            "denoising_steps": 48,
            "temperature": 0.8,
        },
        "q4": {
            "worker": {"endpoint": "http://127.0.0.1:8622", "pid": 12001},
            "summary": summary,
            "stability_summary": {"runs": 4, "contract_passes": 4},
            "deterministic_replay": {
                "exact_text": True,
                "reference_job_id": "q4-reference",
                "replay_job_id": "q4-replay",
            },
            "results": [{"job_id": "q4-job", "text": "fixture output"}],
            "metrics_after": {
                **metrics,
                "q4_checkpoint_dir": "/mnt/work/private/q4-checkpoint",
            },
        },
        "bf16": {
            "worker": {"endpoint": "http://127.0.0.1:8621", "pid": 12002},
            "summary": {**summary, "median_wall_seconds": 5.0},
            "results": [{"job_id": "bf16-job", "text": "fixture output"}],
            "metrics_after": {"memory_allocated_bytes": 48 * 1024**3},
        },
        "comparison": {
            "latency_ratio_q4_to_bf16": 2.0,
            "summary": {"mean_token_edit_similarity": 0.65},
        },
    }
    evaluation_path = tmp_path / "evaluation.json"
    write_json(evaluation_path, evaluation)
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Apache License 2.0 fixture\n", encoding="utf-8")
    return checkpoint, evaluation_path, license_path


def test_release_packager_builds_and_verifies_bundle(tmp_path: Path) -> None:
    packager = load_packager()
    checkpoint, evaluation, license_path = release_fixture(tmp_path)

    manifest = packager.build_release(
        checkpoint_dir=checkpoint,
        evaluation_report=evaluation,
        license_file=license_path,
        source_commit="b" * 40,
        created_at="2026-07-13T00:01:00+00:00",
    )
    result = packager.verify_release(checkpoint)

    assert manifest["format"] == "modeldeck-diffusiongemma-q4-release"
    assert result["evaluation_passed"] is True
    assert result["files_verified"] == 40
    assert result["self_contained"] is True
    assert result["source_commit"] == "b" * 40
    assert (checkpoint / "README.md").is_file()
    assert (checkpoint / "SHA256SUMS").is_file()

    model_card = (checkpoint / "README.md").read_text(encoding="utf-8")
    assert "base_model_relation: quantized" in model_card
    assert "  - modeldeck" in model_card
    assert "self-contained ModelDeck Q4/BF16 hybrid" in model_card
    assert "does **not** download or load the upstream checkpoint" in model_card
    assert "https://github.com/ozyjay/ModelDeck" in model_card

    public_report = json.loads((checkpoint / "q4-quality-evaluation.json").read_text(encoding="utf-8"))
    assert public_report["publication"]["sanitized"] is True
    assert "endpoint" not in public_report["q4"]["worker"]
    assert "pid" not in public_report["q4"]["worker"]
    assert "q4_checkpoint_dir" not in public_report["q4"]["metrics_after"]
    assert "job_id" not in public_report["q4"]["results"][0]
    assert "reference_job_id" not in public_report["q4"]["deterministic_replay"]
    assert "replay_job_id" not in public_report["q4"]["deterministic_replay"]

    source_report = json.loads(evaluation.read_text(encoding="utf-8"))
    assert source_report["q4"]["worker"]["pid"] == 12001
    assert source_report["q4"]["results"][0]["job_id"] == "q4-job"


def test_release_verifier_rejects_tampered_shard(tmp_path: Path) -> None:
    packager = load_packager()
    checkpoint, evaluation, license_path = release_fixture(tmp_path)
    packager.build_release(
        checkpoint_dir=checkpoint,
        evaluation_report=evaluation,
        license_file=license_path,
        source_commit="c" * 40,
    )
    (checkpoint / "experts-layer-01.safetensors").write_bytes(b"tampered")

    with pytest.raises(packager.ReleaseError, match="size mismatch|checksum mismatch"):
        packager.verify_release(checkpoint)


def test_release_packager_rejects_failed_constraints(tmp_path: Path) -> None:
    packager = load_packager()
    checkpoint, evaluation, license_path = release_fixture(tmp_path)
    report = json.loads(evaluation.read_text(encoding="utf-8"))
    report["q4"]["summary"]["constraint_passes"] = 7
    report["q4"]["summary"]["constraint_pass_rate"] = 0.875
    write_json(evaluation, report)

    with pytest.raises(packager.ReleaseError, match="constraint"):
        packager.build_release(
            checkpoint_dir=checkpoint,
            evaluation_report=evaluation,
            license_file=license_path,
            source_commit="d" * 40,
        )


def test_public_evaluation_validator_rejects_local_identifiers() -> None:
    packager = load_packager()

    with pytest.raises(packager.ReleaseError, match="private field"):
        packager.validate_public_evaluation(
            {
                "publication": {"sanitized": True},
                "q4": {"worker": {"endpoint": "http://127.0.0.1:8622"}},
            }
        )


def test_release_packager_keeps_v1_delta_compatibility(tmp_path: Path) -> None:
    packager = load_packager()
    checkpoint, evaluation, license_path = release_fixture(
        tmp_path,
        self_contained=False,
    )

    packager.build_release(
        checkpoint_dir=checkpoint,
        evaluation_report=evaluation,
        license_file=license_path,
        source_commit="e" * 40,
    )
    result = packager.verify_release(checkpoint)

    assert result["files_verified"] == 35
    assert result["self_contained"] is False
    assert result["artifact_type"] == "expert-only-quantized-weight-delta"


def test_release_packager_rejects_bf16_expert_in_non_expert_index(
    tmp_path: Path,
) -> None:
    packager = load_packager()
    checkpoint, _, _ = release_fixture(tmp_path)
    index_path = checkpoint / "non-expert-model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"] = {
        "model.decoder.layers.0.experts.gate_up_proj": ("non-expert-model-00001-of-00001.safetensors")
    }
    write_json(index_path, index)

    with pytest.raises(packager.ReleaseError, match="contains BF16 expert tensors"):
        packager.validate_checkpoint(checkpoint)
