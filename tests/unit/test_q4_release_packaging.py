from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_packager() -> ModuleType:
    path = Path(__file__).with_name("package_diffusiongemma_q4_release.py")
    spec = importlib.util.spec_from_file_location("modeldeck_q4_packager", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def release_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    layers = []
    for layer in range(2):
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
        "format_version": 1,
        "state": "complete",
        "base_model_id": "google/diffusiongemma-26B-A4B-it",
        "base_model_revision": "a" * 40,
        "quantization": {
            "method": "gptq",
            "bits": 4,
            "group_size": 32,
            "symmetric": True,
            "desc_act": False,
            "qzero_format": 2,
            "runtime": "gptqmodel-triton-v2",
        },
        "experts": {"layer_count": 2, "layers": layers},
    }
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
            "summary": summary,
            "stability_summary": {"runs": 4, "contract_passes": 4},
            "deterministic_replay": {"exact_text": True},
            "metrics_after": metrics,
        },
        "bf16": {
            "summary": {**summary, "median_wall_seconds": 5.0},
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
    assert result["files_verified"] == 7
    assert result["source_commit"] == "b" * 40
    assert (checkpoint / "README.md").is_file()
    assert (checkpoint / "SHA256SUMS").is_file()


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
