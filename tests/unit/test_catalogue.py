from __future__ import annotations

import json
from pathlib import Path

from modeldeck.catalogue import discover_huggingface_models, resolve_cache_paths


def test_cache_path_precedence(tmp_path: Path) -> None:
    paths = resolve_cache_paths(
        {"HOME": str(tmp_path), "HF_HOME": str(tmp_path / "hf-home"), "HF_HUB_CACHE": str(tmp_path / "exact")}
    )
    assert paths[:3] == [tmp_path / "exact", tmp_path / "hf-home/hub", tmp_path / ".cache/huggingface/hub"]
    assert Path("/mnt/work/models/huggingface/hub") in paths


def test_discovers_complete_cache_without_claiming_compatibility(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Qwen--Demo" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["DemoForCausalLM"]}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")

    models = discover_huggingface_models([tmp_path])

    assert models[0]["model_id"] == "Qwen/Demo"
    assert models[0]["download_state"] == "installed-untested"
    assert models[0]["generation_family_hint"] == "autoregressive"
    assert models[0]["configuration_support"] == "autoregressive-transformers"
    assert models[0]["runnable"] is False


def test_marks_incomplete_snapshot_partial(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Org--Partial" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.incomplete").write_bytes(b"partial")
    assert discover_huggingface_models([tmp_path])[0]["download_state"] == "partial"


def test_model_payload_without_transformers_config_is_complete_but_unsupported(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "models--kyutai--moshiko-pytorch-bf16" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "tokenizer.model").write_bytes(b"tokenizer")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["download_state"] == "installed-untested"
    assert model["configuration_support"] is None
    assert "Moshiko snapshot is incomplete" in model["configuration_support_reason"]


def test_recognises_complete_moshiko_manifest(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--kyutai--moshiko-pytorch-bf16" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "tokenizer_spm_32k_3.model",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
    ):
        (snapshot / filename).write_bytes(b"local")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["generation_family_hint"] == "speech-conversation"
    assert model["configuration_support"] == "moshiko-speech"


def test_transformers_config_without_model_payload_remains_partial(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Org--ConfigOnly" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["DemoForCausalLM"]}), encoding="utf-8"
    )

    model = discover_huggingface_models([tmp_path])[0]

    assert model["download_state"] == "partial"


def test_ignores_metadata_only_repository_without_a_snapshot(tmp_path: Path) -> None:
    reference = tmp_path / "models--Org--Stale" / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text("missing-snapshot", encoding="utf-8")

    assert discover_huggingface_models([tmp_path]) == []


def test_identifies_gemma4_as_vision_language_without_claiming_readiness(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--google--gemma-4-E2B-it" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {"model_type": "gemma3_text"},
                "vision_config": {"model_type": "siglip_vision_model"},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["generation_family_hint"] == "vision-language"
    assert model["configuration_support"] == "scenechat-gemma4"
    assert model["runnable"] is False


def test_identifies_gemma4_unified_as_scenechat_compatible(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
                "text_config": {},
                "vision_config": {},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["generation_family_hint"] == "vision-language"
    assert model["configuration_support"] == "scenechat-gemma4"


def test_gpt_oss_source_points_to_companion_and_complete_gguf_is_configurable(tmp_path: Path) -> None:
    source = tmp_path / "models--openai--gpt-oss-120b" / "snapshots" / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"source")
    companion = tmp_path / "models--ggml-org--gpt-oss-120b-GGUF" / "snapshots" / "pinned"
    companion.mkdir(parents=True)
    for shard in range(1, 4):
        (companion / f"gpt-oss-120b-mxfp4-{shard:05d}-of-00003.gguf").write_bytes(b"gguf")

    models = {model["model_id"]: model for model in discover_huggingface_models([tmp_path])}

    assert models["openai/gpt-oss-120b"]["configuration_support"] is None
    assert "companion snapshot" in models["openai/gpt-oss-120b"]["configuration_support_reason"]
    runnable = models["ggml-org/gpt-oss-120b-GGUF"]
    assert runnable["configuration_support"] == "gpt-oss-llama-vulkan"
    assert runnable["artifacts"][0]["artifact_id"] == "gpt-oss-120b-mxfp4"


def test_gpt_oss_consolidated_gguf_is_configurable(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--ggml-org--gpt-oss-120b-GGUF" / "snapshots" / "consolidated"
    snapshot.mkdir(parents=True)
    (snapshot / "gpt-oss-120b-MXFP4.gguf").write_bytes(b"gguf")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["configuration_support"] == "gpt-oss-llama-vulkan"
    assert model["artifacts"] == [
        {
            "artifact_id": "gpt-oss-120b-mxfp4",
            "kind": "gguf",
            "format": "mxfp4",
            "filenames": ["gpt-oss-120b-MXFP4.gguf"],
        }
    ]


def test_identifies_diffusiongemma_configuration_support(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--google--diffusiongemma" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "model_type": "diffusion_gemma",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["generation_family_hint"] == "text-diffusion"
    assert model["configuration_support"] == "diffusiongemma-transformers"


def test_does_not_offer_generic_configuration_for_quantised_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Org--Quantised" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DemoForCausalLM"],
                "model_type": "demo",
                "quantization_config": {"quant_method": "gptq"},
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")

    model = discover_huggingface_models([tmp_path])[0]

    assert model["configuration_support"] is None
    assert "Quantised snapshots" in model["configuration_support_reason"]
