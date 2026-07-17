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
    assert models[0]["runnable"] is False


def test_marks_incomplete_snapshot_partial(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Org--Partial" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.incomplete").write_bytes(b"partial")
    assert discover_huggingface_models([tmp_path])[0]["download_state"] == "partial"


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
    assert model["runnable"] is False
