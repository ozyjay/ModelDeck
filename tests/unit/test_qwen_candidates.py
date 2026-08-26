from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from modeldeck.capabilities import compatible_runtime_template_ids
from modeldeck.catalogue import discover_huggingface_models
from modeldeck.qwen_candidates import (
    approve_candidate,
    inspect_candidate,
    load_candidate,
)
from modeldeck.registry import runtime_template_registrations

REPOSITORY = "bartowski/Qwen_Qwen3.5-9B-GGUF"
REVISION = "182be2fd6c7bc44887d88a91cb03ff009cc9f549"
FILENAME = "Qwen_Qwen3.5-9B-Q8_0.gguf"


def _candidate_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    cache_root = tmp_path / "hub"
    model_dir = cache_root / "models--bartowski--Qwen_Qwen3.5-9B-GGUF"
    snapshot = model_dir / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    payload = b"local-qwen35-q8-gguf"
    (snapshot / FILENAME).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    tree_dir = model_dir / "trees"
    tree_dir.mkdir()
    (tree_dir / f"{REVISION}.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": {
                    FILENAME: {
                        "size": len(payload),
                        "lfs_size": len(payload),
                        "lfs_sha256": digest,
                        "blob_id": "a" * 40,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    library_root = tmp_path / ".cache" / "huggingfacepull" / "library"
    marker_dir = library_root / "bartowski--Qwen_Qwen3.5-9B-GGUF" / REVISION
    marker_dir.mkdir(parents=True)
    (marker_dir / ".huggingfacepull.json").write_text(
        json.dumps(
            {
                "repo_id": REPOSITORY,
                "expected_commit": REVISION,
                "revision": REVISION,
                "resolved_revision": REVISION,
                "snapshot_path": str(snapshot),
                "files": [{"path": FILENAME, "size": len(payload), "blob_id": "a" * 40}],
            }
        ),
        encoding="utf-8",
    )
    return cache_root, snapshot, library_root, digest


def test_approves_exact_huggingfacepull_qwen35_candidate(tmp_path: Path) -> None:
    _, snapshot, library_root, digest = _candidate_fixture(tmp_path)
    data_dir = tmp_path / "data"

    inspection = inspect_candidate(
        REPOSITORY,
        REVISION,
        snapshot,
        data_dir=data_dir,
        library_root=library_root,
    )
    assert inspection.eligible is True
    assert inspection.approved is False

    manifest = approve_candidate(
        REPOSITORY,
        REVISION,
        snapshot,
        data_dir=data_dir,
        library_root=library_root,
    )

    assert manifest.id == f"qwen35-9b-q8-{digest[:12]}"
    assert manifest.original_model_id == "Qwen/Qwen3.5-9B"
    assert manifest.original_model_revision is None
    assert manifest.model.sha256 == digest
    assert load_candidate(data_dir, manifest.id) == manifest


def test_rejects_candidate_when_file_does_not_match_tree_checksum(tmp_path: Path) -> None:
    _, snapshot, library_root, _ = _candidate_fixture(tmp_path)
    (snapshot / FILENAME).write_bytes(b"tampered-qwen35-q8")
    tree_path = snapshot.parent.parent / "trees" / f"{REVISION}.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    tree["files"][FILENAME]["size"] = len(b"tampered-qwen35-q8")
    tree["files"][FILENAME]["lfs_size"] = len(b"tampered-qwen35-q8")
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    marker_path = library_root / "bartowski--Qwen_Qwen3.5-9B-GGUF" / REVISION / ".huggingfacepull.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["files"][0]["size"] = len(b"tampered-qwen35-q8")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        approve_candidate(
            REPOSITORY,
            REVISION,
            snapshot,
            data_dir=tmp_path / "data",
            library_root=library_root,
        )


def test_approved_candidate_uses_reusable_thinking_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, snapshot, library_root, _ = _candidate_fixture(tmp_path)
    data_dir = tmp_path / "data"
    approve_candidate(
        REPOSITORY,
        REVISION,
        snapshot,
        data_dir=data_dir,
        library_root=library_root,
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    model = discover_huggingface_models([cache_root], data_dir=data_dir)[0]

    assert model["configuration_support"] == "qwen35-local-q8-vulkan"
    assert model["candidate_registration"]["approved"] is True
    assert model["artifacts"][0]["candidate_manifest_id"].startswith("qwen35-9b-q8-")
    registrations = runtime_template_registrations()
    assert compatible_runtime_template_ids("general-chat", model["configuration_support"], registrations) == [
        "qwen35-local-q8-vulkan",
        "qwen35-local-q8-vulkan-adaptive",
    ]
