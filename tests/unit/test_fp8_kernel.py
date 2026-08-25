from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from modeldeck import fp8_kernel


def _kernel_snapshot(tmp_path, monkeypatch):
    snapshot = fp8_kernel.kernel_snapshot_path(tmp_path)
    variant = snapshot / "build" / "torch-rocm"
    variant.mkdir(parents=True)
    metadata = {
        "name": "finegrained-fp8",
        "id": "_finegrained_fp8_rocm_846165b",
        "version": 3,
        "backend": {"type": "rocm"},
    }
    files = {
        "__init__.py": b"entrypoint",
        "matmul.py": b"kernel",
        "metadata.json": json.dumps(metadata).encode(),
    }
    for name, content in files.items():
        path = variant / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    ref = snapshot.parents[1] / "refs" / "v3"
    ref.parent.mkdir(parents=True)
    ref.write_text(fp8_kernel.KERNEL_COMMIT, encoding="utf-8")
    manifest = {
        "repo_id": fp8_kernel.KERNEL_REPO_ID,
        "revision": "v3",
        "commit": fp8_kernel.KERNEL_COMMIT,
        "kernel_id": "_finegrained_fp8_rocm_846165b",
        "kernel_version": 3,
        "backend": "rocm",
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    monkeypatch.setattr(
        fp8_kernel,
        "_trust_manifest_bytes",
        lambda: json.dumps(manifest, sort_keys=True).encode(),
    )
    return variant


def test_kernel_snapshot_validation_accepts_only_the_pinned_file_set(tmp_path, monkeypatch) -> None:
    variant = _kernel_snapshot(tmp_path, monkeypatch)

    validated = fp8_kernel.validate_fp8_kernel_snapshot(tmp_path)

    assert validated.variant_path == variant.resolve()
    assert validated.tuning_status == "missing"


def test_kernel_snapshot_validation_rejects_changed_or_extra_code(tmp_path, monkeypatch) -> None:
    variant = _kernel_snapshot(tmp_path, monkeypatch)
    (variant / "matmul.py").write_bytes(b"changed")

    with pytest.raises(fp8_kernel.FP8KernelValidationError, match="digest changed"):
        fp8_kernel.validate_fp8_kernel_snapshot(tmp_path)

    (variant / "matmul.py").write_bytes(b"kernel")
    (variant / "unexpected.py").write_bytes(b"code")
    with pytest.raises(fp8_kernel.FP8KernelValidationError, match="file set changed"):
        fp8_kernel.validate_fp8_kernel_snapshot(tmp_path)


def test_tuning_manifest_is_invalidated_by_the_complete_fingerprint(tmp_path) -> None:
    path = tmp_path / "tuning.json"
    fingerprint = {"gpu_architecture": "gfx1151", "kernel_commit": fp8_kernel.KERNEL_COMMIT}
    document = {
        "format": "modeldeck-fp8-tuning",
        "version": 1,
        "fingerprint": fingerprint,
        "winners": [
            {
                "n": 5120,
                "k": 17408,
                "block_size_m": 16,
                "dtype": "bfloat16",
                "num_warps": 4,
                "num_stages": 2,
                "status": "accepted",
            }
        ],
    }
    fp8_kernel.write_tuning_manifest(path, document)

    loaded = fp8_kernel.load_tuning_manifest(path, expected_fingerprint=fingerprint)

    assert loaded is not None
    assert loaded[0][(5120, 17408, 16, "bfloat16")] == {"num_warps": 4, "num_stages": 2}
    assert (
        fp8_kernel.load_tuning_manifest(
            path,
            expected_fingerprint={**fingerprint, "gpu_architecture": "gfx9999"},
        )
        is None
    )
    assert (
        fp8_kernel.load_tuning_manifest(
            path,
            expected_fingerprint=fingerprint,
            required_keys=frozenset({(5120, 17408, 16, "bfloat16"), (6144, 5120, 128, "bfloat16")}),
        )
        is None
    )


def test_config_selector_uses_winner_and_conservative_fallback(monkeypatch) -> None:
    selected = []

    class FakeAutotuner:
        arg_names = ["N", "K", "BLOCK_SIZE_M"]
        configs = ["upstream"]

        def run(self, *args, **kwargs):
            selected.append(self.configs[0])
            return "ok"

    monkeypatch.setitem(
        sys.modules,
        "triton",
        SimpleNamespace(Config=lambda values, **settings: {**values, **settings}),
    )
    tuner = FakeAutotuner()
    fp8_kernel._install_config_selector(
        tuner,
        {(5120, 17408, 16, "bfloat16"): {"num_warps": 8, "num_stages": 3}},
    )

    assert tuner.run(5120, 17408, 16) == "ok"
    assert tuner.run(6144, 5120, 16) == "ok"
    assert selected == [
        {"num_warps": 8, "num_stages": 3},
        fp8_kernel.DEFAULT_CONFIG,
    ]
    assert tuner.configs == ["upstream"]


def test_exact_transformers_skip_matcher_does_not_drop_similar_fp8_siblings() -> None:
    router = "model.language_model.layers.0.mlp.gate"

    assert fp8_kernel._should_convert_exact_module(router, [router]) is False
    assert fp8_kernel._should_convert_exact_module(router + ".weight", [router]) is False
    assert fp8_kernel._should_convert_exact_module(router + "_proj", [router]) is True
    assert fp8_kernel._should_convert_exact_module("model.lm_head", ["lm_head"]) is False
