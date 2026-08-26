from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from modeldeck import llama_runtime
from modeldeck.llama_runtime import LlamaBuildReceipt, QwenLlamaManifest, TrustedArtefact
from modeldeck.workers.llama_vulkan_worker import LlamaEvidence, qwen_llama_command, qwen_request


def _artefact(path: Path, content: bytes, dtype: str) -> TrustedArtefact:
    path.write_bytes(content)
    return TrustedArtefact(
        filename=path.name,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        dtype=dtype,
    )


def _manifest(snapshot: Path) -> QwenLlamaManifest:
    return QwenLlamaManifest(
        format="modeldeck-qwen-llamacpp-runtime",
        version=1,
        id="qwen38-q8-mtp-vulkan",
        status="reviewed-candidate",
        original_model_id="Qwen/Qwen3.8-27B",
        original_model_revision="1" * 40,
        artefact_model_id="ggml-org/Qwen3.8-27B-GGUF",
        artefact_revision="2" * 40,
        quantisation="Q8_0",
        model=_artefact(snapshot / "Qwen3.8-27B-Q8_0.gguf", b"model", "Q8_0"),
        projector=_artefact(snapshot / "mmproj-Qwen3.8-27B-BF16.gguf", b"projector", "BF16"),
        mtp_model=_artefact(snapshot / "mtp-Qwen3.8-27B-Q8_0.gguf", b"mtp", "Q8_0"),
        llama_cpp_commit=llama_runtime.LLAMA_CPP_COMMIT,
        operating_system="linux",
        architecture="x86_64",
        backend="Vulkan",
        qwen_architecture="qwen35",
        chat_template_fingerprint="embedded:test",
        context_length=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        mtp_draft_tokens=4,
        source_url="https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF",
        licence="Apache-2.0",
    )


def _validated_runtime(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = _manifest(snapshot)
    runtime_root = tmp_path / "runtime"
    executable = runtime_root / "bin" / "llama-server"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    executable.chmod(0o755)
    receipt = LlamaBuildReceipt(
        format="modeldeck-llama-build",
        version=1,
        commit=llama_runtime.LLAMA_CPP_COMMIT,
        executable_sha256=hashlib.sha256(b"executable").hexdigest(),
        backend="Vulkan",
        operating_system="linux",
        architecture="x86_64",
    )
    (executable.parent / "modeldeck-build.json").write_text(receipt.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(llama_runtime, "LLAMA_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(llama_runtime, "load_qwen_manifest", lambda _profile: manifest)
    return llama_runtime.validate_qwen_runtime(manifest.id, snapshot)


def test_packaged_qwen_q8_manifest_has_immutable_provenance() -> None:
    manifest = llama_runtime.load_qwen_manifest("qwen38-q8-mtp-vulkan")

    assert manifest.original_model_id == "Qwen/Qwen3.8-27B"
    assert manifest.artefact_revision == "97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827"
    assert manifest.model.sha256 == "f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8"
    assert manifest.projector.dtype == "BF16"
    assert manifest.mtp_draft_tokens == 4
    assert manifest.llama_cpp_commit == llama_runtime.LLAMA_CPP_COMMIT


def test_qwen_command_is_fixed_to_loopback_full_vulkan_offload_and_mtp(monkeypatch, tmp_path) -> None:
    runtime = _validated_runtime(monkeypatch, tmp_path)

    command = qwen_llama_command(runtime=runtime, port=49152)

    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--device") + 1] == "Vulkan0"
    assert command[command.index("--gpu-layers") + 1] == "all"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-n-max") + 1] == "4"
    assert "--mmproj" in command
    assert "--offline" in command


def test_qwen_runtime_rejects_a_tampered_artefact(monkeypatch, tmp_path) -> None:
    runtime = _validated_runtime(monkeypatch, tmp_path)
    runtime.model.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        llama_runtime.validate_qwen_runtime(runtime.manifest.id, runtime.model.parent)


def test_llama_evidence_requires_backend_offload_projector_and_mtp() -> None:
    evidence = LlamaEvidence()
    for line in (
        "ggml_vulkan: Vulkan0 AMD Radeon 8060S (RADV GFX1151)",
        "llm_load_print_meta: general.architecture = qwen35",
        "llm_load_print_meta: general.file_type = Q8_0",
        "load_tensors: offloaded 65/65 layers to GPU",
        "mtmd: loaded mmproj BF16 vision projector",
        "spec_type = draft-mtp",
        "slot print_timing: prompt eval time = 10 ms / 20 tokens (2000.00 tokens per second)",
        "slot print_timing: eval time = 100 ms / 10 tokens (100.00 tokens per second)",
        "slot print_timing: draft acceptance = 0.75000 ( 12 accepted / 16 generated)",
    ):
        evidence.feed(line, quantisation="Q8_0")

    assert evidence.startup_errors() == []
    assert evidence.draft_proposed == 16
    assert evidence.draft_accepted == 12
    assert evidence.acceptance_ratio == 0.75
    assert evidence.prompt_tokens_per_second == 2000.0
    assert evidence.generated_tokens_per_second == 100.0


def test_qwen_request_drops_backend_parameters_and_enforces_generation_limit() -> None:
    body = qwen_request(
        {
            "model": "client-alias",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "reasoning_effort": "high",
            "tools": [],
            "cache_prompt": True,
            "slot_id": 7,
        },
        model_id="ggml-org/Qwen3.8-27B-GGUF",
        maximum_new_tokens=512,
    )

    assert body["model"] == "ggml-org/Qwen3.8-27B-GGUF"
    assert body["reasoning_effort"] == "high"
    assert "cache_prompt" not in body
    assert "slot_id" not in body
    with pytest.raises(ValueError, match="generation limit"):
        qwen_request({"max_tokens": 513}, model_id="model", maximum_new_tokens=512)
