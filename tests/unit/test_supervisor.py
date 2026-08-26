from __future__ import annotations

import hashlib
import socket
import sys

import pytest
from modeldeck.llama_runtime import LLAMA_CPP_COMMIT, QwenLlamaManifest, TrustedArtefact
from modeldeck.profiles import LocalProfileRequest, create_local_profile
from modeldeck.protocol import LifecycleClass
from modeldeck.qwen_candidates import candidate_path
from modeldeck.runtime_trust import TRUSTED_RUNTIME_IDS
from modeldeck.speechshift import (
    QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
    QWEN_TTS_MAXIMUM_CODEC_TOKENS,
    QWEN_TTS_VOICES,
    SPEECHSHIFT_MODEL_SPECS,
)
from modeldeck.supervisor.service import (
    TRUSTED_LAUNCH_BUILDERS,
    WorkerSupervisor,
    build_mock_worker_command,
    build_worker_launch,
    classify_log_level,
    port_available,
    redact_log,
)

from tests.model_profiles import default_model_profiles


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_port_probe_uses_address_reuse_and_rejects_an_active_listener() -> None:
    port = free_port()
    assert port_available(port)

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        assert not port_available(port)


def test_every_trusted_runtime_has_an_explicit_launch_builder() -> None:
    assert set(TRUSTED_LAUNCH_BUILDERS) == set(TRUSTED_RUNTIME_IDS)


def test_worker_command_is_an_argument_array_with_allowlisted_values() -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    command = build_mock_worker_command(profile)
    assert command[:3] == [sys.executable, "-m", "modeldeck.workers.mock_worker"]
    port_index = command.index("--port")
    assert command[port_index : port_index + 2] == ["--port", "8610"]
    assert all(";" not in argument for argument in command)


def test_mock_launch_includes_only_persisted_contract_scenario_options() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    profile = base.model_copy(
        update={
            "settings": {
                "mock_contract_id": "openai-chat-v1",
                "mock_scenario": "delayed",
                "mock_delay_ms": 1250,
            }
        }
    )

    command = build_mock_worker_command(profile)

    assert command[command.index("--contract") + 1] == "openai-chat-v1"
    assert command[command.index("--scenario") + 1] == "delayed"
    assert command[command.index("--delay-ms") + 1] == "1250"


def test_rocm_launch_requires_project_local_runtime(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    missing = tmp_path / "missing-python"
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(missing))
    with pytest.raises(ValueError, match="setup.ps1"):
        build_worker_launch(profile)


@pytest.mark.asyncio
async def test_supervisor_registers_and_removes_only_stopped_profiles() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([])
    supervisor.register_profile(base)

    assert supervisor.get_worker(base.id)["state"] == "stopped"
    await supervisor.remove_profile(base.id)

    with pytest.raises(KeyError, match="Unknown worker"):
        supervisor.get_worker(base.id)


def test_supervisor_reports_the_environment_passed_to_the_worker() -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([profile])
    supervisor.workers[profile.id].launch_environment = build_worker_launch(profile).environment

    assert supervisor.worker_environment(profile.id) == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LD_PRELOAD": None,
    }


def test_rocm_launch_preserves_virtual_environment_entrypoint(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    launch = build_worker_launch(profile)
    assert launch.command[0] == str(runtime_python.absolute())
    assert launch.command[0] != str(runtime_python.resolve())


@pytest.mark.parametrize("profile_id", ["qwen-small-rocm", "qwen-1-5b-rocm", "qwen-3b-rocm"])
def test_qwen_launches_are_allowlisted_offline_and_cache_pinned(monkeypatch, tmp_path, profile_id) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == profile_id)
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.autoregressive_worker",
    ]
    assert launch.command[launch.command.index("--model-id") + 1] == profile.model_id
    assert launch.command[launch.command.index("--revision") + 1] == profile.revision
    assert launch.command[launch.command.index("--port") + 1] == str(profile.port)
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == "/mnt/work/models/huggingface/hub"


def test_wayfinder_prefix_cache_launch_flag_is_opt_in(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "qwen-small-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))

    disabled = build_worker_launch(profile)
    enabled_profile = profile.model_copy(
        update={"settings": {**profile.settings, "prefix_cache_enabled": True}}
    )
    enabled = build_worker_launch(enabled_profile)

    assert "--prefix-cache-enabled" not in disabled.command
    assert "--prefix-cache-enabled" in enabled.command


def test_diffusion_rocm_launch_is_allowlisted_and_offline(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "diffusiongemma-rocm")
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    launch = build_worker_launch(profile)
    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.text_diffusion_worker",
    ]
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == "/mnt/work/models/huggingface/hub"
    assert "LD_PRELOAD" not in launch.environment


def test_scenechat_launch_is_allowlisted_offline_and_api_key_scoped(monkeypatch, tmp_path) -> None:
    profile = next(
        profile for profile in default_model_profiles() if profile.id == "scenechat-gemma4-e2b-rocm"
    )
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    monkeypatch.setenv("MODELDECK_SCENECHAT_API_KEY", "test-local-key")

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.scenechat_worker",
    ]
    assert launch.command[launch.command.index("--port") + 1] == "8000"
    assert launch.command[launch.command.index("--cache-root") + 1] == ("/mnt/work/models/huggingface/hub")
    assert launch.command[launch.command.index("--maximum-new-tokens") + 1] == "512"
    assert launch.command[launch.command.index("--generation-timeout-seconds") + 1] == "60"
    assert launch.command[launch.command.index("--visual-token-budget") + 1] == "280"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["MODELDECK_SCENECHAT_API_KEY"] == "test-local-key"


def test_qwen35_scenechat_launch_uses_dedicated_offline_adapter(monkeypatch, tmp_path) -> None:
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="Qwen/Qwen3.5-4B",
            revision="a" * 40,
            alias="qwen35-4b",
            maximum_new_tokens=1024,
            visual_token_budget=140,
        ),
        cache_root=tmp_path,
        port=8630,
        configuration_support="scenechat-qwen35",
    )
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.qwen35_worker",
    ]
    assert launch.command[launch.command.index("--model-id") + 1] == "Qwen/Qwen3.5-4B"
    assert launch.command[launch.command.index("--maximum-new-tokens") + 1] == "1024"
    assert launch.command[launch.command.index("--visual-token-budget") + 1] == "140"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == str(tmp_path)


def test_qwen38_llamacpp_launch_enforces_adaptive_thinking(tmp_path) -> None:
    artefact = tmp_path / "Qwen3.8-27B-Q8_0.gguf"
    artefact.write_bytes(b"gguf")
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="ggml-org/Qwen3.8-27B-GGUF",
            revision="97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827",
            alias="qwen38-deep",
            runtime_template_id="qwen38-llamacpp-q8-mtp-vulkan",
            context_length=8192,
        ),
        cache_root=tmp_path,
        artifact_path=artefact,
        port=8630,
        configuration_support="qwen38-llamacpp-q8-mtp-vulkan",
    )

    launch = build_worker_launch(profile)

    assert launch.command[launch.command.index("--thinking-mode") + 1] == "adaptive"
    profile.settings["thinking_mode"] = "disabled"
    with pytest.raises(ValueError, match="thinking_mode=adaptive"):
        build_worker_launch(profile)


def test_qwen38_llamacpp_disabled_thinking_uses_its_distinct_template(tmp_path) -> None:
    artefact = tmp_path / "Qwen3.8-27B-Q8_0.gguf"
    artefact.write_bytes(b"gguf")
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="ggml-org/Qwen3.8-27B-GGUF",
            revision="97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827",
            alias="qwen38-no-thinking",
            runtime_template_id="qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
            context_length=8192,
        ),
        cache_root=tmp_path,
        artifact_path=artefact,
        port=8630,
        configuration_support="qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
    )

    launch = build_worker_launch(profile)

    assert launch.command[launch.command.index("--thinking-mode") + 1] == "disabled"
    profile.settings["thinking_mode"] = "adaptive"
    with pytest.raises(ValueError, match="thinking_mode=disabled"):
        build_worker_launch(profile)


def test_qwen35_llamacpp_launch_enforces_disabled_thinking(tmp_path) -> None:
    artefact = tmp_path / "Qwen_Qwen3.5-4B-Q8_0.gguf"
    artefact.write_bytes(b"gguf")
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
            revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
            alias="wayfinder-fast-qwen35-4b",
            runtime_template_id="qwen35-llamacpp-q8-vulkan",
            context_length=8192,
        ),
        cache_root=tmp_path,
        artifact_path=artefact,
        port=8630,
        configuration_support="qwen35-llamacpp-q8-vulkan",
    )

    launch = build_worker_launch(profile)

    assert launch.command[launch.command.index("--runtime-profile") + 1] == "qwen35-4b-q8-vulkan"
    assert launch.command[launch.command.index("--thinking-mode") + 1] == "disabled"
    profile.settings["thinking_mode"] = "adaptive"
    with pytest.raises(ValueError, match="thinking_mode=disabled"):
        build_worker_launch(profile)


def test_qwen35_llamacpp_adaptive_thinking_uses_its_distinct_template(tmp_path) -> None:
    artefact = tmp_path / "Qwen_Qwen3.5-4B-Q8_0.gguf"
    artefact.write_bytes(b"gguf")
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
            revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
            alias="qwen35-adaptive",
            runtime_template_id="qwen35-llamacpp-q8-vulkan-adaptive",
            context_length=8192,
            maximum_new_tokens=1024,
        ),
        cache_root=tmp_path,
        artifact_path=artefact,
        port=8630,
        configuration_support="qwen35-llamacpp-q8-vulkan-adaptive",
    )

    launch = build_worker_launch(profile)

    assert launch.command[launch.command.index("--thinking-mode") + 1] == "adaptive"
    assert launch.command[launch.command.index("--maximum-new-tokens") + 1] == "1024"
    profile.settings["thinking_mode"] = "disabled"
    with pytest.raises(ValueError, match="thinking_mode=adaptive"):
        build_worker_launch(profile)


def test_approved_qwen35_candidate_launch_is_bound_to_manifest_and_data_dir(tmp_path) -> None:
    payload = b"approved-qwen35-9b"
    digest = hashlib.sha256(payload).hexdigest()
    candidate_id = f"qwen35-9b-q8-{digest[:12]}"
    artefact = tmp_path / "Qwen_Qwen3.5-9B-Q8_0.gguf"
    artefact.write_bytes(payload)
    manifest = QwenLlamaManifest(
        format="modeldeck-qwen-llamacpp-runtime",
        version=1,
        id=candidate_id,
        status="approved-local",
        original_model_id="Qwen/Qwen3.5-9B",
        original_model_revision=None,
        artefact_model_id="bartowski/Qwen_Qwen3.5-9B-GGUF",
        artefact_revision="a" * 40,
        quantisation="Q8_0",
        model=TrustedArtefact(
            filename=artefact.name,
            size=len(payload),
            sha256=digest,
            dtype="Q8_0",
        ),
        llama_cpp_commit=LLAMA_CPP_COMMIT,
        operating_system="linux",
        architecture="x86_64",
        backend="Vulkan",
        qwen_architecture="qwen35",
        chat_template_fingerprint=f"embedded-gguf-sha256:{digest}",
        context_length=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        source_url="https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF",
        licence="Apache-2.0",
    )
    data_dir = tmp_path / "data"
    destination = candidate_path(data_dir, candidate_id)
    destination.parent.mkdir(parents=True)
    destination.write_text(manifest.model_dump_json(), encoding="utf-8")
    profile = create_local_profile(
        LocalProfileRequest(
            model_id=manifest.artefact_model_id,
            revision=manifest.artefact_revision,
            alias="qwen35-9b-adaptive",
            runtime_template_id="qwen35-local-q8-vulkan-adaptive",
            context_length=8192,
        ),
        cache_root=tmp_path,
        artifact_path=artefact,
        candidate_manifest_id=candidate_id,
        port=8630,
        configuration_support="qwen35-local-q8-vulkan-adaptive",
    )

    launch = build_worker_launch(profile, data_dir=data_dir)

    assert launch.command[launch.command.index("--runtime-profile") + 1] == ("qwen35-approved-q8-vulkan")
    assert launch.command[launch.command.index("--candidate-manifest-id") + 1] == candidate_id
    assert launch.command[launch.command.index("--data-dir") + 1] == str(data_dir)
    assert launch.command[launch.command.index("--thinking-mode") + 1] == "adaptive"


def test_qwen35_chat_launch_uses_dedicated_offline_adapter(monkeypatch, tmp_path) -> None:
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="Qwen/Qwen3.5-4B",
            revision="a" * 40,
            alias="qwen35-chat-4b",
            maximum_new_tokens=512,
        ),
        cache_root=tmp_path,
        port=8630,
        configuration_support="qwen35-chat-transformers-rocm",
    )
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.qwen35_chat_worker",
    ]
    assert launch.command[launch.command.index("--model-id") + 1] == "Qwen/Qwen3.5-4B"
    assert launch.command[launch.command.index("--maximum-new-tokens") + 1] == "512"
    assert launch.command[launch.command.index("--thinking-mode") + 1] == "disabled"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == str(tmp_path)


@pytest.mark.parametrize(
    ("configuration_support", "worker_module"),
    [
        ("scenechat-qwen38-fp8", "modeldeck.workers.qwen35_worker"),
        ("qwen38-fp8-chat-transformers-rocm", "modeldeck.workers.qwen35_chat_worker"),
    ],
)
def test_qwen38_native_fp8_launch_is_exact_offline_and_separate(
    monkeypatch, tmp_path, configuration_support, worker_module
) -> None:
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="Qwen/Qwen3.8-27B-FP8",
            revision="a" * 40,
            alias="qwen38-native-fp8",
        ),
        cache_root=tmp_path,
        port=8630,
        configuration_support=configuration_support,
    )
    runtime_python = tmp_path / "bin/python"
    runtime_python.parent.mkdir()
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_PYTHON", str(runtime_python))
    monkeypatch.setenv("MODELDECK_DATA_DIR", str(tmp_path / "data"))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [str(runtime_python.absolute()), "-m", worker_module]
    assert launch.command[launch.command.index("--execution-mode") + 1] == "native_fp8"
    assert launch.command[launch.command.index("--data-dir") + 1] == str(tmp_path / "data")
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["TRITON_CACHE_AUTOTUNING"] == "1"
    assert launch.environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_opus_translation_launch_is_isolated_directional_and_offline(monkeypatch, tmp_path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["Helsinki-NLP/opus-mt-en-fr"]
    profile = create_local_profile(
        LocalProfileRequest(
            model_id=spec.model_id,
            revision=spec.revision,
            alias="speechshift-en-fr",
        ),
        cache_root=tmp_path,
        port=8630,
        configuration_support="opus-translation-cpu",
    )
    runtime_python = tmp_path / "marian/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_MARIAN_PYTHON", str(runtime_python))

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.translation_worker",
    ]
    assert launch.command[launch.command.index("--source-language") + 1] == "en"
    assert launch.command[launch.command.index("--target-language") + 1] == "fr"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"


def test_qwen_tts_launch_is_isolated_offline_and_has_no_arch_override(monkeypatch, tmp_path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
    profile = create_local_profile(
        LocalProfileRequest(
            model_id=spec.model_id,
            revision=spec.revision,
            alias="speechshift-voice",
        ),
        cache_root=tmp_path,
        port=8631,
        configuration_support="qwen3-tts-rocm",
    )
    assert profile.settings["allowed_voices"] == ",".join(QWEN_TTS_VOICES)
    runtime_python = tmp_path / "tts/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_QWEN_TTS_PYTHON", str(runtime_python))
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "unsafe")

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.tts_worker",
    ]
    assert launch.command[launch.command.index("--maximum-codec-tokens") + 1] == str(
        QWEN_TTS_MAXIMUM_CODEC_TOKENS
    )
    assert launch.command[launch.command.index("--maximum-audio-seconds") + 1] == "90"
    assert launch.command[launch.command.index("--generation-timeout-seconds") + 1] == str(
        QWEN_TTS_GENERATION_TIMEOUT_SECONDS
    )
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HSA_OVERRIDE_GFX_VERSION" not in launch.environment


def test_whisper_launch_is_isolated_offline_and_allowlisted(monkeypatch, tmp_path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    profile = create_local_profile(
        LocalProfileRequest(model_id=spec.model_id, revision=spec.revision, alias="speechshift-stt"),
        cache_root=tmp_path,
        port=8632,
        configuration_support="whisper-small-en-rocm",
    )
    runtime_python = tmp_path / "whisper/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_WHISPER_PYTHON", str(runtime_python))
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "unsafe")

    launch = build_worker_launch(profile)

    assert launch.command[:3] == [
        str(runtime_python.absolute()),
        "-m",
        "modeldeck.workers.speech_recognition_worker",
    ]
    assert launch.command[launch.command.index("--recognition-timeout-seconds") + 1] == "30"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert launch.environment["HF_HUB_CACHE"] == str(tmp_path)
    assert "HSA_OVERRIDE_GFX_VERSION" not in launch.environment


def test_diffusion_q4_launch_uses_isolated_runtime_and_checkpoint(monkeypatch, tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "diffusiongemma-q4-rocm")
    runtime_python = tmp_path / "q4/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(sys.executable)
    monkeypatch.setenv("MODELDECK_ROCM72_Q4_PYTHON", str(runtime_python))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    launch = build_worker_launch(profile)

    assert launch.command[0] == str(runtime_python.absolute())
    assert "--cache-root" not in launch.command
    assert launch.command[launch.command.index("--q4-checkpoint-dir") + 1].endswith(
        "/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32"
    )
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_HUB_CACHE" not in launch.environment


@pytest.mark.asyncio
async def test_starting_exclusive_worker_stops_existing_exclusive_worker() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-diffusion")
    first_port = free_port()
    second_port = free_port()
    while second_port == first_port:
        second_port = free_port()
    first = base.model_copy(update={"id": "mock-diffusion-one", "port": first_port})
    second = base.model_copy(update={"id": "mock-diffusion-two", "port": second_port})
    supervisor = WorkerSupervisor([first, second], startup_timeout=8, stop_timeout=2)

    try:
        await supervisor.start(first.id)
        await supervisor.start(second.id)

        assert supervisor.get_worker(first.id)["state"] == "stopped"
        assert supervisor.get_worker(first.id)["pid"] is None
        assert supervisor.get_worker(second.id)["state"] == "ready"
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_on_demand_worker_can_run_with_exclusive_worker() -> None:
    base = next(profile for profile in default_model_profiles() if profile.id == "mock-diffusion")
    exclusive_port = free_port()
    on_demand_port = free_port()
    while on_demand_port == exclusive_port:
        on_demand_port = free_port()
    exclusive = base.model_copy(update={"id": "mock-exclusive", "port": exclusive_port})
    on_demand = base.model_copy(
        update={
            "id": "mock-on-demand",
            "port": on_demand_port,
            "lifecycle": LifecycleClass.ON_DEMAND,
        }
    )
    supervisor = WorkerSupervisor([exclusive, on_demand], startup_timeout=8, stop_timeout=2)

    try:
        await supervisor.start(exclusive.id)
        await supervisor.start(on_demand.id)

        assert supervisor.get_worker(exclusive.id)["state"] == "ready"
        assert supervisor.get_worker(on_demand.id)["state"] == "ready"
    finally:
        await supervisor.stop_all()


def test_log_redaction_removes_prompt_and_credentials() -> None:
    assert redact_log("prompt=private visitor words") == "prompt=[redacted]"
    assert "secret" not in redact_log('{"api_key":"secret","status":"failed"}')


def test_worker_logs_are_redacted_bounded_and_restored(tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([profile], log_dir=tmp_path)
    supervisor._append_log(profile.id, "stderr", "prompt=private visitor words")
    for index in range(501):
        supervisor._append_log(profile.id, "stderr", f"diagnostic {index}")

    restored = WorkerSupervisor([profile], log_dir=tmp_path)
    logs = restored.logs(profile.id)

    assert len(logs) == 500
    assert all("private visitor words" not in item["message"] for item in logs)
    assert logs[-1]["message"] == "diagnostic 500"
    assert len((tmp_path / "mock-ar.jsonl").read_text().splitlines()) == 500


def test_worker_logs_are_scoped_to_the_current_session_and_classified(tmp_path) -> None:
    profile = next(profile for profile in default_model_profiles() if profile.id == "mock-ar")
    supervisor = WorkerSupervisor([profile], log_dir=tmp_path)
    worker = supervisor.workers[profile.id]
    worker.log_session_id = "first"
    supervisor._append_log(profile.id, "stderr", "ERROR: old failure")
    worker.log_session_id = "second"
    supervisor._append_log(profile.id, "stderr", "UserWarning: current warning")

    logs = supervisor.logs(profile.id)

    assert len(logs) == 1
    assert logs[0]["session_id"] == "second"
    assert logs[0]["level"] == "warning"
    assert classify_log_level("Traceback (most recent call last)") == "error"
    assert classify_log_level('{{- raise_exception("Invalid chat-template message") }}') == "info"
    assert classify_log_level("Application startup complete") == "info"
