from __future__ import annotations

from modeldeck.compatibility import CompatibilityStore, evidence_fingerprint


def test_fingerprint_is_stable_and_version_sensitive() -> None:
    first = {"model_id": "org/model", "runtime": "mock", "rocm_version": "7.1.1"}
    reordered = {"rocm_version": "7.1.1", "runtime": "mock", "model_id": "org/model"}
    changed = {**first, "rocm_version": "7.2.0"}
    assert evidence_fingerprint(first) == evidence_fingerprint(reordered)
    assert evidence_fingerprint(first) != evidence_fingerprint(changed)


def test_fingerprint_invalidates_native_fp8_kernel_and_tuning_changes() -> None:
    baseline = {
        "model_id": "Qwen/Qwen3.8-27B-FP8",
        "runtime": "qwen38-fp8-chat-transformers-rocm",
        "execution_mode": "native_fp8",
        "kernel_commit": "a" * 40,
        "kernel_manifest_sha256": "b" * 64,
        "tuning_profile_sha256": "c" * 64,
        "triton_version": "3.5.1+rocm7.2.1.gita272dfa8",
        "kernels_version": "0.15.2",
    }

    for field in ("kernel_commit", "kernel_manifest_sha256", "tuning_profile_sha256"):
        assert evidence_fingerprint(baseline) != evidence_fingerprint(
            {**baseline, field: "d" * len(baseline[field])}
        )


def test_records_compatibility_without_overwriting_negative_history(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()
    evidence = {"model_id": "org/model", "runtime": "transformers-rocm", "rocm_version": "7.2.1"}
    failed = store.record_test(evidence, result="transient-failure", failure_class="smoke-failure")
    passed = store.record_test(evidence, result="tested-working")
    observation = store.record_test_observation(
        passed["id"],
        {
            "shutdown_result": "success",
            "memory_recovery_result": "not-measured-process-exit-confirmed",
        },
    )
    records = store.list_tests()
    assert failed["fingerprint"] == passed["fingerprint"]
    assert [record["result"] for record in records] == ["tested-working", "transient-failure"]
    assert observation["observation"]["shutdown_result"] == "success"
    assert "memory_recovery_result" not in records[0]["evidence"]
    assert records[0]["observations"][0]["observation"]["memory_recovery_result"] == (
        "not-measured-process-exit-confirmed"
    )
    assert records[0]["fingerprint_version"] == 2


def test_model_cache_policy_defaults_allowed_and_persists_disallowed_revision(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()

    assert store.model_cache_allowed("google/model", "revision-1") is True
    store.set_model_cache_allowed("google/model", "revision-1", allowed=False)

    assert store.model_cache_allowed("google/model", "revision-1") is False
    assert store.list_model_cache_policy() == {("google/model", "revision-1"): False}


def test_model_capability_policy_defaults_disallowed_and_preserves_intent(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()

    assert store.model_capability_allowed("Qwen/model", "revision-1", "general-chat") is False
    store.set_model_capability_allowed("Qwen/model", "revision-1", "general-chat", allowed=True)
    store.set_model_cache_allowed("Qwen/model", "revision-1", allowed=False)

    assert store.model_capability_allowed("Qwen/model", "revision-1", "general-chat") is True
    assert store.model_cache_allowed("Qwen/model", "revision-1") is False


def test_route_tool_calling_rehearsal_state_is_revision_scoped_and_coarse(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()

    assert store.route_tool_calling_state("profile-1", 1, "capability-1") == {
        "supported": False,
        "rehearsed": False,
        "last_rehearsal": None,
        "failure_code": None,
    }
    stored = store.save_route_tool_calling_rehearsal(
        "profile-1",
        1,
        "capability-1",
        supported=True,
        failure_code=None,
        evidence={"probe_count": 2, "probes": [{"tool_call_count": 1, "result_category": "valid"}]},
    )

    assert stored["supported"] is True
    assert stored["rehearsed"] is True
    assert stored["last_rehearsal"]
    assert stored["failure_code"] is None
    assert store.route_tool_calling_state("profile-1", 2, "capability-1")["supported"] is False


def test_capability_setup_is_idempotent_and_events_are_append_only(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()
    setup = {
        "id": "setup-1",
        "request_id": "request-1",
        "request_fingerprint": "a" * 64,
        "state": "queued",
    }

    assert store.create_capability_setup(setup) == setup
    assert store.create_capability_setup({**setup, "id": "ignored-duplicate"}) == setup
    first = store.record_capability_setup_event("setup-1", "queued", {"message": "Queued"})
    second = store.record_capability_setup_event("setup-1", "starting-worker", {"message": "Starting"})

    assert [item["id"] for item in store.list_capability_setup_events("setup-1")] == [
        first["id"],
        second["id"],
    ]
