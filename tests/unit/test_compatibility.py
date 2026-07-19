from __future__ import annotations

from modeldeck.compatibility import CompatibilityStore, evidence_fingerprint


def test_fingerprint_is_stable_and_version_sensitive() -> None:
    first = {"model_id": "org/model", "runtime": "mock", "rocm_version": "7.1.1"}
    reordered = {"rocm_version": "7.1.1", "runtime": "mock", "model_id": "org/model"}
    changed = {**first, "rocm_version": "7.2.0"}
    assert evidence_fingerprint(first) == evidence_fingerprint(reordered)
    assert evidence_fingerprint(first) != evidence_fingerprint(changed)


def test_records_compatibility_without_overwriting_negative_history(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()
    evidence = {"model_id": "org/model", "runtime": "transformers-rocm", "rocm_version": "7.2.1"}
    failed = store.record_test(evidence, result="transient-failure", failure_class="smoke-failure")
    passed = store.record_test(evidence, result="tested-working")
    updated = store.update_test_evidence(
        passed["id"],
        {
            "shutdown_result": "success",
            "memory_recovery_result": "not-measured-process-exit-confirmed",
        },
    )
    records = store.list_tests()
    assert failed["fingerprint"] == passed["fingerprint"]
    assert [record["result"] for record in records] == ["tested-working", "transient-failure"]
    assert updated["evidence"]["shutdown_result"] == "success"
    assert records[0]["evidence"]["memory_recovery_result"] == "not-measured-process-exit-confirmed"


def test_model_cache_policy_defaults_allowed_and_persists_disallowed_revision(tmp_path) -> None:
    store = CompatibilityStore(tmp_path / "evidence.sqlite3")
    store.initialise()

    assert store.model_cache_allowed("google/model", "revision-1") is True
    store.set_model_cache_allowed("google/model", "revision-1", allowed=False)

    assert store.model_cache_allowed("google/model", "revision-1") is False
    assert store.list_model_cache_policy() == {("google/model", "revision-1"): False}
