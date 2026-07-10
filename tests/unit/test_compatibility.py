from __future__ import annotations

from modeldeck.compatibility import evidence_fingerprint


def test_fingerprint_is_stable_and_version_sensitive() -> None:
    first = {"model_id": "org/model", "runtime": "mock", "rocm_version": "7.1.1"}
    reordered = {"rocm_version": "7.1.1", "runtime": "mock", "model_id": "org/model"}
    changed = {**first, "rocm_version": "7.2.0"}
    assert evidence_fingerprint(first) == evidence_fingerprint(reordered)
    assert evidence_fingerprint(first) != evidence_fingerprint(changed)
