from __future__ import annotations

import json

import pytest
from modeldeck.smoke_probes import PROTOCOL_PROBES, probe_for_capability, validate_probe_response


@pytest.mark.parametrize("probe", PROTOCOL_PROBES.values())
@pytest.mark.parametrize("payload", [None, [], "ready", 7, {}, {"error": "failed"}])
def test_probes_reject_missing_protocol_evidence(probe, payload) -> None:
    assert not validate_probe_response(probe, payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "answer", "frames": []},
        {"text": "answer", "frames": [{"text": ""}]},
        {"text": "answer", "frames": [{"text": "answer", "cancelled": True}]},
        {"text": "answer", "frames": [{"text": "answer"}], "state": "cancelled"},
        {"text": "answer", "frames": [{"text": None}, {"text": "answer"}]},
    ],
)
def test_diffusion_probe_rejects_empty_final_output_and_cancelled_runs(payload) -> None:
    assert not validate_probe_response(probe_for_capability("text-refinement"), payload)


def test_scene_probe_requires_a_schema_valid_analysis() -> None:
    probe = probe_for_capability("scene-analysis")
    valid = {
        "summary": "A bounded local probe image.",
        "objects": [],
        "relationships": [],
        "uncertainties": ["The image is intentionally minimal."],
        "safety_notes": [],
    }

    assert validate_probe_response(
        probe,
        {"choices": [{"message": {"content": json.dumps(valid)}}]},
    )
    assert not validate_probe_response(
        probe,
        {"choices": [{"message": {"content": json.dumps({"summary": "Incomplete"})}}]},
    )


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {"audio_seconds": 0.1},
        {"audio_seconds": 0, "inference_seconds": 0, "total_worker_seconds": 0},
        {"audio_seconds": True, "inference_seconds": 0, "total_worker_seconds": 0},
        {"audio_seconds": 0.1, "inference_seconds": float("nan"), "total_worker_seconds": 0},
    ],
)
def test_empty_transcript_requires_completed_inference_evidence(metrics) -> None:
    payload = {"object": "audio.transcription", "language": "en", "text": "", "metrics": metrics}
    assert not validate_probe_response(probe_for_capability("speech-recognition"), payload)
