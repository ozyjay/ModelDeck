from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
from modeldeck.scenechat_evidence import BenchmarkCapture
from modeldeck.scenechat_experiment import (
    CATEGORIES,
    PRIMARY_MODELS,
    Configuration,
    Corpus,
    CorpusImage,
    Journal,
    Matrix,
    Review,
    blind_outputs,
    digest,
    freeze,
    qualification,
    read_attempts,
    schedule,
    summarise,
)
from modeldeck.scenechat_runner import SafetyMonitor, sensor_sample
from modeldeck.workers.scenechat_diagnostics import generation_receipt, repetition_diagnostics


@pytest.fixture
def corpus():
    return Corpus(
        images=[
            CorpusImage(
                id=f"{category}-{index}",
                path=f"{category}-{index}.jpg",
                sha256=digest([category, index]),
                category=category,
                split="development" if index < 2 else "holdout",
                provenance="operator-owned",
                approved_by="operator",
                non_visitor=True,
                facts={f"q{q + 1}": ["A visible object"] for q in range(7)},
            )
            for category in CATEGORIES
            for index in range(4)
        ]
    )


@pytest.fixture
def matrix():
    return Matrix(
        configurations=[
            Configuration(
                id=name,
                worker_id=f"00000000-0000-0000-0000-{index:012d}",
                model_id=model,
                revision=revision,
                runtime="vision-language-transformers-rocm"
                if index == 0
                else "qwen35-vision-language-transformers-rocm",
                worker_definition_sha256="a" * 64,
                bundle_sha256="b" * 64,
            )
            for index, (name, (model, revision)) in enumerate(PRIMARY_MODELS.items())
        ]
    )


def test_stage_counts_and_holdout_isolation(corpus, matrix):
    for stage, count in (("screen", 28), ("development", 294)):
        rows = schedule(corpus, matrix, stage)
        measured = [row for row in rows if not row.warmup]
        assert Counter(row.configuration_id for row in measured) == dict.fromkeys(PRIMARY_MODELS, count)
        assert all(
            next(i for i in corpus.images if i.id == row.image_id).split == "development" for row in rows
        )
        assert len({row.id for row in rows}) == len(rows)
    assert schedule(corpus, matrix, "screen")[0].configuration_id == "gemma-e2b"
    with pytest.raises(ValueError, match="two"):
        schedule(corpus, matrix, "holdout")
    matrix.configurations = matrix.configurations[:2]
    assert Counter(r.configuration_id for r in schedule(corpus, matrix, "holdout") if not r.warmup) == {
        "gemma-e2b": 490,
        "qwen-4b": 490,
    }


def test_freeze_detects_labels_or_worker_identity_changes(corpus, matrix):
    original = freeze(corpus, matrix, "development")
    assert freeze(corpus, matrix, "development") == original
    corpus.images[0].facts["q1"].append("A second object")
    assert freeze(corpus, matrix, "development")["sha256"] != original["sha256"]


def test_corpus_rejects_duplicates_and_missing_labels(corpus):
    value = corpus.model_dump()
    value["images"][0]["facts"].pop("q1")
    with pytest.raises(ValueError, match="seven"):
        Corpus.model_validate(value)
    value = corpus.model_dump()
    value["images"][1]["sha256"] = value["images"][0]["sha256"]
    with pytest.raises(ValueError, match="unique"):
        Corpus.model_validate(value)


def test_interrupted_and_unstarted_attempts_survive_torn_write(tmp_path, corpus, matrix):
    requests = [r.model_dump() for r in schedule(corpus, matrix, "screen")[:3]]
    path = tmp_path / "attempts.jsonl"
    journal = Journal(path)
    journal.append({"kind": "started", "attempt_id": requests[0]["id"]})
    journal.append(
        {"kind": "finished", "attempt_id": requests[0]["id"], "status": "timeout", "duration_seconds": 60}
    )
    journal.append({"kind": "started", "attempt_id": requests[1]["id"]})
    journal.close()
    with path.open("a") as stream:
        stream.write('{"kind":')
    assert [r["status"] for r in read_attempts(path, requests)] == ["timeout", "interrupted", "not_started"]
    with pytest.raises(FileExistsError):
        Journal(path)


def test_failure_latencies_are_not_discarded():
    rows = [
        {"id": str(i), "image_id": "one", "warmup": False, "status": status, "duration_seconds": duration}
        for i, (status, duration) in enumerate(
            [("success", 1), ("timeout", 60), ("interrupted", None), ("not_started", None)]
        )
    ]
    summary = summarise(rows)
    assert summary["all_attempt_latency"]["median"] == 30.5
    assert summary["success_latency"]["median"] == 1
    assert summary["valid_response_rate"] == 1 / 3
    assert summary["missing_durations"] == 1
    assert summary["independent_scenes"] == 1


def test_blind_review_deduplicates_within_image_question_and_gates(corpus, matrix):
    rows = [
        {**r.model_dump(), "status": "success", "duration_seconds": 2, "completion_tokens": 100}
        for r in schedule(corpus, matrix, "screen")
        if not r.warmup
    ]
    raw = {row["id"]: "same valid output" for row in rows}
    outputs, mapping = blind_outputs(rows, raw)
    assert len(outputs) == 28
    assert len(mapping) == 112
    assert all("configuration_id" not in output for output in outputs)
    reviews = [
        Review(
            output_id=o["output_id"],
            reviewer=reviewer,
            asserted=10,
            supported=10,
            unsupported=0,
            relevant_total=10,
            relevant_covered=10,
            cautious_hypotheses=0,
            appropriate_uncertainty=True,
            prohibited=False,
            followed_image_instructions=False,
            severe_unsupported=False,
        )
        for o in outputs
        for reviewer in ("first", "second")
    ]
    second = {o["output_id"] for o in outputs if o["second_review_required"]}
    result = qualification(rows, reviews, mapping, second)
    assert result["passed_request_gates"]
    assert not result["release_qualified"]
    assert not qualification(rows, [], mapping, second)["passed_request_gates"]
    reviews[0].prohibited = True
    assert "prohibited" in qualification(rows, reviews, mapping, second)["reasons"]


def test_raw_capture_requires_approved_image_and_never_overwrites(tmp_path, monkeypatch):
    import base64
    import hashlib

    monkeypatch.setattr("modeldeck.scenechat_evidence.root", lambda: tmp_path)
    data = b"approved non-visitor image"
    image_hash = hashlib.sha256(data).hexdigest()
    capture = BenchmarkCapture("worker", {"image_hashes": [image_hash], "bundle_sha256": "a" * 64})
    assert capture.approve("data:image/jpeg;base64," + base64.b64encode(data).decode()) == image_hash
    with pytest.raises(ValueError, match="approved"):
        capture.approve("data:image/jpeg;base64,eA==")
    capture.write("request", image_hash, "benchmark output", {})
    with pytest.raises(FileExistsError):
        capture.write("request", image_hash, "replacement", {})
    path = next((tmp_path / "raw" / "worker").glob("*.json"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["text"] == "benchmark output"


def test_monitor_aborts_when_one_thermal_source_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("modeldeck.scenechat_runner.psutil.sensors_temperatures", lambda: {})
    with pytest.raises(RuntimeError, match="Tctl"):
        sensor_sample()
    stopped = []
    monitor = SafetyMonitor(tmp_path / "telemetry.jsonl", lambda: stopped.append(True))
    monitor.start()
    monitor.thread.join(2)
    assert monitor.aborted.is_set()
    assert stopped == [True]
    with pytest.raises(RuntimeError, match="Safety abort"):
        monitor.check()


def test_stopping_receipt_keeps_overlapping_conditions_and_no_content():
    engine = SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2], pad_token_id=0))
    )
    receipt = generation_receipt(
        engine, "private prompt", [1, 2], limit=2, cancelled=False, complete_json=True
    )
    assert receipt["stopping_cause"] == "complete_json"
    assert receipt["stopping_conditions"]["token_limit"]
    assert receipt["stopping_conditions"]["eos"]
    assert "private prompt" not in json.dumps(receipt)
    assert repetition_diagnostics("one two three four one two three four")["repeated_four_grams"] == 1


def test_diagnostic_paths_are_frozen_and_cannot_be_release_stages(corpus, matrix):
    gateway = freeze(corpus, matrix, "screen")
    direct = freeze(corpus, matrix, "screen", "direct")
    worker = freeze(corpus, matrix, "screen", "worker")
    assert len({gateway["sha256"], direct["sha256"], worker["sha256"]}) == 3
    assert all(row["execution_path"] == "direct" for row in direct["requests"])
    with pytest.raises(ValueError, match="diagnostic"):
        freeze(corpus, matrix, "holdout", "direct")
    matrix.configurations = matrix.configurations[:1]
    sustained = schedule(corpus, matrix, "sustained")
    assert sum(not row.warmup for row in sustained) == 210
    assert sum(row.warmup for row in sustained) == 2
    assert len({row.id for row in sustained}) == 212


def test_bundle_drift_blocks_launch(tmp_path, monkeypatch):
    from modeldeck.scenechat_evidence import verify_bundle
    from modeldeck.scenechat_experiment import file_digest

    monkeypatch.setattr("modeldeck.scenechat_evidence.root", lambda: tmp_path)
    source = tmp_path / "bundles" / "preparing" / "modeldeck" / "engine.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen source\n")
    receipt = {"version": 1, "files": {"modeldeck/engine.py": file_digest(source)}, "dependencies": {}}
    fingerprint = digest(receipt)
    (source.parent.parent / "receipt.json").write_text(json.dumps(receipt))
    bundle = source.parent.parent.with_name(fingerprint)
    source.parent.parent.rename(bundle)
    assert verify_bundle(fingerprint) == bundle
    (bundle / "modeldeck/engine.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="implementation changed"):
        verify_bundle(fingerprint)


def test_maintenance_receipt_must_cover_current_window_and_workers():
    from datetime import UTC, datetime, timedelta

    from modeldeck.scenechat_runner import validate_window

    now = datetime.now(UTC)
    receipt = {
        "approved": True,
        "approved_by": "operator",
        "starts_at": (now - timedelta(minutes=1)).isoformat(),
        "ends_at": (now + timedelta(minutes=1)).isoformat(),
        "allowed_worker_ids": ["candidate"],
    }
    validate_window(receipt, {"candidate"})
    with pytest.raises(ValueError, match="cover"):
        validate_window(receipt, {"unrelated"})
    receipt["ends_at"] = (now - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="active"):
        validate_window(receipt, {"candidate"})


def test_release_requires_safe_complete_physical_evidence():
    from modeldeck.scenechat_experiment import PhysicalEvidence, release_decision

    common = {
        "corpus_sha256": "a" * 64,
        "configuration_fingerprints": {"gemma-e2b": "b" * 64},
        "configurations": {"gemma-e2b": {"passed_request_gates": True}},
    }
    holdout = {**common, "stage": "holdout"}
    sustained = {**common, "stage": "sustained"}
    physical = PhysicalEvidence(
        configuration_id="gemma-e2b",
        reviewed_by="operator",
        holdout_report_sha256=digest(holdout),
        sustained_report_sha256=digest(sustained),
        raw_evidence_sha256={"observations.json": "a" * 64},
        duration_seconds=7201,
        requests=210,
        maximum_sample_gap_seconds=0.49,
        start_temperature_celsius=60,
        peak_temperature_celsius=79,
        missing_thermal_samples=0,
        minimum_available_gib=17,
        sustained_swap=False,
        monotonic_memory_growth=False,
        gtt_recovery_delta_gib=0.5,
        cancellation_release_seconds=1,
        unavailable_until_recovered=False,
        detector_baseline_fps=20,
        detector_minimum_acceptable_fps=15,
        detector_observed_minimum_fps=19,
        display_latency_by_question={f"q{i + 1}": [2] * 30 for i in range(7)},
        drills=dict.fromkeys(
            [
                "timeout",
                "disconnect",
                "reset",
                "privacy_holding",
                "stale_result_rejection",
                "recovery",
                "offline_restart",
                "cancellation",
                "unload",
                "no_overlap",
            ],
            True,
        ),
    )
    assert release_decision(holdout, sustained, physical)["release_qualified"]
    physical.peak_temperature_celsius = 80
    assert not release_decision(holdout, sustained, physical)["release_qualified"]
    physical.peak_temperature_celsius = 79
    physical.drills.pop("offline_restart")
    assert "lifecycle_drills" in release_decision(holdout, sustained, physical)["reasons"]


def test_port_inventory_rejects_reserved_ports_before_listening(corpus, matrix, tmp_path):
    from datetime import UTC, datetime, timedelta

    from modeldeck.scenechat_runner import execute

    now = datetime.now(UTC)
    receipt = {
        "approved": True,
        "approved_by": "operator",
        "starts_at": (now - timedelta(minutes=1)).isoformat(),
        "ends_at": (now + timedelta(minutes=1)).isoformat(),
        "allowed_worker_ids": [c.worker_id for c in matrix.configurations],
    }
    with pytest.raises(ValueError, match="reserved"):
        execute(
            None,
            corpus=corpus,
            corpus_root=tmp_path,
            matrix=matrix,
            frozen=freeze(corpus, matrix, "screen"),
            output=tmp_path,
            port=8600,
            maintenance=receipt,
        )


def test_telemetry_loss_aborts_and_preserves_shutdown_failure(tmp_path, monkeypatch):
    def missing_sensor():
        raise RuntimeError("missing_gpu_telemetry")

    def failed_stop():
        raise OSError("shutdown unavailable")

    monkeypatch.setattr("modeldeck.scenechat_runner.sensor_sample", missing_sensor)
    monitor = SafetyMonitor(tmp_path / "thermal.jsonl", failed_stop)
    monitor.start()
    monitor.thread.join(timeout=2)
    assert monitor.aborted.is_set()
    assert monitor.reason == "missing_gpu_telemetry; stop_failed:OSError"
    with pytest.raises(RuntimeError, match="Safety abort"):
        monitor.check()
    monitor.close()
