"""Approved, isolated direct-runtime diagnostic process; never a public endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

from modeldeck.contracts.scenechat import (
    CURATED_QUESTIONS,
    ModelOutputValidationError,
    canonicalise_model_output,
)
from modeldeck.scenechat_evidence import BenchmarkCapture
from modeldeck.scenechat_experiment import Corpus, Journal, Matrix, file_digest, freeze
from modeldeck.scenechat_runner import SafetyMonitor, validate_window, validate_worker
from modeldeck.workers.qwen35_worker import TransformersQwen35Engine
from modeldeck.workers.scenechat_worker import (
    EngineConfig,
    TransformersSceneChatEngine,
    _decode_image_with_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "corpus", "matrix", "schedule", "maintenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    corpus = Corpus.model_validate_json(args.corpus.read_text())
    matrix = Matrix.model_validate_json(args.matrix.read_text())
    if len(matrix.configurations) != 1:
        raise ValueError("Direct diagnostics require exactly one configuration")
    frozen = freeze(corpus, matrix, "screen", "direct")
    if json.loads(args.schedule.read_text()) != frozen:
        raise ValueError("Frozen direct-runtime schedule changed")
    corpus.verify_files(args.corpus.parent, {row["image_id"] for row in frozen["requests"]})
    configuration = matrix.configurations[0]
    worker = json.loads(args.worker.read_text())
    validate_worker(worker, configuration)
    maintenance = json.loads(args.maintenance.read_text())
    validate_window(maintenance, {configuration.worker_id})
    capture = BenchmarkCapture.for_worker(configuration.worker_id)
    if capture is None or capture.value["bundle_sha256"] != configuration.bundle_sha256:
        raise ValueError("Direct diagnostics require the registered frozen installation")
    if capture.value["diagnostic_timing"] != configuration.diagnostic_timing:
        raise ValueError("Diagnostic timing differs from the frozen matrix")
    if set(capture.value["image_hashes"]) != {r["image_sha256"] for r in frozen["requests"]}:
        raise ValueError("Direct capture allowlist must match the exact diagnostic corpus")
    cache = Path(worker["settings"]["cache_root"])
    snapshot = (
        cache
        / ("models--" + configuration.model_id.replace("/", "--"))
        / "snapshots"
        / configuration.revision
    )
    inventory = {str(p.relative_to(snapshot)): file_digest(p) for p in snapshot.rglob("*") if p.is_file()}
    if not inventory or inventory != configuration.artifact_files:
        raise ValueError("Direct-runtime artifact inventory changed")
    config = EngineConfig(
        model_id=configuration.model_id,
        revision=configuration.revision,
        cache_root=cache,
        maximum_new_tokens=1024,
        visual_token_budget=140,
        diagnostic_timing=configuration.diagnostic_timing,
    )
    engine = (
        TransformersSceneChatEngine(config)
        if configuration.id == "gemma-e2b"
        else TransformersQwen35Engine(config)
    )
    journal = Journal(args.worker.parent / "attempts.jsonl")
    journal.append({"kind": "run_receipt", "schedule_sha256": frozen["sha256"], "execution_path": "direct"})
    cancellation = threading.Event()

    def abort() -> None:
        cancellation.set()
        # The process is an isolated physical diagnostic. A wedged kernel must not
        # outlive the safety guard; the fsynced started record remains interrupted.
        timer = threading.Timer(2, lambda: os._exit(2))
        timer.daemon = True
        timer.start()

    monitor = SafetyMonitor(args.worker.parent / "telemetry.jsonl", abort)
    monitor.worker_pid = os.getpid()
    monitor.start()
    try:
        deadline = time.monotonic() + 2
        while monitor.latest is None and not monitor.aborted.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        monitor.cooldown()
        loaded_at = time.monotonic()
        engine.load()
        monitor.check()
        monitor.worker_metrics = engine.memory_metrics
        journal.append(
            {
                "kind": "load",
                "cold_load_seconds": time.monotonic() - loaded_at,
                "resolved_identity": engine.runtime_details,
            }
        )
        images = {image.id: image for image in corpus.images}
        for row in frozen["requests"]:
            validate_window(maintenance, {configuration.worker_id})
            monitor.check()
            image = images[row["image_id"]]
            path = args.corpus.parent / image.path
            if file_digest(path) != row["image_sha256"]:
                raise ValueError("Diagnostic image changed")
            request_id = str(uuid.uuid4())
            cancellation.clear()
            started = time.monotonic()
            journal.append({"kind": "started", "attempt_id": row["id"], "request_id": request_id})
            timeout = threading.Timer(60, abort)
            timeout.start()
            diagnostics = {}
            status = "internal_error"
            try:
                prepared, _ = _decode_image_with_metadata(
                    "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
                )
                decode_seconds = time.monotonic() - started
                try:
                    result = engine.generate(
                        image=prepared,
                        question=CURATED_QUESTIONS[int(row["question_id"][1:]) - 1],
                        max_tokens=1024,
                        cancellation=cancellation,
                    )
                finally:
                    prepared.close()
                capture.write(request_id, row["image_sha256"], result.text, result.diagnostics)
                diagnostics = {
                    **result.diagnostics,
                    "decode_seconds": decode_seconds,
                    "preprocessing_seconds": result.preprocessing_seconds,
                    "inference_seconds": result.inference_seconds,
                    "completion_tokens": result.completion_tokens,
                    "visual_tokens": result.visual_tokens,
                    "token_limit_reached": result.completion_tokens >= 1024,
                }
                validation_started = time.monotonic()
                try:
                    canonicalise_model_output(result.text)
                    status = "success"
                except ModelOutputValidationError as error:
                    status = "invalid_model_output"
                    diagnostics["validator_category"] = error.category
                diagnostics["validation_seconds"] = time.monotonic() - validation_started
                if result.completion_tokens >= 1024:
                    status = "token_limit"
                if cancellation.is_set():
                    status = "cancelled_or_timeout"
            finally:
                timeout.cancel()
                journal.append(
                    {
                        **diagnostics,
                        "kind": "finished",
                        "attempt_id": row["id"],
                        "request_id": request_id,
                        "status": status,
                        "duration_seconds": time.monotonic() - started,
                    }
                )
            monitor.check()
            if cancellation.is_set():
                raise TimeoutError("Direct generation exceeded its deadline")
    finally:
        monitor.worker_metrics = None
        failures = []
        try:
            engine.unload()
        except Exception as error:
            failures.append(f"unload:{type(error).__name__}")
        try:
            monitor.close()
        except Exception as error:
            failures.append(f"monitor:{type(error).__name__}")
        journal.append({"kind": "cleanup", "safety_abort": monitor.reason, "failures": failures})
        journal.close()
        if failures:
            raise RuntimeError("Direct-runtime cleanup failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
