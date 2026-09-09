"""Local evidence commands, invoked through the PowerShell experiment tools wrapper."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from modeldeck.scenechat_evidence import freeze_installation, root, verify_bundle
from modeldeck.scenechat_experiment import (
    PRIMARY_MODELS,
    Configuration,
    Corpus,
    Matrix,
    PhysicalEvidence,
    Policy,
    Review,
    blind_outputs,
    digest,
    file_digest,
    qualification,
    read_attempts,
    release_decision,
)


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_frozen(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop("sha256")
    if digest(value) != expected:
        raise ValueError("Frozen schedule digest mismatch")
    Policy.model_validate(value["policy"])
    return {**value, "sha256": expected}


def export_review(schedule_path: Path, journal_path: Path, matrix_path: Path, output: Path) -> None:
    frozen = load_frozen(schedule_path)
    matrix = Matrix.model_validate_json(matrix_path.read_text())
    if digest(matrix.model_dump()) != frozen["matrix_sha256"]:
        raise ValueError("Matrix changed after freezing")
    rows = read_attempts(journal_path, frozen["requests"])
    configurations = {c.id: c for c in matrix.configurations}
    raw = {}
    for row in rows:
        if "request_id" not in row:
            continue
        configuration = configurations[row["configuration_id"]]
        path = root() / "raw" / configuration.worker_id / f"{digest(row['request_id'])}.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if (
            value["request_id"] != row["request_id"]
            or value["image_sha256"] != row["image_sha256"]
            or value["bundle_sha256"] != configuration.bundle_sha256
        ):
            raise ValueError("Raw evidence identity mismatch")
        raw[row["id"]] = value["text"]
    outputs, mapping = blind_outputs(rows, raw)
    write_new(output / "blind-review.json", {"version": 1, "outputs": outputs})
    write_new(
        output / "private-mapping.json",
        {
            "version": 1,
            "mapping": mapping,
            "schedule_sha256": frozen["sha256"],
            "journal_sha256": digest(journal_path.read_text()),
        },
    )
    write_new(output / "review.schema.json", Review.model_json_schema())


def report(
    schedule_path: Path, journal_path: Path, reviews_path: Path, review_dir: Path, output: Path
) -> None:
    frozen = load_frozen(schedule_path)
    private = json.loads((review_dir / "private-mapping.json").read_text())
    if private["schedule_sha256"] != frozen["sha256"] or private["journal_sha256"] != digest(
        journal_path.read_text()
    ):
        raise ValueError("Review mapping refers to different evidence")
    rows = read_attempts(journal_path, frozen["requests"])
    reviews = [Review.model_validate(r) for r in json.loads(reviews_path.read_text())]
    outputs = json.loads((review_dir / "blind-review.json").read_text())["outputs"]
    second = {o["output_id"] for o in outputs if o["second_review_required"]}
    configurations = {}
    for configuration_id in sorted({r["configuration_id"] for r in rows}):
        configurations[configuration_id] = qualification(
            [r for r in rows if r["configuration_id"] == configuration_id],
            reviews,
            private["mapping"],
            second,
        )
    write_new(
        output,
        {
            "version": 1,
            "format": "modeldeck-scenechat-stage-report",
            "stage": frozen["stage"],
            "schedule_sha256": frozen["sha256"],
            "matrix_sha256": frozen["matrix_sha256"],
            "corpus_sha256": frozen["corpus_sha256"],
            "configuration_fingerprints": frozen["configuration_fingerprints"],
            "configurations": configurations,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schemas = commands.add_parser("schemas")
    schemas.add_argument("--output", type=Path, required=True)
    definitions = commands.add_parser("prepare-workers")
    definitions.add_argument("--output", type=Path, required=True)
    matrix_command = commands.add_parser("matrix")
    matrix_command.add_argument("--workers", type=Path, required=True)
    matrix_command.add_argument("--bundle", required=True)
    matrix_command.add_argument("--output", type=Path, required=True)
    commands.add_parser("freeze-installation")
    register = commands.add_parser("register")
    register.add_argument("--matrix", type=Path, required=True)
    register.add_argument("--schedule", type=Path, required=True)
    register.add_argument("--diagnostic-timing", action="store_true")
    register.add_argument("--replace-stopped", action="store_true")
    for name in ("export-review", "report"):
        command = commands.add_parser(name)
        command.add_argument("--schedule", type=Path, required=True)
        command.add_argument("--journal", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "export-review":
            command.add_argument("--matrix", type=Path, required=True)
        else:
            command.add_argument("--reviews", type=Path, required=True)
            command.add_argument("--review-directory", type=Path, required=True)
    finalists = commands.add_parser("finalists")
    finalists.add_argument("--report", type=Path, required=True)
    finalists.add_argument("--matrix", type=Path, required=True)
    finalists.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("release-decision")
    release.add_argument("--holdout", type=Path, required=True)
    release.add_argument("--sustained", type=Path, required=True)
    release.add_argument("--physical-evidence", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "schemas":
        for model in (Corpus, Matrix, Review, Policy, PhysicalEvidence):
            write_new(arguments.output / f"{model.__name__.lower()}.schema.json", model.model_json_schema())
    elif arguments.command == "prepare-workers":
        from modeldeck.v2_api import WorkerCreateRequest

        for name, (model_id, revision) in PRIMARY_MODELS.items():
            payload = WorkerCreateRequest(
                name=f"scenechat-evaluation-{name}",
                model_id=model_id,
                revision=revision,
                dtype="bfloat16",
                lifecycle="on-demand",
                context_length=8192,
                maximum_new_tokens=1024,
                visual_token_budget=140,
                runtime_template_id="scenechat-gemma4" if name == "gemma-e2b" else "scenechat-qwen35",
                capability_id="scene-analysis",
            )
            write_new(arguments.output / f"{name}.json", payload.model_dump(exclude_none=True))
    elif arguments.command == "matrix":
        from modeldeck.domain import WorkerDefinition
        from modeldeck.scenechat_runner import validate_worker

        verify_bundle(arguments.bundle)
        supplied = json.loads(arguments.workers.read_text())
        configurations = []
        for name, worker in supplied.items():
            definition = WorkerDefinition.model_validate(
                {k: v for k, v in worker.items() if k in WorkerDefinition.model_fields}
            )
            snapshot = (
                Path(definition.settings["cache_root"])
                / ("models--" + definition.model_id.replace("/", "--"))
                / "snapshots"
                / definition.revision
            )
            inventory = {
                str(p.relative_to(snapshot)): file_digest(p) for p in snapshot.rglob("*") if p.is_file()
            }
            if not inventory:
                raise ValueError("The pinned local artifact snapshot is missing")
            configuration = Configuration(
                id=name,
                worker_id=definition.id,
                model_id=definition.model_id,
                revision=definition.revision,
                runtime=definition.runtime,
                worker_definition_sha256=digest(definition.model_dump(mode="json")),
                bundle_sha256=arguments.bundle,
                artifact_files=inventory,
            )
            validate_worker(worker, configuration)
            configurations.append(configuration)
        write_new(arguments.output, Matrix(configurations=configurations).model_dump())
    elif arguments.command == "freeze-installation":
        destination = root() / "bundles" / f"preparing-{uuid.uuid4()}"
        receipt = freeze_installation(destination)
        final = destination.with_name(digest(receipt))
        if final.exists():
            raise FileExistsError(
                "An identical bundle already exists; retain it and remove the preparation copy after review"
            )
        destination.rename(final)
        print(f"Frozen installation receipt: {final / 'receipt.json'}")
    elif arguments.command == "register":
        matrix = Matrix.model_validate_json(arguments.matrix.read_text())
        frozen = load_frozen(arguments.schedule)
        if digest(matrix.model_dump()) != frozen["matrix_sha256"]:
            raise ValueError("Schedule and matrix differ")
        for configuration in matrix.configurations:
            verify_bundle(configuration.bundle_sha256)
        from urllib.request import urlopen

        with urlopen("http://127.0.0.1:3600/api/workers", timeout=5) as response:
            workers = {w["id"]: w for w in json.load(response)}
        if any(workers.get(c.worker_id, {}).get("state") != "stopped" for c in matrix.configurations):
            raise ValueError("Every registered benchmark Worker must be stopped")
        for configuration in matrix.configurations:
            if configuration.diagnostic_timing != arguments.diagnostic_timing:
                raise ValueError("Timing instrumentation must match the frozen matrix")
            if (
                root() / "workers" / f"{configuration.worker_id}.json"
            ).exists() and not arguments.replace_stopped:
                raise FileExistsError("Use --replace-stopped to archive and replace an existing registration")
        for configuration in matrix.configurations:
            destination = root() / "workers" / f"{configuration.worker_id}.json"
            value = {
                "version": 1,
                "worker_id": configuration.worker_id,
                "bundle_sha256": configuration.bundle_sha256,
                "image_hashes": sorted({r["image_sha256"] for r in frozen["requests"]}),
                "diagnostic_timing": arguments.diagnostic_timing,
            }
            if configuration.diagnostic_timing != arguments.diagnostic_timing:
                raise ValueError("Timing instrumentation must match the frozen matrix")
            if destination.exists() and arguments.replace_stopped:
                old = json.loads(destination.read_text())
                archive = root() / "registration-history" / f"{configuration.worker_id}-{uuid.uuid4()}.json"
                write_new(archive, old)
                temporary = destination.with_suffix(f".{uuid.uuid4()}.tmp")
                write_new(temporary, value)
                temporary.replace(destination)
            else:
                write_new(destination, value)
    elif arguments.command == "export-review":
        export_review(arguments.schedule, arguments.journal, arguments.matrix, arguments.output)
    elif arguments.command == "report":
        report(
            arguments.schedule,
            arguments.journal,
            arguments.reviews,
            arguments.review_directory,
            arguments.output,
        )
    elif arguments.command == "finalists":
        value = json.loads(arguments.report.read_text())
        matrix = Matrix.model_validate_json(arguments.matrix.read_text())
        if value["stage"] != "development" or value["matrix_sha256"] != digest(matrix.model_dump()):
            raise ValueError("Finalists require the matching development matrix and report")
        results = value["configurations"]
        passing = [c for c in matrix.configurations if results[c.id]["passed_request_gates"]]
        passing.sort(
            key=lambda c: (
                results[c.id]["summary"]["all_attempt_latency"]["p95"],
                results[c.id]["summary"]["all_attempt_latency"]["median"],
                results[c.id]["unsupported_rate"],
            )
        )
        if not passing:
            write_new(
                arguments.output,
                {"version": 1, "decision": "no_qualifier", "source_report_sha256": digest(value)},
            )
        else:
            write_new(arguments.output, Matrix(configurations=passing[:2]).model_dump())
    elif arguments.command == "release-decision":
        physical = PhysicalEvidence.model_validate_json(arguments.physical_evidence.read_text())
        for name, expected in physical.raw_evidence_sha256.items():
            if file_digest(arguments.physical_evidence.parent / name) != expected:
                raise ValueError("Physical evidence file changed")
        write_new(
            arguments.output,
            release_decision(
                json.loads(arguments.holdout.read_text()),
                json.loads(arguments.sustained.read_text()),
                physical,
            ),
        )


if __name__ == "__main__":
    main()
