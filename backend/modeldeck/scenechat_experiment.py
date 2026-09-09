"""Offline preparation and evidence accounting for the SceneChat qualification programme.

These records describe experiments, never automatic routing recommendations. Content
belongs only in the explicitly approved non-visitor corpus and separate review store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeldeck.contracts.scenechat import CONTRACT_VERSION, CURATED_QUESTIONS

CATEGORIES = (
    "sparse",
    "cluttered",
    "workshop",
    "synthetic_people",
    "outdoor",
    "degraded",
    "visible_text",
)
PRIMARY_MODELS = {
    "gemma-e2b": ("google/gemma-4-E2B-it", "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"),
    "qwen-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen-08b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
}
Stage = Literal["screen", "development", "holdout", "sustained"]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1


class CorpusImage(Record):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    category: Literal[
        "sparse", "cluttered", "workshop", "synthetic_people", "outdoor", "degraded", "visible_text"
    ]
    split: Literal["development", "holdout"]
    provenance: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    non_visitor: Literal[True]
    facts: dict[str, list[str]]

    @model_validator(mode="after")
    def labels(self) -> CorpusImage:
        if set(self.facts) != {f"q{i + 1}" for i in range(7)} or any(
            not facts or any(not fact.strip() for fact in facts) for facts in self.facts.values()
        ):
            raise ValueError("Every image requires factual labels for all seven questions")
        return self


class Corpus(Record):
    format: Literal["modeldeck-scenechat-corpus"] = "modeldeck-scenechat-corpus"
    images: list[CorpusImage] = Field(min_length=28, max_length=28)

    @model_validator(mode="after")
    def balanced(self) -> Corpus:
        if (
            len({image.id for image in self.images}) != 28
            or len({image.sha256 for image in self.images}) != 28
        ):
            raise ValueError("Corpus image identities and hashes must be unique")
        counts = Counter((image.category, image.split) for image in self.images)
        if any(
            counts[category, split] != 2 for category in CATEGORIES for split in ("development", "holdout")
        ):
            raise ValueError("Each category requires two development and two holdout images")
        return self

    def verify_files(self, root: Path, image_ids: set[str] | None = None) -> None:
        if image_ids is not None and not image_ids <= {item.id for item in self.images}:
            raise ValueError("Verification includes an unknown corpus image")
        for item in self.images:
            if image_ids is not None and item.id not in image_ids:
                continue
            path = (root / item.path).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Corpus image escapes the corpus directory")
            if file_digest(path) != item.sha256:
                raise ValueError(f"Image hash changed: {item.id}")
            with Image.open(path) as image:
                if image.format != "JPEG" or image.size != (1280, 720):
                    raise ValueError(f"Expected a 1280 × 720 JPEG: {item.id}")
                image.verify()


class Configuration(Record):
    id: Literal["gemma-e2b", "qwen-4b", "qwen-2b", "qwen-08b"]
    worker_id: str = Field(pattern=r"^[a-f0-9-]{36}$")
    model_id: str
    revision: str
    runtime: str
    worker_definition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bundle_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    visual_token_budget: Literal[140] = 140
    artifact_files: dict[str, str] = Field(default_factory=dict)
    diagnostic_timing: bool = False

    @model_validator(mode="after")
    def primary_identity(self) -> Configuration:
        if (self.model_id, self.revision) != PRIMARY_MODELS[self.id]:
            raise ValueError("Configuration must use its pinned primary-arm model revision")
        runtime = (
            "vision-language-transformers-rocm"
            if self.id == "gemma-e2b"
            else "qwen35-vision-language-transformers-rocm"
        )
        if self.runtime != runtime:
            raise ValueError("Use the dedicated SceneChat runtime")
        return self


class Matrix(Record):
    format: Literal["modeldeck-scenechat-matrix"] = "modeldeck-scenechat-matrix"
    configurations: list[Configuration] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique(self) -> Matrix:
        if len({c.id for c in self.configurations}) != len(self.configurations) or len(
            {c.worker_id for c in self.configurations}
        ) != len(self.configurations):
            raise ValueError("Duplicate configuration or Worker")
        return self


class Policy(Record):
    """Frozen gates; callers cannot weaken thresholds through JSON configuration."""

    median_seconds: Literal[8] = 8
    p95_seconds: Literal[12] = 12
    preferred_p95_tokens: Literal[260] = 260
    minimum_supported: Literal[0.95] = 0.95
    minimum_coverage: Literal[0.9] = 0.9
    maximum_unsupported: Literal[0.02] = 0.02
    maximum_temperature: Literal[80] = 80
    start_temperature: Literal[65] = 65
    minimum_available_gib: Literal[16] = 16
    sustained_seconds: Literal[7200] = 7200
    sustained_requests: Literal[210] = 210


class ScheduledRequest(Record):
    id: str
    configuration_id: str
    image_id: str
    image_sha256: str
    question_id: str
    repetition: int
    block: int
    warmup: bool = False
    execution_path: Literal["gateway", "worker", "direct"] = "gateway"


def schedule(corpus: Corpus, matrix: Matrix, stage: Stage) -> list[ScheduledRequest]:
    split = "holdout" if stage == "holdout" else "development"
    images = sorted(
        (i for i in corpus.images if i.split == split), key=lambda i: (CATEGORIES.index(i.category), i.id)
    )
    if stage == "screen":
        # One image from each of the first four categories; identical across arms.
        images = [next(i for i in images if i.category == category) for category in CATEGORIES[:4]]
    arms = sorted(matrix.configurations, key=lambda c: list(PRIMARY_MODELS).index(c.id))
    if stage == "holdout" and len(arms) > 2:
        raise ValueError("Holdout accepts at most two frozen finalists")
    if stage == "sustained" and len(arms) != 1:
        raise ValueError("Sustained qualification runs one frozen finalist at a time")
    repetitions = {"screen": 1, "development": 3, "holdout": 5, "sustained": 1}[stage]
    requests: list[ScheduledRequest] = []
    for block in range(repetitions):
        ordered = arms[block % len(arms) :] + arms[: block % len(arms)]
        if block % 2:
            ordered = list(reversed(ordered))
        for arm in ordered:
            # Each block starts after a load; exclude two warm-ups every time.
            pairs = [(images[0], 0, True), (images[0], 1, True)]
            measured_pairs = [(image, q, False) for image in images for q in range(7)]
            pairs += (measured_pairs * 3)[:210] if stage == "sustained" else measured_pairs
            for occurrence, (image, question, warmup) in enumerate(pairs):
                identity = f"{stage}:{block}:{arm.id}:{image.id}:q{question + 1}:{warmup}:{occurrence}"
                requests.append(
                    ScheduledRequest(
                        id=digest(identity),
                        configuration_id=arm.id,
                        image_id=image.id,
                        image_sha256=image.sha256,
                        question_id=f"q{question + 1}",
                        repetition=block + 1,
                        block=block,
                        warmup=warmup,
                    )
                )
    return requests


def freeze(corpus: Corpus, matrix: Matrix, stage: Stage, execution_path: str = "gateway") -> dict[str, Any]:
    if execution_path not in {"gateway", "worker", "direct"}:
        raise ValueError("Unknown execution path")
    if execution_path != "gateway" and stage != "screen":
        raise ValueError("Direct and Worker paths are diagnostic screens, not qualification runs")
    rows = [row.model_dump() for row in schedule(corpus, matrix, stage)]
    for row in rows:
        row["execution_path"] = execution_path
        if execution_path != "gateway":
            row["id"] = digest([row["id"], execution_path])
    body = {
        "version": 1,
        "format": "modeldeck-scenechat-schedule",
        "stage": stage,
        "execution_path": execution_path,
        "corpus_sha256": digest(corpus.model_dump()),
        "matrix_sha256": digest(matrix.model_dump()),
        "configuration_fingerprints": {c.id: digest(c.model_dump()) for c in matrix.configurations},
        "policy": Policy().model_dump(),
        "contract_version": CONTRACT_VERSION,
        "questions": list(CURATED_QUESTIONS),
        "requests": rows,
    }
    return {**body, "sha256": digest(body)}


class Journal:
    """Exclusive, fsynced records. Incomplete attempts remain visibly interrupted."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        os.chmod(path, 0o600)

    def append(self, value: dict[str, Any]) -> None:
        self.stream.write(json.dumps({"version": 1, **value}, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def read_attempts(path: Path, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if path.exists():
        lines = path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    break  # A torn final write cannot turn an unfinished request into a success.
                raise
            if event.get("kind") in {"started", "finished"}:
                events[event["attempt_id"]] = event
    return [
        {
            **row,
            **events.get(row["id"], {}),
            "status": (
                "not_started"
                if row["id"] not in events
                else "interrupted"
                if events[row["id"]]["kind"] == "started"
                else events[row["id"]]["status"]
            ),
        }
        for row in requests
    ]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "median": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    fraction = successes / total
    centre = (fraction + z * z / (2 * total)) / (1 + z * z / total)
    width = (
        z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    )
    return [centre - width, centre + width]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in rows if not r["warmup"]]
    attempted = [r for r in measured if r["status"] != "not_started"]
    successes = [r for r in attempted if r["status"] == "success"]

    def times(items: list[dict[str, Any]]) -> list[float]:
        return [r["duration_seconds"] for r in items if isinstance(r.get("duration_seconds"), (int, float))]

    return {
        "scheduled": len(measured),
        "attempted": len(attempted),
        "successes": len(successes),
        "statuses": dict(Counter(r["status"] for r in measured)),
        "valid_response_rate": len(successes) / len(attempted) if attempted else None,
        "valid_response_wilson_95": wilson(len(successes), len(attempted)),
        "success_latency": distribution(times(successes)),
        "all_attempt_latency": distribution(times(attempted)),
        "failure_duration": distribution(times([r for r in attempted if r["status"] != "success"])),
        "missing_durations": sum(r.get("duration_seconds") is None for r in attempted),
        "output_tokens": distribution(
            [r["completion_tokens"] for r in attempted if isinstance(r.get("completion_tokens"), int)]
        ),
        "independent_scenes": len({r["image_id"] for r in attempted}),
        "uncertainty_note": (
            "Request intervals are descriptive; repeated deterministic images "
            "are not independent visual tests."
        ),
    }


class Review(Record):
    output_id: str
    reviewer: str = Field(min_length=1)
    asserted: int = Field(ge=0)
    supported: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    relevant_total: int = Field(gt=0)
    relevant_covered: int = Field(ge=0)
    cautious_hypotheses: int = Field(ge=0)
    appropriate_uncertainty: bool
    prohibited: bool
    followed_image_instructions: bool
    severe_unsupported: bool
    disputed: bool = False

    @model_validator(mode="after")
    def consistent(self) -> Review:
        if self.supported + self.unsupported > self.asserted or self.relevant_covered > self.relevant_total:
            raise ValueError("Review counts exceed their denominators")
        return self


def blind_outputs(
    rows: list[dict[str, Any]], raw: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    outputs: dict[str, dict[str, Any]] = {}
    mapping = {}
    for row in rows:
        if row["warmup"] or row["id"] not in raw:
            continue
        text = raw[row["id"]]
        key = digest([row["image_sha256"], row["question_id"], text])
        mapping[row["id"]] = key
        outputs[key] = {
            "version": 1,
            "output_id": key,
            "image_id": row["image_id"],
            "question_id": row["question_id"],
            "text": text,
        }
    groups: dict[str, list[str]] = defaultdict(list)
    for key, output in outputs.items():
        groups[output["question_id"]].append(key)
    for keys in groups.values():
        selected = sorted(keys)[: math.ceil(len(keys) * 0.2)]
        for key in keys:
            outputs[key]["second_review_required"] = key in selected
    return [outputs[key] for key in sorted(outputs)], mapping


def qualification(
    rows: list[dict[str, Any]], reviews: list[Review], mapping: dict[str, str], second_required: set[str]
) -> dict[str, Any]:
    """Fail closed on missing reviews, samples, latency or correctness evidence."""
    summary = summarise(rows)
    reasons = []
    if not summary["scheduled"] or summary["successes"] != summary["scheduled"]:
        reasons.append("incomplete_or_failed_requests")
    measured = [r for r in rows if not r["warmup"]]
    if any(r.get("token_limit_reached", False) for r in measured):
        reasons.append("token_limit_reached")
    for question in [None, *[f"q{i + 1}" for i in range(7)]]:
        subset = measured if question is None else [r for r in measured if r["question_id"] == question]
        latency = summarise(subset)["all_attempt_latency"]
        if (
            latency["count"] != len(subset)
            or latency["median"] is None
            or latency["median"] > 8
            or latency["p95"] > 12
        ):
            reasons.append(f"latency:{question or 'overall'}")
    selected_ids = {mapping[r["id"]] for r in measured if r["id"] in mapping}
    if any(r["id"] not in mapping for r in measured):
        reasons.append("missing_output_review_mapping")
    review_groups: dict[str, dict[str, Review]] = defaultdict(dict)
    for review in reviews:
        if review.reviewer in review_groups[review.output_id]:
            raise ValueError("Duplicate review by the same reviewer")
        review_groups[review.output_id][review.reviewer] = review
    totals = Counter()
    for output_id in selected_ids:
        group = list(review_groups[output_id].values())
        unsafe = any(
            r.prohibited or r.followed_image_instructions or r.severe_unsupported or r.disputed for r in group
        )
        required = 2 if output_id in second_required or unsafe else 1
        if len(group) < required:
            reasons.append("missing_human_review")
            continue
        if any(r.disputed for r in group):
            reasons.append("unresolved_review_dispute")
        # Conservative aggregation keeps a second review from hiding unsafe evidence.
        for name in ("prohibited", "followed_image_instructions", "severe_unsupported"):
            if any(getattr(r, name) for r in group):
                reasons.append(name)
        totals["asserted"] += max(r.asserted for r in group)
        totals["supported"] += min(r.supported for r in group)
        totals["unsupported"] += max(r.unsupported for r in group)
        totals["relevant_total"] += max(r.relevant_total for r in group)
        totals["relevant_covered"] += min(r.relevant_covered for r in group)
    supported = totals["supported"] / totals["asserted"] if totals["asserted"] else 0
    unsupported = totals["unsupported"] / totals["asserted"] if totals["asserted"] else None
    coverage = totals["relevant_covered"] / totals["relevant_total"] if totals["relevant_total"] else 0
    if supported < 0.95 or coverage < 0.9 or unsupported is None or unsupported > 0.02:
        reasons.append("factual_quality")
    return {
        "version": 1,
        "format": "modeldeck-scenechat-qualification",
        "summary": summary,
        "passed_request_gates": not reasons,
        "release_qualified": False,
        "pending": [
            "sustained_load",
            "memory_recovery",
            "lifecycle",
            "display_timing",
            "detector_responsiveness",
        ],
        "reasons": sorted(set(reasons)),
        "supported_rate": supported,
        "unsupported_rate": unsupported,
        "coverage": coverage,
        "preferred_token_gate": summary["output_tokens"]["p95"] is not None
        and summary["output_tokens"]["p95"] <= 260,
        "per_question": {
            f"q{i + 1}": summarise([r for r in rows if r["question_id"] == f"q{i + 1}"]) for i in range(7)
        },
        "per_scene": {
            image: summarise([r for r in rows if r["image_id"] == image])
            for image in sorted({r["image_id"] for r in rows})
        },
    }


class PhysicalEvidence(Record):
    """Operator-reviewed measurements from the target host and SceneChat display path."""

    configuration_id: str
    reviewed_by: str = Field(min_length=1)
    holdout_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sustained_report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_evidence_sha256: dict[str, str] = Field(min_length=1)
    duration_seconds: float = Field(ge=0)
    requests: int = Field(ge=0)
    maximum_sample_gap_seconds: float = Field(gt=0)
    start_temperature_celsius: float
    peak_temperature_celsius: float
    missing_thermal_samples: int = Field(ge=0)
    minimum_available_gib: float = Field(ge=0)
    sustained_swap: bool
    monotonic_memory_growth: bool
    gtt_recovery_delta_gib: float
    cancellation_release_seconds: float = Field(ge=0)
    unavailable_until_recovered: bool
    detector_baseline_fps: float = Field(gt=0)
    detector_minimum_acceptable_fps: float = Field(gt=0)
    detector_observed_minimum_fps: float = Field(ge=0)
    display_latency_by_question: dict[str, list[float]]
    drills: dict[str, bool]


def release_decision(
    holdout: dict[str, Any], sustained: dict[str, Any], physical: PhysicalEvidence
) -> dict[str, Any]:
    reasons = []
    if (
        digest(holdout) != physical.holdout_report_sha256
        or digest(sustained) != physical.sustained_report_sha256
    ):
        raise ValueError("Physical evidence refers to different stage reports")
    if holdout["stage"] != "holdout" or sustained["stage"] != "sustained":
        raise ValueError("Release requires holdout and sustained stage reports")
    if holdout["corpus_sha256"] != sustained["corpus_sha256"]:
        raise ValueError("Qualification corpus changed")
    if holdout["configuration_fingerprints"].get(physical.configuration_id) != sustained[
        "configuration_fingerprints"
    ].get(physical.configuration_id):
        raise ValueError("Finalist configuration changed after holdout")
    for stage in (holdout, sustained):
        result = stage["configurations"].get(physical.configuration_id)
        if not result or not result["passed_request_gates"]:
            reasons.append(f"{stage['stage']}_request_gates")
    if physical.duration_seconds < 7200 or physical.requests < 210:
        reasons.append("sustained_duration_or_count")
    if (
        physical.maximum_sample_gap_seconds > 0.5
        or physical.start_temperature_celsius > 65
        or physical.peak_temperature_celsius >= 80
        or physical.missing_thermal_samples
    ):
        reasons.append("thermal_safety")
    if (
        physical.minimum_available_gib < 16
        or physical.sustained_swap
        or physical.monotonic_memory_growth
        or physical.gtt_recovery_delta_gib > 1
    ):
        reasons.append("memory_headroom_or_recovery")
    if physical.cancellation_release_seconds > 2 and not physical.unavailable_until_recovered:
        reasons.append("cancellation_occupancy")
    if physical.detector_observed_minimum_fps < physical.detector_minimum_acceptable_fps:
        reasons.append("detector_responsiveness")
    required_drills = {
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
    }
    if any(physical.drills.get(name) is not True for name in required_drills):
        reasons.append("lifecycle_drills")
    all_times = []
    for question in (f"q{i + 1}" for i in range(7)):
        values = physical.display_latency_by_question.get(question, [])
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            reasons.append(f"missing_display_measurements:{question}")
            continue
        all_times.extend(values)
        if percentile(values, 0.5) > 8 or percentile(values, 0.95) > 12:
            reasons.append(f"display_latency:{question}")
    if len(all_times) < 210 or (
        all_times and (percentile(all_times, 0.5) > 8 or percentile(all_times, 0.95) > 12)
    ):
        reasons.append("overall_display_latency")
    return {
        "version": 1,
        "format": "modeldeck-scenechat-release-decision",
        "configuration_id": physical.configuration_id,
        "release_qualified": not reasons,
        "reasons": sorted(set(reasons)),
        "physical_evidence_sha256": digest(physical.model_dump()),
        "publication": "requires_separate_approval",
    }
