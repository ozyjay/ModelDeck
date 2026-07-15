from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerState(StrEnum):
    DISCOVERED = "discovered"
    STOPPED = "stopped"
    VALIDATING = "validating"
    STARTING = "starting"
    LOADING = "loading"
    WARMING = "warming"
    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    FAILED = "failed"
    ORPHANED = "orphaned"
    INCOMPATIBLE = "incompatible"


class GenerationFamily(StrEnum):
    AUTOREGRESSIVE = "autoregressive"
    TEXT_DIFFUSION = "text-diffusion"
    EMBEDDING = "embedding"
    VISION_LANGUAGE = "vision-language"
    VISION_DETECTION = "vision-detection"
    FEATURE_EXTRACTION = "feature-extraction"


class LifecycleClass(StrEnum):
    RESIDENT = "resident"
    ON_DEMAND = "on-demand"
    EXCLUSIVE = "exclusive"


class CapabilitySet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat: bool | Literal["compatibility-only"] = False
    completions: bool = False
    streaming: bool = True
    cancellation: bool = True
    logits: bool | Literal["model-specific"] = False
    top_k_trace: bool = False
    hidden_states: bool | Literal["optional"] = False
    iterative_refinement: bool = False
    intermediate_frames: bool = False
    seeded_generation: bool = False
    image_input: bool = False
    structured_output: bool = False


class WorkerHealth(BaseModel):
    protocol_version: Literal["1"] = "1"
    worker_id: str
    runtime: str
    generation_family: GenerationFamily
    state: WorkerState
    model_id: str
    model_revision: str
    device: str = "cpu"
    device_name: str = "Mock device"
    rocm_version: str | None = None
    ready: bool


class WorkerEvent(BaseModel):
    worker_id: str
    state: WorkerState
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)
