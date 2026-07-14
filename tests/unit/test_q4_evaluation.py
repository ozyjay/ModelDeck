from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_evaluator() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts/evaluate_diffusiongemma_q4.py"
    spec = importlib.util.spec_from_file_location("modeldeck_q4_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_q4_evaluation_constraint_checks_are_accent_insensitive() -> None:
    evaluator = load_evaluator()
    spec = evaluator.PromptSpec(
        id="translation",
        prompt="translate",
        required_groups=(("bibliothèque",), ("neuf",)),
        minimum_words=5,
    )

    result = evaluator.evaluate_constraints(
        spec,
        "La bibliotheque ouvre a neuf heures demain matin.",
    )

    assert result["passed"] is True
    assert result["required_group_results"] == [True, True]


def test_q4_evaluation_uses_default_q4_and_explicit_bf16_aliases() -> None:
    evaluator = load_evaluator()

    assert evaluator.Q4_ALIAS == "text-diffusion"
    assert evaluator.BF16_ALIAS == "text-diffusion-bf16"


def test_q4_evaluation_creative_constraint_accepts_rain_synonyms() -> None:
    evaluator = load_evaluator()
    spec = next(item for item in evaluator.DEFAULT_PROMPTS if item.id == "creative-scene")

    result = evaluator.evaluate_constraints(
        spec,
        (
            "The robot stepped outside as a sudden downpour drummed on its metal shell, "
            "turning every reflected streetlight into a trembling constellation. It raised "
            "one careful hand, watched silver drops gather across its palm, and laughed "
            "when thunder answered from the dark clouds."
        ),
    )

    assert result["passed"] is True
    assert result["required_group_results"] == [True, True]

    petrichor_result = evaluator.evaluate_constraints(
        spec,
        (
            "The robot watched one drop strike its shoulder, then another bead of water "
            "gather on its palm. Its sensors identified petrichor as the grey sky opened, "
            "and it stood outside studying the cool cascade instead of seeking shelter."
        ),
    )

    assert petrichor_result["passed"] is True
    assert petrichor_result["required_group_results"] == [True, True]


def test_q4_evaluation_arithmetic_prompt_requests_answer_first() -> None:
    evaluator = load_evaluator()
    spec = next(item for item in evaluator.DEFAULT_PROMPTS if item.id == "arithmetic-reasoning")

    assert "numerical average speed first" in spec.prompt
    result = evaluator.evaluate_constraints(
        spec,
        (
            "72 kilometres per hour. Divide the 180-kilometre distance by the "
            "2.5-hour travel time to calculate the train's average speed."
        ),
    )

    assert result["passed"] is True


def test_q4_evaluation_phase_summary_tracks_contracts_and_memory_range() -> None:
    evaluator = load_evaluator()
    results = [
        {
            "wall_seconds": 7.0,
            "contract_passed": True,
            "constraint": {"passed": True},
            "quality": {"passed": True},
        },
        {
            "wall_seconds": 9.0,
            "contract_passed": False,
            "constraint": {"passed": True},
            "quality": {"passed": False},
        },
    ]

    summary = evaluator.phase_summary(results, [18_000, 19_500])

    assert summary["runs"] == 2
    assert summary["contract_passes"] == 1
    assert summary["constraint_pass_rate"] == 1.0
    assert summary["quality_pass_rate"] == 0.5
    assert summary["median_wall_seconds"] == 8.0
    assert summary["memory_allocated_range_bytes"] == 1_500


def test_q4_evaluation_rejects_reasoning_leaks_repetition_and_truncation() -> None:
    evaluator = load_evaluator()

    clean = evaluator.evaluate_output_quality(
        "Pi is the ratio of circumference to diameter.",
        {"finish_reason": "stop"},
    )
    leaked = evaluator.evaluate_output_quality(
        "thought\nPi is approximately 3.14159.",
        {"finish_reason": "stop"},
    )
    repeated = evaluator.evaluate_output_quality(
        "Pi is is is a ratio.",
        {"finish_reason": "length"},
    )

    assert clean["passed"] is True
    assert leaked["checks"]["no_reasoning_channel_leak"] is False
    assert repeated["checks"]["not_length_limited"] is False
    assert repeated["checks"]["no_excessive_consecutive_repetition"] is False
