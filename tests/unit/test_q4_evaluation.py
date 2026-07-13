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


def test_q4_evaluation_phase_summary_tracks_contracts_and_memory_range() -> None:
    evaluator = load_evaluator()
    results = [
        {
            "wall_seconds": 7.0,
            "contract_passed": True,
            "constraint": {"passed": True},
        },
        {
            "wall_seconds": 9.0,
            "contract_passed": False,
            "constraint": {"passed": True},
        },
    ]

    summary = evaluator.phase_summary(results, [18_000, 19_500])

    assert summary["runs"] == 2
    assert summary["contract_passes"] == 1
    assert summary["constraint_pass_rate"] == 1.0
    assert summary["median_wall_seconds"] == 8.0
    assert summary["memory_allocated_range_bytes"] == 1_500
