from __future__ import annotations

from typing import Literal

VisualTokenBudget = Literal[70, 140, 280, 560, 1120]

ALLOWED_VISUAL_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)
DEFAULT_VISUAL_TOKEN_BUDGET = 280
GEMMA4_PATCH_SIZE = 16
GEMMA4_POOLING_KERNEL_SIZE = 3
