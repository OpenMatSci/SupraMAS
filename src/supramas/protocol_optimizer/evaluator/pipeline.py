"""Evaluator pipeline skeleton."""

from __future__ import annotations

from typing import Any


class EvaluatorPipeline:
    """Public interface for expert-critic protocol review."""

    def __init__(self, *args, **kwargs):
        self.options = kwargs

    def forward(self, plan: Any, type: str = "idea"):
        return {
            "review_target": type,
            "criteria": ["scientific_consistency", "feasibility", "reproducibility"],
            "status": "interface-only",
        }

