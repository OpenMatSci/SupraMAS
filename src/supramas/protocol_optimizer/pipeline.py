"""Protocol-optimization skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProtocolDraft:
    objective: str
    sub_tasks: list[str] = field(default_factory=list)
    review_state: str = "interface-only"


class AGDPipeline:
    """Public interface for Agent-Guided Design protocol generation."""

    stages = ("draft_generation", "experiment_decomposition", "knowledge_grounding", "expert_review", "plan_writing")

    def __init__(self, type: str = "experiment", config: dict | None = None, **kwargs):
        self.type = type
        self.config = config or {}
        self.options = kwargs

    def forward(self, idea: dict[str, Any] | None = None, type: str | None = None, policy_tree: dict | None = None):
        mode = type or self.type
        return {
            "mode": mode,
            "stages": list(self.stages),
            "draft": ProtocolDraft(
                objective="convert a mechanism-informed idea into an executable protocol",
                sub_tasks=["materials", "growth", "characterization", "feedback"],
            ).__dict__,
            "status": "interface-only",
        }


AGDPipline = AGDPipeline

