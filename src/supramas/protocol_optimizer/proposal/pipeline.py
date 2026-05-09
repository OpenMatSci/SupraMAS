"""Proposal-generation skeleton."""

from __future__ import annotations

from typing import Any


class ProposalPipeline:
    """Public interface for idea/protocol proposal generation."""

    def __init__(self, config: dict | None = None, retrival: Any | None = None, type: str = "idea", **kwargs):
        self.config = config or {}
        self.retrieval = retrival
        self.type = type
        self.options = kwargs

    def idea_generate(self, policy_tree: dict, topic: str | None = None):
        return (
            {"topic": topic, "source": "strategy_tree", "status": "interface-only"},
            {"topic": topic, "stage": "refined_idea", "status": "interface-only"},
        )

    def experiment_generate(self, idea: dict):
        return {"source": "idea", "stage": "experiment_plan", "status": "interface-only"}

