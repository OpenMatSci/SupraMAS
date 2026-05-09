"""Experiment-agent interface skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentPlan:
    idea: dict | None = None
    draft: str = ""
    steps: list[dict] = field(default_factory=list)
    final_plan: str = ""

    def to_dict(self) -> dict:
        return {"idea": self.idea, "draft": self.draft, "steps": self.steps, "final_plan": self.final_plan}


class DraftGenerationAgent:
    def process(self, payload: dict):
        return {"stage": "draft_generation", "status": "interface-only"}


class ExperimentDecompositionAgent(DraftGenerationAgent):
    pass


class RetrievalAgent(DraftGenerationAgent):
    pass


class KnowledgeBaseAgent(DraftGenerationAgent):
    pass


class ExpertAgent(DraftGenerationAgent):
    pass


class ReviewAgent(DraftGenerationAgent):
    pass


class PlanWritingAgent(DraftGenerationAgent):
    pass

