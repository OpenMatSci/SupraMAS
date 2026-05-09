"""Top-level SupraMAS agent orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from supramas.llm import ExternalLLMPlaceholder, get_llm


@dataclass
class ResearchRequest:
    """Input contract for a SupraMAS research-planning run."""

    topic: str
    anchor_paper_path: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowArtifacts:
    """High-level outputs produced by the full private implementation."""

    strategy_tree: dict[str, Any]
    idea: dict[str, Any]
    protocol: dict[str, Any]
    optimization: dict[str, Any]
    traceback: dict[str, Any]


class DeepResearchAgent:
    """Coordinator for the five-stage SupraMAS workflow.

    The public class documents the orchestration boundary. The detailed private
    prompts, expert-critic loops, retrieval policies, and refinement logic are
    intentionally omitted from this reviewer-facing release.
    """

    def __init__(self, llm: ExternalLLMPlaceholder | None = None, **kwargs):
        self.llm = llm or get_llm()
        self.options = kwargs

    def run(self, request: ResearchRequest) -> WorkflowArtifacts:
        return WorkflowArtifacts(
            strategy_tree={"stage": "strategy_mining", "topic": request.topic},
            idea={"stage": "idea_innovation", "status": "interface-only"},
            protocol={"stage": "experiment_construction", "status": "interface-only"},
            optimization={"stage": "mechanism_informed_optimization", "status": "interface-only"},
            traceback={"stage": "explainability_tracking", "status": "interface-only"},
        )

    def idea_innovation(self, policy_tree: dict[str, Any]):
        initial = {"stage": "idea_innovation", "input": "strategy_tree", "status": "interface-only"}
        refined = {"stage": "idea_refinement", "status": "interface-only"}
        return initial, refined

    def experiment_construction(self, idea: dict[str, Any]):
        return {"stage": "experiment_construction", "input": "idea", "status": "interface-only"}


def get_llms(*args, **kwargs) -> ExternalLLMPlaceholder:
    """Compatibility helper returning the public LLM placeholder."""

    return get_llm()

