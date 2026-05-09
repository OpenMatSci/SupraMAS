"""Idea-agent interface skeletons."""

from __future__ import annotations


class IdeaAgent:
    def process(self, payload: dict):
        return {"stage": "idea_generation", "payload_keys": sorted(payload.keys()), "status": "interface-only"}


class ExpertAgent(IdeaAgent):
    pass


class RetrievalAgent(IdeaAgent):
    pass


class KnowledgeBaseAgent(IdeaAgent):
    pass


class ReviewAgent(IdeaAgent):
    pass

