"""Semantic literature-search skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Result:
    title: str
    source: str = ""
    summary: str = ""


class SemanticSearcher:
    """Public interface for external literature search."""

    def search(self, query: str, limit: int = 10) -> list[Result]:
        return [Result(title="Search interface placeholder", summary=query[:200])][:limit]


SementicSearcher = SemanticSearcher

