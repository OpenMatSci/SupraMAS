"""Protocol-template retrieval placeholder."""

from __future__ import annotations


class VectorRetrieval:
    def __call__(self, queries: list[str]):
        return [{"query": query, "templates": [], "status": "interface-only"} for query in queries]

