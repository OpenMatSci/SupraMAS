"""Vector-retrieval interface placeholder."""

from __future__ import annotations


class VectorRetrieval:
    """Public placeholder for the strategy-library retriever."""

    def __call__(self, queries: list[str]):
        return [{"query": query, "records": [], "status": "interface-only"} for query in queries]

