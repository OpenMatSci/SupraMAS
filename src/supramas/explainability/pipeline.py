"""Mechanism-aware traceback skeleton."""

from __future__ import annotations


class TracebackPipeline:
    """Public interface for tracing protocol decisions to evidence."""

    def run(self, protocol: dict, strategy_tree: dict | None = None, mechanisms: dict | None = None) -> dict:
        return {
            "trace_units": ["strategy_tree", "mechanism_library", "protocol_step"],
            "status": "interface-only",
        }

