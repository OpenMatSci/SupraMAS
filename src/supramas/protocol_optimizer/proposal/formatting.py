"""Formatting helpers for public protocol artifacts."""

from __future__ import annotations

import json


def to_markdown(protocol: dict) -> str:
    return "```json\n" + json.dumps(protocol, ensure_ascii=False, indent=2) + "\n```"

