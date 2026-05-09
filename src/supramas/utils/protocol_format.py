"""Protocol-format helpers for public artifacts."""

from __future__ import annotations

import json


def protocol_to_markdown(protocol: dict) -> str:
    return "```json\n" + json.dumps(protocol, ensure_ascii=False, indent=2) + "\n```"

