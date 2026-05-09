"""Small public utility helpers."""

from __future__ import annotations

import json
from pathlib import Path


def get_content_between_a_b(start_tag: str, end_tag: str, text: str) -> str:
    start = text.find(start_tag)
    if start == -1:
        return ""
    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return ""
    return text[start + len(start_tag):end].strip()


def extract(text: str, type: str, hard: bool = True) -> str:
    value = get_content_between_a_b(f"<{type}>", f"</{type}>", text or "")
    return value or (text if hard else "")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

