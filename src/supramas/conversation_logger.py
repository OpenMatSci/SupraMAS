"""Lightweight conversation-log shape for the public skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ConversationLogger:
    """Append-only in-memory logger with optional JSON export."""

    log_file: str | Path | None = None
    records: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str, **metadata) -> None:
        self.records.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "role": role,
                "content": content,
                "metadata": metadata,
            }
        )

    def save(self) -> None:
        if not self.log_file:
            return
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

