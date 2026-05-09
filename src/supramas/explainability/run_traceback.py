"""CLI helper for the public traceback skeleton."""

from __future__ import annotations

import json

from supramas.explainability import TracebackPipeline


def main() -> int:
    print(json.dumps(TracebackPipeline().run(protocol={}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

