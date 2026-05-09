"""Minimal result-format CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from supramas.utils import load_json
from supramas.utils.protocol_format import protocol_to_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a public protocol JSON file to Markdown.")
    parser.add_argument("json_path")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.json_path).with_suffix(".md")
    output.write_text(protocol_to_markdown(load_json(args.json_path)), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

