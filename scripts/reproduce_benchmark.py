#!/usr/bin/env python3
"""Benchmark reproduction helper.

The public repository ships benchmark inputs, labels, and reported outputs.
This helper validates those files and prints a compact summary without making
LLM calls by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def print_summary() -> None:
    summary = REPO_ROOT / "benchmark/results/benchmark_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)
    with summary.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Benchmark summary rows: {len(rows)}")
    for row in rows[:10]:
        name = row.get("model") or row.get("framework") or row.get("Model") or str(row)
        print(name)


def validate_inputs() -> None:
    required = [
        "benchmark/data/benchmark_topics.json",
        "benchmark/data/post_cutoff_paper_ideas.json",
        "benchmark/data/idea_foresight_label_strategies.json",
        "benchmark/data/protocol_reference_labels.json",
        "benchmark/results/benchmark_summary.csv",
    ]
    for rel in required:
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
    topics = load_json(REPO_ROOT / "benchmark/data/benchmark_topics.json")
    print(f"Benchmark topics: {len(topics)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize released benchmark artifacts.")
    parser.add_argument("--summary-only", action="store_true", help="Print reported benchmark summary only.")
    parser.add_argument("--validate", action="store_true", help="Validate required input/result files.")
    args = parser.parse_args()

    if args.validate or not args.summary_only:
        validate_inputs()
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

