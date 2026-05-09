"""Command-line entry point for the public SupraMAS framework skeleton."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TOPIC = "Improve the bearing current capacity of superconducting strips through pinning"


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def export_env(config: dict[str, Any]) -> None:
    for key, value in config.items():
        if key == "agents" or value in ("", None):
            continue
        os.environ[key] = str(value)


def framework_overview(topic: str, output_dir: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_type": "public_framework_skeleton",
        "topic": topic,
        "output_dir": output_dir,
        "config_keys": sorted(config.keys()),
        "workflow": [
            "strategy_mining",
            "idea_innovation",
            "experiment_construction",
            "real_world_verification",
            "mechanism_informed_optimization",
            "explainability_tracking",
        ],
        "note": (
            "This public skeleton exposes module boundaries and data flow only. "
            "Detailed agent prompts and production implementation are omitted."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the public SupraMAS framework skeleton.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Research topic to display in the workflow overview.")
    parser.add_argument("--anchor-paper-path", default=None, help="Retained for interface compatibility.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for optional overview output.")
    parser.add_argument("--max-tree-depth", type=int, default=3, help="Retained for interface compatibility.")
    parser.add_argument("--config", default=None, help="YAML config path. Example: configs/supramas.example.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the framework overview without writing files.")
    parser.add_argument("--write-overview", action="store_true", help="Write outputs/framework_overview.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    export_env(config)
    overview = framework_overview(args.topic, args.output_dir, config)

    if args.write_overview and not args.dry_run:
        output_path = Path(args.output_dir) / "framework_overview.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(json.dumps(overview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

