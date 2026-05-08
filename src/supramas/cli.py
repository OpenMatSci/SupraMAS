"""Command-line entry point for SupraMAS."""

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
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def export_env(config: dict[str, Any]) -> None:
    for key, value in config.items():
        if key == "agents" or value in (None, ""):
            continue
        os.environ[key] = str(value)
    # Backward compatibility for the historical misspelling.
    if "SEMANTIC_SEARCH_API_KEY" in os.environ and "SEMENTIC_SEARCH_API_KEY" not in os.environ:
        os.environ["SEMENTIC_SEARCH_API_KEY"] = os.environ["SEMANTIC_SEARCH_API_KEY"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SupraMAS research-planning workflow.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Research topic to plan for.")
    parser.add_argument("--anchor-paper-path", default=None, help="Optional anchor-paper PDF path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated workflow outputs.")
    parser.add_argument("--max-tree-depth", type=int, default=3, help="Maximum strategy-tree depth.")
    parser.add_argument("--config", default=None, help="YAML config path. Example: configs/supramas.example.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments/config without calling LLMs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    export_env(config)

    topic_path = Path(args.output_dir) / args.topic.replace("/", "_")
    if args.dry_run:
        print(json.dumps({
            "topic": args.topic,
            "anchor_paper_path": args.anchor_paper_path,
            "output_path": str(topic_path),
            "max_tree_depth": args.max_tree_depth,
            "config_keys": sorted(config.keys()),
        }, ensure_ascii=False, indent=2))
        return 0

    from termcolor import cprint

    from supramas.agents import DeepResearchAgent, get_llms
    from supramas.conversation_logger import ConversationLogger
    from supramas.strategy_tree.pipeline import PolicyTreePipline

    topic_path.mkdir(parents=True, exist_ok=True)
    conversation_logger = ConversationLogger(log_file=topic_path / "conversation.json")
    llm = get_llms(conversation_logger)

    runtime_args = argparse.Namespace(
        topic=args.topic,
        anchor_paper_path=args.anchor_paper_path,
        topic_path=topic_path,
        max_tree_depth=args.max_tree_depth,
        config=config,
        header_dir=args.output_dir,
    )

    deep_research_agent = DeepResearchAgent(llm=llm, **vars(runtime_args))
    policy_tree_agent = PolicyTreePipline(runtime_args, llm)

    cprint(f"Initialized SupraMAS topic: {args.topic}", "green")
    policy_tree = policy_tree_agent.forward()
    initial_idea, refined_idea = deep_research_agent.idea_innovation(policy_tree)

    with (topic_path / "initial_idea.json").open("w", encoding="utf-8") as fh:
        json.dump(initial_idea, fh, ensure_ascii=False, indent=2)
    with (topic_path / "refined_idea.json").open("w", encoding="utf-8") as fh:
        json.dump(refined_idea, fh, ensure_ascii=False, indent=2)

    refined_experiment = deep_research_agent.experiment_construction(idea=refined_idea)
    with (topic_path / "refined_experiment.json").open("w", encoding="utf-8") as fh:
        json.dump(refined_experiment, fh, ensure_ascii=False, indent=2)

    cprint(f"SupraMAS workflow complete: {topic_path}", "green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

